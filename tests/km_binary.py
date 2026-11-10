"""The backend's broadcast binary program (csrc/aten/eltwise/ktt_binary) written at the
metal level, for tests: add / sub / mul / div and the complex multiply of a and a b
broadcast over a's leading dims or rows (torch ops ``kohakutt_test::binary_km`` and
``kohakutt_test::complex_mul_km``)."""

import math

import kohakutt.dsl.metal as km

# a tiles per block beside b in the registers; the launch picks one that divides nr.
BLOCKS = (1, 2, 3)
# Pages of the a and output buffers: a multiple of every block size.
RING = 12
# Fewest a tiles a core takes.
MIN_TILES_PER_CORE = 4

FILL_ROWS = km.CppFn(
    "fill_rows",
    """
    // Copies row 0 of a 32x32 tile (four 16x16 faces) across the tile, in L1.
    template <typename T>
    void fill_rows(uint32_t addr) {
        volatile T* t = reinterpret_cast<volatile T*>(addr);
        constexpr uint32_t face = 256;
        for (uint32_t f = 0; f < 4; ++f) {
            for (uint32_t i = 0; i < 16; ++i) {
                for (uint32_t c = 0; c < 16; ++c) {
                    t[f * face + i * 16 + c] = t[(f % 2) * face + c];
                }
            }
        }
    }
    """,
)


def complex_face(conj_a, conj_b):
    """The vector-unit face function of a * b on interleaved (re, im) fp32 pairs;
    `conj_a` / `conj_b` conjugate that operand."""
    neg_ii, neg_ri, neg_ir = int(conj_a == conj_b), int(conj_b), int(conj_a)
    return km.SfpuFn(
        "complex_mul_face",
        f"""
        // re = ar br -+ ai bi, im = +- ar bi +- ai br: SFPMUL's modifier 1 negates.
        inline void complex_mul_face(uint32_t in0, uint32_t in1, uint32_t out) {{
            using namespace ckernel;
            constexpr uint32_t L0 = p_sfpu::LREG0, L1 = p_sfpu::LREG1;
            constexpr uint32_t L2 = p_sfpu::LREG2, L3 = p_sfpu::LREG3;
            constexpr uint32_t L4 = p_sfpu::LREG4, L5 = p_sfpu::LREG5;
            constexpr uint32_t L6 = p_sfpu::LREG6, L7 = p_sfpu::LREG7;
            constexpr uint32_t zero = p_sfpu::LCONST_0, one = p_sfpu::LCONST_1;
            // A tile is 64 register rows; +2 selects the odd columns.
            const uint32_t a = in0 * 64, b = in1 * 64, o = out * 64;
            for (uint32_t group = 0; group < 4; ++group) {{
                TT_SFPLOAD(L0, 0, ADDR_MOD_7, a);
                TT_SFPLOAD(L1, 0, ADDR_MOD_7, a + 2);
                TT_SFPLOAD(L2, 0, ADDR_MOD_7, b);
                TT_SFPLOAD(L3, 0, ADDR_MOD_7, b + 2);
                TTI_SFPMUL(L0, L2, zero, L4, 0);
                TTI_SFPMUL(L1, L3, zero, L5, {neg_ii});
                TTI_SFPMUL(L0, L3, zero, L6, {neg_ri});
                TTI_SFPMUL(L1, L2, zero, L7, {neg_ir});
                TTI_SFPADD(one, L4, L5, L4, 0);
                TTI_SFPADD(one, L6, L7, L6, 0);
                TT_SFPSTORE(L4, 0, ADDR_MOD_7, o);
                TT_SFPSTORE(L6, 0, ADDR_MOD_7, o + 2);
                TTI_INCRWC(0, 4, 0, 0);
            }}
        }}
        """,
    )


@km.program("binary")
def binary_program(
    p,
    a: km.TensorRef,
    b: km.TensorRef,
    out: km.Out,
    op: km.Static[str] = "add",
    conj_a: km.Static[bool] = False,
    conj_b: km.Static[bool] = False,
):
    """out = a `op` b ("add", "sub", "mul", "div" or "complex"). Groups: one b tile and
    its `nr` a tiles, tile r of group j at `r * sr + j`; `fill` copies b's row 0 down
    its tile; `block` a tiles go through the registers at a time."""
    nr = p.runtime("nr", multiple_of="block")
    sr, fill = p.runtime("sr"), p.runtime("fill")
    block = p.runtime("block", choices=BLOCKS)
    first, count = p.split(p.runtime("groups"), min_per_core=p.runtime("min_groups"))
    cb_a = p.cb(0, like=a, pages=RING)
    cb_b = p.cb(1, like=b, pages=2)
    cb_out = p.cb(16, like=out, pages=RING)
    word = "uint32_t" if a.dtype == "float32" else "uint16_t"
    math_op = "mul" if op == "complex" else op

    @p.dm0
    def reader(k):
        k.helper(FILL_ROWS)
        for g in k.range(count, name="g"):
            j = k.let("j", first + g)
            b_slot = cb_b.reserve(1)
            k.read(b.page(j), b_slot)
            k.reads_done()
            k.inline(
                f"if (fill != 0) {{\n    fill_rows<{word}>(addr);\n}}",
                uses=[km.CBUse(cb_b, written=cb_b.page_bytes)],
                args={"fill": fill, "addr": b_slot},
            )
            cb_b.push(1)
            for r in k.range(nr, step=block, name="r"):
                a_slot = cb_a.reserve(block)
                for i in k.range(block):
                    k.read(a.page((r + i) * sr + j), a_slot[i])
                k.reads_done()
                cb_a.push(block)

    fp32 = [cb_a, cb_b] if a.dtype == "float32" else []

    @p.compute(fp32_dest=True, unpack_fp32=fp32, fidelity="LoFi", approx=False)
    def compute(c):
        c.startup(cb_a, cb_b, cb_out)
        for _ in c.range(count, name="g"):
            cb_b.wait(1)
            for _ in c.range(nr, step=block, name="r"):
                cb_a.wait(block)
                cb_out.reserve(block)
                with c.registers():
                    c.init("copy", cb_a)
                    for i in c.range(block):
                        c.copy(cb_a, i, i)
                    c.init("copy", cb_b)
                    c.copy(cb_b, 0, block)
                    c.init(math_op)
                    for i in c.range(block):
                        if op == "complex":
                            c.sfpu_binary(complex_face(conj_a, conj_b), i, block, i)
                        else:
                            c.binary(op, i, block, i)
                    c.commit()
                    for i in c.range(block):
                        c.pack(i, cb_out)
                cb_a.pop(block)
                cb_out.push(block)
            cb_b.pop(1)

    @p.dm1
    def writer(k):
        for g in k.range(count, name="g"):
            j = k.let("j", first + g)
            for r in k.range(nr, step=block, name="r"):
                cb_out.wait(block)
                for i in k.range(block):
                    k.write(cb_out, out.page((r + i) * sr + j), index=i)
                k.flushed()
                cb_out.pop(block)
        k.writes_done()


def plan(a_shape, b_shape):
    """The runtime values of binary_program for device shapes `a_shape` and
    `b_shape` (b the same, or broadcast over a's leading dims or over its rows), or
    None for other shapes."""
    rank = len(a_shape)
    a = list(a_shape)
    b = [1] * (rank - len(b_shape)) + list(b_shape)
    if rank < 2 or len(b) != rank:
        return None
    ht, wt = math.ceil(a[-2] / 32), math.ceil(a[-1] / 32)
    lead_a, lead_b = math.prod(a[:-2]), math.prod(b[:-2])
    rows, cols = b[-2] == a[-2], b[-1] == a[-1]
    if a == b:
        nr, sr, fill = 1, 0, 0
    elif rows and cols and lead_b == 1 and lead_a > 1:
        nr, sr, fill = lead_a, ht * wt, 0
    elif b[-2] == 1 and cols and lead_b == 1:
        nr, sr, fill = lead_a * ht, wt, 1
    else:
        return None
    block = next(n for n in sorted(BLOCKS, reverse=True) if nr % n == 0)
    return {
        "nr": nr,
        "sr": sr,
        "fill": fill,
        "block": block,
        "groups": lead_a * ht * wt // nr,
        "min_groups": max(1, MIN_TILES_PER_CORE // nr),
    }


def _call(a, b, static):
    values = plan(a.shape, b.shape)
    if values is None or a.dtype != b.dtype:
        raise ValueError("binary_km: b must broadcast over a's leading dims or rows")
    return km.Call(
        inputs={"a": a, "b": b},
        outputs={"out": (list(a.shape), a.dtype)},
        runtime=values,
        static=static,
    )


def register():
    """Registers the two test ops (once per process); returns their namespace."""
    import torch

    if "kohakutt_test::binary_km" not in torch._C._dispatch_get_all_op_names():
        schema = "kohakutt_test::binary_km(Tensor a, Tensor b, str op) -> Tensor"

        @km.op(schema, binary_program)
        def binary(a, b, op):
            return _call(a, b, {"op": op})

        schema = (
            "kohakutt_test::complex_mul_km(Tensor a, Tensor b, bool conj_a, "
            "bool conj_b) -> Tensor"
        )

        @km.op(schema, binary_program)
        def complex_mul(a, b, conj_a, conj_b):
            if a.shape[-1] % 2:
                raise ValueError("complex_mul_km: pairs need an even last dim")
            return _call(a, b, {"op": "complex", "conj_a": conj_a, "conj_b": conj_b})

    return torch.ops.kohakutt_test
