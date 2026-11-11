"""Programs for tests of generated vector-unit functions (`km.sfpu_fn`): the same
arithmetic over up to four tensors once as a chain of tile ops and once as one
generated function, each a metal-level program run with km.Launcher."""

import kohakutt.dsl.metal as km

f = km.sfpu

# The backend's complex multiply of one face, in vector-unit instructions.
HAND_COMPLEX_MUL = km.SfpuFn(
    "hand_complex_mul",
    """
    inline void hand_complex_mul(uint32_t in0, uint32_t in1, uint32_t out) {
        using namespace ckernel;
        constexpr uint32_t L0 = p_sfpu::LREG0, L1 = p_sfpu::LREG1, L2 = p_sfpu::LREG2;
        constexpr uint32_t L3 = p_sfpu::LREG3, L4 = p_sfpu::LREG4, L5 = p_sfpu::LREG5;
        constexpr uint32_t L6 = p_sfpu::LREG6, L7 = p_sfpu::LREG7;
        constexpr uint32_t zero = p_sfpu::LCONST_0, one = p_sfpu::LCONST_1;
        const uint32_t a = in0 * 64, b = in1 * 64, o = out * 64;
        for (uint32_t group = 0; group < 4; ++group) {
            TT_SFPLOAD(L0, 0, ADDR_MOD_7, a);
            TT_SFPLOAD(L1, 0, ADDR_MOD_7, a + 2);
            TT_SFPLOAD(L2, 0, ADDR_MOD_7, b);
            TT_SFPLOAD(L3, 0, ADDR_MOD_7, b + 2);
            TTI_SFPMUL(L0, L2, zero, L4, 0);
            TTI_SFPMUL(L1, L3, zero, L5, 1);
            TTI_SFPMUL(L0, L3, zero, L6, 0);
            TTI_SFPMUL(L1, L2, zero, L7, 0);
            TTI_SFPADD(one, L4, L5, L4, 0);
            TTI_SFPADD(one, L6, L7, L6, 0);
            TT_SFPSTORE(L4, 0, ADDR_MOD_7, o);
            TT_SFPSTORE(L6, 0, ADDR_MOD_7, o + 2);
            TTI_INCRWC(0, 4, 0, 0);
        }
    }
    """,
)


# ktt_flash's exp in its own instructions (sfpu.hpp exp_pair_body): out = 2^(in0 A +
# b - 639), b at the even columns of in1 (one value per row), A = FLASH_A.
FLASH_A = 0.18033688011112042
FLASH_EXP_RAW = km.SfpuFn(
    "flash_exp_raw",
    """
    inline void flash_exp_raw(uint32_t in0, uint32_t in1, uint32_t out) {
        using namespace ckernel;
        constexpr uint32_t L0 = p_sfpu::LREG0, L1 = p_sfpu::LREG1, L2 = p_sfpu::LREG2;
        constexpr uint32_t L3 = p_sfpu::LREG3, L4 = p_sfpu::LREG4, L5 = p_sfpu::LREG5;
        constexpr uint32_t L6 = p_sfpu::LREG6, L7 = p_sfpu::LREG7;
        constexpr uint32_t C512 = 11, CA = 12, CB2 = 13, CB1 = 14;
        auto load = [](uint32_t lreg, float x) {
            const uint32_t bits = __builtin_bit_cast(uint32_t, x);
            TT_SFPLOADI(lreg, sfpi::SFPLOADI_MOD0_UPPER, bits >> 16);
            TT_SFPLOADI(lreg, sfpi::SFPLOADI_MOD0_LOWER, bits & 0xffff);
        };
        auto set = [&](uint32_t c, float x) {
            load(L0, x);
            TT_SFPCONFIG(0, c, 0);
            TTI_SFPNOP;
        };
        set(C512, 512.0f);
        set(CA, A_VALUE);
        set(CB2, 0.33718854f);
        set(CB1, -0.01674134f);
        load(L6, 0.68127801f);
        const uint32_t a = in0 * 64, b = in1 * 64, o = out * 64;
        for (uint32_t group = 0; group < 4; ++group) {
            TT_SFPLOAD(L7, 0, ADDR_MOD_7, b);
            TT_SFPLOAD(L0, 0, ADDR_MOD_7, a);
            TT_SFPLOAD(L3, 0, ADDR_MOD_7, a + 2);
            TTI_SFPMAD(L0, CA, L7, L0, 0);
            TTI_SFPMAD(L3, CA, L7, L3, 0);
            TTI_SFPSWAP(0, L0, C512, 1);
            TTI_SFPSWAP(0, L3, C512, 1);
            TTI_SFPSHFT(9, L0, L0, 5);
            TTI_SFPSHFT(9, L3, L3, 5);
            TTI_SFPSETEXP(127, L0, L1, 1);
            TTI_SFPSETEXP(127, L3, L4, 1);
            TTI_SFPMAD(L1, CB2, CB1, L2, 0);
            TTI_SFPMAD(L4, CB2, CB1, L5, 0);
            TTI_SFPMAD(L2, L1, L6, L2, 0);
            TTI_SFPMAD(L5, L4, L6, L5, 0);
            TTI_SFPSETEXP(0, L2, L0, 2);
            TTI_SFPSETEXP(0, L5, L3, 2);
            TT_SFPSTORE(L0, 0, ADDR_MOD_7, o);
            TT_SFPSTORE(L3, 0, ADDR_MOD_7, o + 2);
            TTI_INCRWC(0, 4, 0, 0);
        }
    }
    """.replace("A_VALUE", f"{FLASH_A.hex()}f"),
    ("lreg11",),
)


def _ops(*steps):
    """A body running tile ops in turn: (name, slots...) or (name, slots, value)."""

    def body(c):
        for name, *rest in steps:
            value = rest.pop() if isinstance(rest[-1], float) else None
            c.init(name)
            c.op(name, *rest, value=value)

    return body


def _function(name, result):
    """A body running the generated function of `result` into slot 0."""
    fn = km.sfpu_fn(name, result)

    def body(c):
        c.init(fn.init)
        c.run(fn, *range(fn.operands), out=0)

    body.fn = fn
    return body


def _hand(fn):
    def body(c):
        c.init("mul")
        c.sfpu_binary(fn, 0, 1, 0)

    return body


def cases():
    """{name: (inputs, body of tile ops or hand-written code, body of one generated
    function)}: each pair computes the same value into slot 0."""
    a, b, c, d = f.operands(4)
    return {
        "flash_exp": (
            2,
            _hand(FLASH_EXP_RAW),
            _function("flash_exp_fn", f.flash_exp(a, FLASH_A, b)),
        ),
        "silu": (1, _ops(("silu", 0)), _function("silu_fn", f.silu(a))),
        "swiglu": (
            2,
            _ops(("silu", 0), ("mul", 0, 1, 0)),
            _function("swiglu_fn", f.silu(a) * b),
        ),
        "complex_mul": (
            2,
            _hand(HAND_COMPLEX_MUL),
            _function(
                "complex_mul_fn",
                {
                    "even": a.even * b.even - a.odd * b.odd,
                    "odd": a.even * b.odd + a.odd * b.even,
                },
            ),
        ),
        "chain6": (
            4,
            _ops(
                ("mul", 0, 1, 0),
                ("add", 0, 2, 0),
                ("sub", 0, 3, 0),
                ("max", 0, 2, 0),
                ("abs", 0),
                ("mul_scalar", 0, 0.375),
            ),
            _function("chain6_fn", abs(f.maximum(a * b + c - d, c)) * 0.375),
        ),
        "gate": (
            4,
            _ops(("mul", 0, 1, 0), ("add", 0, 2, 0), ("sigmoid", 0), ("mul", 0, 3, 0)),
            _function("gate_fn", f.sigmoid(a * b + c) * d),
        ),
        "exp": (1, _ops(("exp", 0)), _function("exp_fn", f.exp(a))),
        "tanh": (1, _ops(("tanh", 0)), _function("tanh_fn", f.tanh(a))),
        "sqrt": (
            1,
            _ops(("abs", 0), ("sqrt", 0)),
            _function("sqrt_fn", f.sqrt(abs(a))),
        ),
        "rsqrt": (
            1,
            _ops(("abs", 0), ("rsqrt", 0)),
            _function("rsqrt_fn", f.rsqrt(abs(a))),
        ),
        "recip": (1, _ops(("recip", 0)), _function("recip_fn", f.recip(a))),
    }


def on_pack(cut=None):
    """A body: slot 1 = slot 1 * 1 on the math thread, then the flash exp of slots 0
    and 1 into slot 0 on the pack thread. `cut` "early" seeds the run before the
    commit."""
    a, b = f.operands(2)
    same = km.sfpu_fn("same_fn", a * 1.0)
    exp = km.sfpu_fn("flash_exp_fn", f.flash_exp(a, FLASH_A, b))

    def body(c):
        c.init(same.init)
        c.run(same, 1, out=1)
        if cut != "early":
            c.commit()
        c.run(exp, 0, 1, out=0, thread="pack")

    return body


def checks():
    """{name: (inputs, body of one generated function, the same on torch tensors)}
    for functions with no chain of tile ops to compare with."""
    import torch

    a, b, c, d = f.operands(4)
    return {
        "select": (
            4,
            _function(
                "select_fn", f.select(a < b, c, d) + f.select(a >= 0.5, 1.0, 0.0)
            ),
            lambda a, b, c, d: torch.where(a < b, c, d) + (a >= 0.5).float(),
        ),
        "mad": (3, _function("mad_fn", f.mad(a, b, c)), lambda a, b, c: a * b + c),
        "clamp": (
            3,
            _function("clamp_fn", f.minimum(f.maximum(f.mad(a, b, c), -1.5), 1.5)),
            lambda a, b, c: (a * b + c).clamp(-1.5, 1.5),
        ),
    }


def fused_program(machine, name, inputs, body, dtype="float32", fp32=True, row=None):
    """`body(c)` over `inputs` tensors of one shape, tile by tile: tensor i is copied
    into register slot i (tensor `row` with its first row repeated), slot 0 is
    packed. Runtime value: `tiles`."""
    p = km.Program(name, machine)
    xs = [p.input(f"x{i}", dtype) for i in range(inputs)]
    out = p.output("out", dtype)
    start, count = p.split(p.runtime("tiles"))
    cbs = [p.cb(i, like=x) for i, x in enumerate(xs)]
    cb_out = p.cb(16, like=out)

    k = p.kernel("dm0")
    with k.loop(count) as i:
        tile = k.let("tile", start + i)
        slots = [k.reserve(cb, 1) for cb in cbs]
        for x, slot in zip(xs, slots, strict=True):
            k.read(x.page(tile), slot)
        k.reads_done()
        for cb in cbs:
            k.push(cb, 1)

    unpack = cbs if dtype == "float32" and fp32 else []
    c = p.kernel("compute", fp32_dest=fp32, unpack_fp32=unpack, approx=False)
    c.startup(cbs[0], cbs[min(1, inputs - 1)], cb_out)
    with c.loop(count):
        for cb in cbs:
            c.wait(cb, 1)
        c.acquire()
        for slot, cb in enumerate(cbs):
            if slot == row:
                c.init("bcast_row", cb)
                c.bcast(cb, 0, slot, "row")
            else:
                c.init("copy", cb)
                c.copy(cb, 0, slot)
        body(c)
        c.reserve(cb_out, 1)
        c.pack(0, cb_out)
        c.release()
        c.push(cb_out, 1)
        for cb in cbs:
            c.pop(cb, 1)

    w = p.kernel("dm1")
    with w.loop(count) as i:
        w.wait(cb_out, 1)
        w.write(cb_out, out.page(start + i))
        w.flushed()
        w.pop(cb_out, 1)
    w.writes_done()
    return p
