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
