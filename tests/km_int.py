"""int32 tiles through the compute kernel: per tile of a and b (int32) and s (int32
shift counts), out holds one tile per op of OPS_RUN (lanewise int32 ops) and f the
fp32 conversion of a; g holds the int32 conversion of fp32 tensor x."""

import kohakutt.dsl.metal as km

OPS_RUN = (
    "add_int32",
    "sub_int32",
    "mul_int32",
    "eq_int32",
    "lt_int32",
    "ge_int32",
    "and_int32",
    "or_int32",
    "xor_int32",
    "shl_int32",
    "shr_int32",
)
SHIFTS = ("shl_int32", "shr_int32")


def int_program(machine):
    """The program above; runtime value `tiles`."""
    p = km.Program("ints", machine)
    a, b, s = (p.input(n, "int32") for n in ("a", "b", "s"))
    x = p.input("x", "float32")
    out, f_out, g_out = (
        p.output("out", "int32"),
        p.output("f", "float32"),
        p.output("g", "int32"),
    )
    start, count = p.split(p.runtime("tiles"))
    cbs = [p.cb(i, like=t, pages=2) for i, t in enumerate((a, b, s, x))]
    cb_a, cb_b, cb_s, cb_x = cbs
    cb_out = p.cb(16, like=out, pages=len(OPS_RUN))
    cb_f, cb_g = p.cb(17, like=f_out, pages=2), p.cb(18, like=g_out, pages=2)

    k = p.kernel("dm0")
    with k.loop(count) as i:
        for cb, ref in zip(cbs, (a, b, s, x)):
            slot = k.reserve(cb, 1)
            k.read(ref.page(start + i), slot)
            k.reads_done()
            k.push(cb, 1)

    c = p.kernel("compute", fp32_dest=True, unpack_fp32=cbs, exact=True)
    c.startup(cb_a, cb_b, cb_out)
    with c.loop(count):
        for cb in cbs:
            c.wait(cb, 1)
        c.reserve(cb_out, len(OPS_RUN))
        for at, name in enumerate(OPS_RUN):
            c.acquire()
            c.init("copy", cb_a)
            c.copy(cb_a, 0, 0)
            c.copy(cb_s if name in SHIFTS else cb_b, 0, 1)
            c.init(name)
            c.op(name, 0, 1, 2)
            c.commit()
            c.pack(2, cb_out, at=at)
            c.release()
        c.push(cb_out, len(OPS_RUN))
        for cast, src, cb in (
            ("int32_to_fp32", cb_a, cb_f),
            ("fp32_to_int32", cb_x, cb_g),
        ):
            c.acquire()
            c.srca_format(src)
            c.init("copy", src)
            c.copy(src, 0, 0)
            c.srca_format(cb_a)
            c.init(cast)
            c.op(cast, 0)
            c.commit()
            c.reserve(cb, 1)
            c.pack_format(cb)
            c.pack(0, cb)
            c.pack_format(cb_out)
            c.release()
            c.push(cb, 1)
        for cb in cbs:
            c.pop(cb, 1)

    w = p.kernel("dm1")
    with w.loop(count) as i:
        w.wait(cb_out, len(OPS_RUN))
        with w.loop(len(OPS_RUN), name="j") as j:
            w.write(cb_out, out.page((start + i) * len(OPS_RUN) + j), index=j)
        for cb, ref in ((cb_f, f_out), (cb_g, g_out)):
            w.wait(cb, 1)
            w.write(cb, ref.page(start + i))
        w.flushed()
        w.pop(cb_out, len(OPS_RUN))
        w.pop(cb_f, 1)
        w.pop(cb_g, 1)
    w.writes_done()
    return p


def expected(a, b, s):
    """The OPS_RUN results in order, as int32 CPU tensors."""

    return [
        a + b,
        a - b,
        a * b,
        (a == b).int(),
        (a < b).int(),
        (a >= b).int(),
        a & b,
        a | b,
        a ^ b,
        a << s,
        a >> s,
    ]


def run(launcher, a, b, s, x):
    """(out, f, g) of the launch for tt tensors a, b, s (int32) and x (fp32) of one
    shape [32, C * 32]; out is [32, len(OPS_RUN) * C * 32] by tiles."""
    import torch

    tiles = a.numel() // 1024
    rows = a.shape[0]
    return launcher(
        {"a": a, "b": b, "s": s, "x": x},
        outputs={
            "out": ([rows, 32 * len(OPS_RUN) * tiles], torch.int32),
            "f": (list(a.shape), torch.float32),
            "g": (list(a.shape), torch.int32),
        },
        runtime={"tiles": tiles},
    )
