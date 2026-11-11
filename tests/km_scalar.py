"""A generated vector-unit function with an fp32 scalar argument and a bf16 rounding
inside its chain: out = bf16(a * scale) + b on fp32 tiles, `scale` a runtime value
passed as its bits (no register slot, no fill)."""

import struct

import kohakutt.dsl.metal as km

f = km.sfpu


def scaled_fn():
    """bf16(a * s0) + b."""
    a, b = f.operands(2)
    return km.sfpu_fn("scaled", f.bf16(a * f.scalar(0)) + b)


def scalar_program(machine):
    """The program above; runtime values `tiles` and `scale` (fp32 bits)."""
    p = km.Program("scalar", machine)
    a, b = p.input("a", "float32"), p.input("b", "float32")
    out = p.output("out", "float32")
    scale = p.runtime("scale")
    start, count = p.split(p.runtime("tiles"))
    cb_a, cb_b, cb_out = p.cb(0, like=a), p.cb(1, like=b), p.cb(16, like=out)

    k = p.kernel("dm0")
    with k.loop(count) as i:
        sa, sb = k.reserve(cb_a, 1), k.reserve(cb_b, 1)
        k.read(a.page(start + i), sa)
        k.read(b.page(start + i), sb)
        k.reads_done()
        k.push(cb_a, 1)
        k.push(cb_b, 1)

    fn = scaled_fn()
    c = p.kernel("compute", fp32_dest=True, unpack_fp32=[cb_a, cb_b], approx=False)
    c.startup(cb_a, cb_b, cb_out)
    with c.loop(count):
        c.wait(cb_a, 1)
        c.wait(cb_b, 1)
        c.acquire()
        c.init("copy", cb_a)
        c.copy(cb_a, 0, 0)
        c.copy(cb_b, 0, 1)
        c.init(fn.init)
        c.run(fn, 0, 1, out=0, values=(scale,))
        c.reserve(cb_out, 1)
        c.pack(0, cb_out)
        c.release()
        c.push(cb_out, 1)
        c.pop(cb_a, 1)
        c.pop(cb_b, 1)

    w = p.kernel("dm1")
    with w.loop(count) as i:
        w.wait(cb_out, 1)
        w.write(cb_out, out.page(start + i))
        w.flushed()
        w.pop(cb_out, 1)
    w.writes_done()
    return p


def run(launcher, a, b, scale):
    """out of the launch for tt fp32 tensors a, b ([R * 32, C * 32]) and a float."""
    import torch

    tiles = a.numel() // 1024
    bits = struct.unpack("<I", struct.pack("<f", float(scale)))[0]
    (out,) = launcher(
        {"a": a, "b": b},
        outputs={"out": (list(a.shape), torch.float32)},
        runtime={"tiles": tiles, "scale": bits},
    )
    return out
