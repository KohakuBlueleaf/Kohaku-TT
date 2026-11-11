"""Exact row means at the metal level with other vector-unit work between the chunks
of a row: per tile row of x ([R * 32, C * 32], fp32), mean_j(scale * x) of each row
into column 0 of one output tile. The statistics are spilled to two register slots
while a generated function scales each chunk, then restored."""

import struct

import kohakutt.dsl.metal as km

f = km.sfpu


def stats_program(machine, spill=True):
    """The program above; `spill` False keeps the statistics in the registers (no
    function between chunks). Runtime values: `rows`, `wt` (tiles per row), `scale`
    and `inv_w` (fp32 bits of the scale and of 1 / W)."""
    p = km.Program("stats_spill" if spill else "stats", machine)
    x, out = p.input("x", "float32"), p.output("out", "float32")
    wt, scale, inv_w = p.runtime("wt"), p.runtime("scale"), p.runtime("inv_w")
    start, count = p.split(p.runtime("rows"))
    cb_x, cb_out = p.cb(0, like=x, pages=2), p.cb(16, like=out, pages=2)

    k = p.kernel("dm0")
    with k.loop(count, name="r") as r, k.loop(wt, name="j") as j:
        slot = k.reserve(cb_x, 1)
        k.read(x.page((start + r) * wt + j), slot)
        k.reads_done()
        k.push(cb_x, 1)

    scaled = f.operands(1)[0] * f.scalar(0)
    fn = km.sfpu_fn("scaled", scaled)
    c = p.kernel("compute", fp32_dest=True, unpack_fp32=[cb_x], exact=True)
    c.startup(cb_x, None, cb_out)
    with c.loop(count, name="r"):
        c.acquire()
        c.init("stats")
        c.stats_clear()
        if spill:
            c.stats_save(2)
        with c.loop(wt, name="j") as j:
            c.wait(cb_x, 1)
            c.init("transpose", cb_x)
            c.transpose(cb_x, 0, 0)
            if spill:
                c.init(fn.init)
                c.run(fn, 0, out=0, values=(scale,))
                c.init("stats")
                c.stats_restore(2)
            with c.when(j.eq(0)):
                c.stats_sum(0, anchor=True)
            with c.otherwise():
                c.stats_sum(0)
            if spill:
                c.stats_save(2)
            c.pop(cb_x, 1)
        if spill:
            c.stats_restore(2)
        c.stats_mean(inv_w)
        c.stats_finalize(2, inv_w)
        c.init("transpose_dest")
        c.op("transpose_dest", 2)
        c.commit()
        c.reserve(cb_out, 1)
        c.pack(2, cb_out)
        c.release()
        c.push(cb_out, 1)

    w = p.kernel("dm1")
    with w.loop(count, name="r") as r:
        w.wait(cb_out, 1)
        w.write(cb_out, out.page(start + r))
        w.flushed()
        w.pop(cb_out, 1)
    w.writes_done()
    return p


def bits(value):
    """The fp32 bits of float `value`."""
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def run(launcher, x, scale):
    """The [R * 32, 32] output (row means in column 0) for tt fp32 x."""
    import torch

    rows, wt = x.shape[0] // 32, x.shape[1] // 32
    (out,) = launcher(
        {"x": x},
        outputs={"out": ([rows * 32, 32], torch.float32)},
        runtime={
            "rows": rows,
            "wt": wt,
            "scale": bits(scale),
            "inv_w": bits(1.0 / x.shape[1]),
        },
    )
    return out
