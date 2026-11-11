"""Packer accumulation across the iterations of a loop: per tile row of x ([R * 32,
C * 32]), out's tile is the sum of the row's C tiles, packed into one page held over
the loop (the first iteration writes it, the later ones add to it), in fp32."""

import kohakutt.dsl.metal as km


def accumulate_program(machine, cut=None):
    """The program above; runtime values `rows` and `wt` (tiles per row). `cut` seeds
    a bug: "inside" (the page reserved inside the loop), "moving" (a pack at the
    loop's variable)."""
    p = km.Program("accumulate", machine)
    x, out = p.input("x", "float32"), p.output("out", "float32")
    wt = p.runtime("wt")
    start, count = p.split(p.runtime("rows"))
    cb_x, cb_out = p.cb(0, like=x, pages=2), p.cb(16, like=out, pages=2)

    k = p.kernel("dm0")
    with k.loop(count, name="r") as r, k.loop(wt, name="j") as j:
        slot = k.reserve(cb_x, 1)
        k.read(x.page((start + r) * wt + j), slot)
        k.reads_done()
        k.push(cb_x, 1)

    c = p.kernel("compute", fp32_dest=True, unpack_fp32=[cb_x])
    c.startup(cb_x, None, cb_out)
    c.init("copy", cb_x)
    with c.loop(count, name="r"):
        if cut != "inside":
            c.reserve(cb_out, 1)
        with c.loop(wt, name="j") as j:
            if cut == "inside":
                c.reserve(cb_out, 1)
            c.wait(cb_x, 1)
            c.acquire()
            c.copy(cb_x, 0, 0)
            c.commit()
            c.pack_accumulate("after_first", cb_out, after=j)
            c.pack(0, cb_out, at=j if cut == "moving" else 0)
            c.pack_accumulate(0)
            c.release()
            c.pop(cb_x, 1)
            if cut == "inside":
                c.push(cb_out, 1)
        if cut != "inside":
            c.push(cb_out, 1)

    w = p.kernel("dm1")
    with w.loop(count, name="r") as r, w.loop(wt if cut == "inside" else 1) as _:
        w.wait(cb_out, 1)
        w.write(cb_out, out.page(start + r))
        w.flushed()
        w.pop(cb_out, 1)
    w.writes_done()
    return p


def run(launcher, x):
    """out of the launch for tt fp32 x [R * 32, C * 32]: [R * 32, 32]."""
    import torch

    rows, wt = x.shape[0] // 32, x.shape[1] // 32
    (out,) = launcher(
        {"x": x},
        outputs={"out": ([rows * 32, 32], torch.float32)},
        runtime={"rows": rows, "wt": wt},
    )
    return out
