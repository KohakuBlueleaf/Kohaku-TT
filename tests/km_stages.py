"""A per-core stage table: each core walks its rows (stage, source tile, tiles read,
destination tile); stage 0 copies a tile of x to out, stage 1 writes the sum of two
consecutive tiles. All three kernels switch on the same table word."""

import kohakutt.dsl.metal as km

ROWS = 8
WIDTH = 4
SOURCE, COUNT, TARGET = 1, 2, 3


def stages_program(machine, cut=None):
    """The program above; runtime value `cores` (cores with rows). `cut` seeds a bug:
    "unswitched" (the compute does not switch: its pops differ from the pushes),
    "less" (the reader switches on a comparison other than equality)."""
    p = km.Program("stages", machine)
    x, out = p.input("x", "bfloat16"), p.output("out", "bfloat16")
    p.layout(cores=p.runtime("cores"))
    st = p.table(
        "stages",
        width=WIDTH,
        rows=ROWS,
        most={0: 1, COUNT: 2},
        pages={SOURCE: (x, COUNT), TARGET: (out, COUNT)},
    )
    cb_x, cb_out = p.cb(0, like=x, pages=4), p.cb(16, like=out, pages=2)

    k = p.kernel("dm0")
    with k.loop(st.used, name="j") as j:
        with k.when(st[j, 0] < 1 if cut == "less" else st[j, 0].eq(0)):
            slot = k.reserve(cb_x, 1)
            k.read(x.page(st[j, SOURCE]), slot)
            k.reads_done()
            k.push(cb_x, 1)
        with k.otherwise():
            slot = k.reserve(cb_x, 2)
            with k.loop(st[j, COUNT], name="t") as t:
                k.read(x.page(st[j, SOURCE] + t), slot[t])
            k.reads_done()
            k.push(cb_x, 2)

    c = p.kernel("compute", fp32_dest=True)
    c.startup(cb_x, cb_x, cb_out)

    def stage(tiles):
        c.wait(cb_x, tiles)
        c.acquire()
        if tiles == 1:
            c.init("copy", cb_x)
            c.copy(cb_x, 0, 0)
        else:
            c.tiles(c.tiles_init("add", cb_x, cb_x), 0, 1, 0)
        c.commit()
        c.reserve(cb_out, 1)
        c.pack(0, cb_out)
        c.release()
        c.push(cb_out, 1)
        c.pop(cb_x, tiles)

    with c.loop(st.used, name="j") as j:
        if cut == "unswitched":
            stage(1)
        else:
            with c.when(st[j, 0].eq(0)):
                stage(1)
            with c.otherwise():
                stage(2)

    w = p.kernel("dm1")
    with w.loop(st.used, name="j") as j:
        w.wait(cb_out, 1)
        w.write(cb_out, out.page(st[j, TARGET]))
        w.flushed()
        w.pop(cb_out, 1)
    w.writes_done()
    return p


def run(launcher, x, rows_per_core):
    """out [32, T * 32] of the launch for tt x [32, T * 32] and per-core rows."""
    import torch

    (out,) = launcher(
        {"x": x},
        outputs={"out": (list(x.shape), torch.bfloat16)},
        runtime={"cores": len(rows_per_core)},
        tables={"stages": rows_per_core},
    )
    return out
