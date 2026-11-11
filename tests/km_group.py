"""Exchange inside groups of cores: each core holds its tile of x; after a barrier of
its group of GROUP consecutive cores it fetches every member's tile and writes their
sum, so every core of a group ends with the group's total (a split-row norm's
partial statistics, exchanged)."""

import kohakutt.dsl.metal as km

GROUP = 4


def group_program(machine):
    """The program above; runtime value `working` (cores, a multiple of GROUP)."""
    p = km.Program("group", machine)
    x, out = p.input("x", "bfloat16"), p.output("out", "bfloat16")
    p.layout(cores=p.runtime("working"))
    point = p.sync_point(group=GROUP)
    cb_mine = p.cb(0, like=x, pages=1)
    cb_all = p.cb(1, like=x, pages=GROUP, name="all")
    cb_out = p.cb(16, like=out, pages=2)

    k = p.kernel("dm0")
    mine = k.reserve(cb_mine, 1)
    k.read(x.page(p.core), mine)
    k.reads_done()
    k.sync(point)
    every = k.reserve(cb_all, GROUP)
    for i in range(GROUP):
        k.fetch(mine, point.member(i), every[i], x.page_bytes)
    k.reads_done()
    k.push(cb_all, GROUP)
    k.sync(point)
    k.push(cb_mine, 1)

    c = p.kernel("compute", fp32_dest=True)
    c.startup(cb_all, cb_all, cb_out)
    c.wait(cb_all, GROUP)
    c.acquire()
    c.tiles(c.tiles_init("add", cb_all, cb_all), 0, 1, 0)
    c.tiles(c.tiles_init("add", cb_all, cb_all, acc=True), 2, 3, 0)
    c.commit()
    c.reserve(cb_out, 1)
    c.pack(0, cb_out)
    c.release()
    c.push(cb_out, 1)
    c.pop(cb_all, GROUP)

    w = p.kernel("dm1")
    w.wait(cb_out, 1)
    w.write(cb_out, out.page(p.core))
    w.flushed()
    w.pop(cb_out, 1)
    w.wait(cb_mine, 1)
    w.pop(cb_mine, 1)
    w.writes_done()
    return p
