"""Data kept in L1 across the stages of one program: every core reads its tile of x
into L1 and holds it; after a barrier each core fetches the held tile of its right
neighbour (same grid row, wrapping) into its output tile; a second barrier lets the
neighbours finish before the tiles are let go."""

import kohakutt.dsl.metal as km


def sync_program(machine, cut=None):
    """The program above on the whole grid (one tile per core). `cut` seeds a bug:
    "when" (the second sync under a per-core condition), "twice" (syncs of one
    point in both data-movement kernels)."""
    p = km.Program("sync", machine)
    x, out = p.input("x", "bfloat16"), p.output("out", "bfloat16")
    point = p.sync_point()
    right = p.core_ref(lambda cx, cy: ((cx + 1) % machine.cols, cy), name="right")
    cb_mine, cb_got = p.cb(0, like=x, pages=1), p.cb(16, like=out, pages=1)

    k = p.kernel("dm0")
    mine = k.reserve(cb_mine, 1)
    k.read(x.page(p.core), mine)
    k.reads_done()
    k.sync(point)
    got = k.reserve(cb_got, 1)
    k.fetch(mine, right, got, x.page_bytes)
    k.reads_done()
    k.push(cb_got, 1)
    if cut == "when":
        with k.when(p.core.eq(0)):
            k.sync(point)
    else:
        k.sync(point)
    k.push(cb_mine, 1)

    w = p.kernel("dm1")
    w.wait(cb_got, 1)
    w.write(cb_got, out.page(p.core))
    w.flushed()
    w.pop(cb_got, 1)
    w.wait(cb_mine, 1)
    w.pop(cb_mine, 1)
    if cut == "twice":
        w.sync(point)
    w.writes_done()
    return p


def neighbours(cols, cores):
    """The core whose tile each core's output holds."""
    return [(c // cols) * cols + (c % cols + 1) % cols for c in range(cores)]
