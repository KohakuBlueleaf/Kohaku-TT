"""Matrix-unit whole-tile ops and reductions: per row tile of x, y ([R * 32, 64]), out
holds x0 + y0, x1 * y1, x0 - v's first row, x1 * v's first column, x0 + v[0, 0],
x1 + (x0 + y0) accumulated on the slot, the row sums of [x0 x1], the column maxima
of x0 and the sum of x1 (v a [32, 32] tile, s a tile of ones)."""

import kohakutt.dsl.metal as km

OUT_TILES = 9


def tiles_program(machine):
    """The program above; runtime value `rows` (row tiles of x)."""
    p = km.Program("tiles", machine)
    x, y, v, s = (p.input(name, "bfloat16") for name in ("x", "y", "v", "s"))
    out = p.output("out", "bfloat16")
    first, count = p.split(p.runtime("rows"))
    cb_x, cb_y = p.cb(0, like=x, pages=4), p.cb(1, like=y, pages=4)
    cb_v, cb_s = p.cb(2, like=v, pages=1), p.cb(3, like=s, pages=1)
    cb_out = p.cb(16, like=out, pages=2 * OUT_TILES)

    k = p.kernel("dm0")
    for cb, ref in ((cb_v, v), (cb_s, s)):
        slot = k.reserve(cb, 1)
        k.read(ref.page(0), slot)
        k.reads_done()
        k.push(cb, 1)
    with k.loop(count, name="m") as m:
        for cb, ref in ((cb_x, x), (cb_y, y)):
            slot = k.reserve(cb, 2)
            with k.loop(2, name="j") as j:
                k.read(ref.page((first + m) * 2 + j), slot[j])
            k.reads_done()
            k.push(cb, 2)

    c = p.kernel("compute", fp32_dest=True, min_fidelity="HiFi4")
    c.startup(cb_x, cb_y, cb_out)
    c.wait(cb_v, 1)
    c.wait(cb_s, 1)
    with c.loop(count, name="m"):
        c.wait(cb_x, 2)
        c.wait(cb_y, 2)
        c.reserve(cb_out, OUT_TILES)
        c.acquire()
        c.tiles(c.tiles_init("add", cb_x, cb_y), 0, 0, 0)
        c.tiles(c.tiles_init("mul", cb_x, cb_y), 1, 1, 1)
        c.tiles(c.tiles_init("sub", cb_x, cb_v, bcast="row"), 0, 0, 2)
        c.commit()
        for slot in range(3):
            c.pack(slot, cb_out, at=slot)
        c.release()
        c.acquire()
        c.tiles(c.tiles_init("mul", cb_x, cb_v, bcast="col"), 1, 0, 0)
        c.tiles(c.tiles_init("add", cb_x, cb_v, bcast="scalar"), 0, 0, 1)
        c.init("copy", cb_x)
        c.copy(cb_x, 1, 2)
        c.tiles(c.tiles_init("add", cb_x, cb_y, acc=True), 0, 0, 2)
        c.commit()
        for slot in range(3):
            c.pack(slot, cb_out, at=3 + slot)
        c.release()
        for at, (kind, dim, tiles) in enumerate(
            (("sum", "row", (0, 1)), ("max", "col", (0,)), ("sum", "scalar", (1,)))
        ):
            c.acquire()
            r = c.reduce_init(kind, dim, cb_x, cb_s, cb_out)
            for t in tiles:
                c.reduce(r, t, 0, 0)
            c.commit()
            c.pack(0, cb_out, at=6 + at)
            c.release()
        c.reduce_done()
        c.push(cb_out, OUT_TILES)
        c.pop(cb_x, 2)
        c.pop(cb_y, 2)
    c.pop(cb_v, 1)
    c.pop(cb_s, 1)

    d = p.kernel("dm1")
    with d.loop(count, name="m") as m:
        d.wait(cb_out, OUT_TILES)
        with d.loop(OUT_TILES, name="j") as j:
            d.write(cb_out, out.page((first + m) * OUT_TILES + j), index=j)
        d.flushed()
        d.pop(cb_out, OUT_TILES)
    d.writes_done()
    return p


def expected(x, y, v):
    """The nine output tiles per row tile, in fp32, for CPU tensors x, y, v."""
    import torch

    x, y, v = x.float(), y.float(), v.float()
    x0, x1, y0, y1 = x[:, :32], x[:, 32:], y[:, :32], y[:, 32:]
    rows = x.shape[0]
    zero = torch.zeros(rows, 32)
    row_sum = zero.clone()
    row_sum[:, 0] = x.sum(1)
    col_max = zero.clone().view(-1, 32, 32)
    col_max[:, 0, :] = x0.view(-1, 32, 32).amax(1)
    total = zero.clone().view(-1, 32, 32)
    total[:, 0, 0] = x1.view(-1, 32, 32).sum((1, 2))
    tiles = [
        x0 + y0,
        x1 * y1,
        x0 - v[0:1, :],
        x1 * v[:, 0:1].repeat(rows // 32, 1),
        x0 + v[0, 0],
        x1 + x0 + y0,
        row_sum,
        col_max.view(rows, 32),
        total.view(rows, 32),
    ]
    return torch.cat(tiles, 1)


def run(launcher, x, y, v, s):
    """out of the launch for tt tensors x, y ([R * 32, 64]), v and s ([32, 32])."""
    import torch

    rows = x.shape[0] // 32
    (out,) = launcher(
        {"x": x, "y": y, "v": v, "s": s},
        outputs={"out": ([rows * 32, 32 * OUT_TILES], torch.bfloat16)},
        runtime={"rows": rows},
    )
    return out
