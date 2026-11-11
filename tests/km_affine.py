"""A layer norm's affine step as the backend's norm kernel writes it: per row of tiles
the mean m and reciprocal r (column broadcasts) share one slot (S), the gamma and
beta rows of two output tiles share another (P), and one generated function
y = (x - m) r gamma + beta runs over two x tiles, reading S and P once per band."""

import kohakutt.dsl.metal as km

f = km.sfpu


def affine_fn(first=0):
    """(x - m) * r * gamma + beta with m, r in one slot and gamma, beta of tile t in
    row bands 2 (first + t), 2 (first + t) + 1 of another."""
    x = f.operand(0)
    m, r = f.operand(1, along="col", half=0), f.operand(2, along="col", half=1)
    g = f.operand(3, along="row", band=2 * first, step=2)
    b = f.operand(4, along="row", band=2 * first + 1, step=2)
    return km.sfpu_fn(f"affine_fn{first}", f.mad((x - m) * r, g, b))


def affine_program(machine, tiles=2):
    """out = (x - m) r gamma + beta for x [32 R, 64 P] (fp32), m and r [32 R, 32]
    (column 0 per row), gamma and beta [32, 64 P] (row 0); runtime values `rows` (R)
    and `pairs` (P). `tiles` 1 runs the function once per x tile."""
    p = km.Program(f"affine{tiles}", machine)
    x, m, r = (p.input(n, "float32") for n in ("x", "m", "r"))
    g, b = p.input("g", "float32"), p.input("b", "float32")
    out = p.output("out", "float32")
    rows, pairs = p.runtime("rows"), p.runtime("pairs")
    cb_x, cb_m, cb_r = p.cb(0, like=x, pages=2), p.cb(1, like=m), p.cb(2, like=r)
    cb_g, cb_b = p.cb(3, like=g, pages=2), p.cb(4, like=b, pages=2)
    cb_out = p.cb(16, like=out, pages=2)
    fn = affine_fn()

    k = p.kernel("dm0")
    with k.loop(rows, name="i") as i, k.loop(pairs, name="j") as j:
        for cb, ref, first, count in (
            (cb_x, x, i * pairs * 2 + j * 2, 2),
            (cb_m, m, i, 1),
            (cb_r, r, i, 1),
            (cb_g, g, j * 2, 2),
            (cb_b, b, j * 2, 2),
        ):
            slot = k.reserve(cb, count)
            for t in range(count):
                k.read(ref.page(first + t), slot[t])
            k.reads_done()
            k.push(cb, count)

    cbs = [cb_x, cb_m, cb_r, cb_g, cb_b]
    c = p.kernel("compute", fp32_dest=True, unpack_fp32=cbs, approx=False)
    c.startup(cb_x, cb_x, cb_out)
    with c.loop(rows, name="i"), c.loop(pairs, name="j"):
        for cb, n in zip(cbs, (2, 1, 1, 2, 2), strict=True):
            c.wait(cb, n)
        c.acquire()
        c.init("mul")
        for t in range(2):
            c.init("bcast_row", cb_g)
            c.bcast(cb_g, t, 0, "row")
            c.init("bcast_row", cb_b)
            c.bcast(cb_b, t, 1, "row")
            c.run(f.put_rows(2 * t), 0, out=3)
            c.run(f.put_rows(2 * t + 1), 1, out=3)
        c.init("bcast_col", cb_m)
        c.bcast(cb_m, 0, 2, "col")
        c.init("bcast_col", cb_r)
        c.bcast(cb_r, 0, 0, "col")
        c.run(f.join_cols(), 2, 0, out=2)
        c.init("copy", cb_x)
        c.copy(cb_x, 0, 0)
        c.copy(cb_x, 1, 1)
        if tiles == 2:
            c.run(fn, 0, 2, 2, 3, 3, out=0, tiles=2)
        else:
            for t in range(2):
                c.run(affine_fn(t), t, 2, 2, 3, 3, out=t)
        c.reserve(cb_out, 2)
        c.pack(0, cb_out)
        c.pack(1, cb_out)
        c.release()
        c.push(cb_out, 2)
        for cb, n in zip(cbs, (2, 1, 1, 2, 2), strict=True):
            c.pop(cb, n)

    w = p.kernel("dm1")
    with w.loop(rows, name="i") as i, w.loop(pairs, name="j") as j:
        w.wait(cb_out, 2)
        for t in range(2):
            w.write(cb_out, out.page(i * pairs * 2 + j * 2 + t), t)
        w.flushed()
        w.pop(cb_out, 2)
    w.writes_done()
    return p
