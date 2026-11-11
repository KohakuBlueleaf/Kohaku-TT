"""Row-major data: x [R, W] read by rows (one row per page), tilized into the tiles of
out, and those tiles untilized again into the rows of y (also by rows). W = 32 *
BLOCK, R a multiple of 32."""

import kohakutt.dsl.metal as km

BLOCK = 2
WIDTH = 32 * BLOCK


def rows_program(machine, dtype="bfloat16", cut=None):
    """The program above; runtime value `blocks` (32-row blocks of x). `cut` seeds a
    bug: "double_rows" (row buffer pages of two rows), "tile_back" (untilize into a
    buffer of tile pages)."""
    p = km.Program(f"rowmajor_{dtype}", machine)
    x = p.input("x", dtype, rows=WIDTH)
    out, y = p.output("out", dtype), p.output("y", dtype, rows=WIDTH)
    start, count = p.split(p.runtime("blocks"))
    row = x.page_bytes
    page = 2 * row if cut == "double_rows" else row
    cb_rows = p.cb(0, dtype=dtype, pages=64, page=page, name="rows")
    cb_tiles = p.cb(1, dtype=dtype, pages=2 * BLOCK, name="tiles")
    cb_out = p.cb(16, dtype=dtype, pages=2 * BLOCK, name="out")
    back = None if cut == "tile_back" else row
    cb_back = p.cb(17, dtype=dtype, pages=64, page=back, name="back")

    k = p.kernel("dm0")
    with k.loop(count, name="m") as m, k.loop(32, name="r") as r:
        slot = k.reserve(cb_rows, 1)
        k.read(x.page((start + m) * 32 + r), slot)
        k.reads_done()
        k.push(cb_rows, 1)

    fp32 = [cb_rows, cb_tiles] if dtype == "float32" else []
    c = p.kernel("compute", fp32_dest=dtype == "float32", unpack_fp32=fp32)
    c.startup(cb_rows, None, cb_tiles)
    with c.loop(count, name="m"):
        c.wait(cb_rows, 32)
        c.reserve(cb_tiles, BLOCK)
        c.tilize(cb_rows, cb_tiles, BLOCK)
        c.push(cb_tiles, BLOCK)
        c.pop(cb_rows, 32)
        c.wait(cb_tiles, BLOCK)
        c.format(srca=cb_tiles, srcb=cb_tiles)
        c.pack_format(cb_out)
        c.init("copy", cb_tiles)
        c.reserve(cb_out, BLOCK)
        with c.loop(BLOCK, name="j") as j:
            c.acquire()
            c.copy(cb_tiles, j, 0)
            c.commit()
            c.pack(0, cb_out, at=j)
            c.release()
        c.push(cb_out, BLOCK)
        c.reserve(cb_back, 32)
        c.untilize(cb_tiles, cb_back, BLOCK)
        c.push(cb_back, 32)
        c.pop(cb_tiles, BLOCK)

    w = p.kernel("dm1")
    with w.loop(count, name="m") as m:
        w.wait(cb_out, BLOCK)
        with w.loop(BLOCK, name="j") as j:
            w.write(cb_out, out.page((start + m) * BLOCK + j), index=j)
        w.flushed()
        w.pop(cb_out, BLOCK)
        with w.loop(32, name="r") as r:
            w.wait(cb_back, 1)
            w.write(cb_back, y.page((start + m) * 32 + r))
            w.flushed()
            w.pop(cb_back, 1)
    w.writes_done()
    return p


def run(launcher, x):
    """(out, y) of the launch for tt tensor x [R, WIDTH]."""
    rows = x.shape[0]
    return launcher(
        {"x": x},
        outputs={"out": ([rows, WIDTH], x.dtype), "y": ([rows, WIDTH], x.dtype)},
        runtime={"blocks": rows // 32},
    )
