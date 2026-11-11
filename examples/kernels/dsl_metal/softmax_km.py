"""Row softmax at the metal level with the matrix unit's statements: per tile row of
x, the row maximum (reduction), exp(x - max) (a column-broadcast subtraction, then the
exp table op), the row sum (reduction) and its reciprocal, and the product (a
column-broadcast multiply). Usage: python examples/kernels/dsl_metal/softmax_km.py"""

import argparse
import math

import torch

import kohakutt
import kohakutt.dsl.metal as km

# Column tiles a row may have.
LIMIT = 32


def softmax_program(machine):
    """softmax(x, -1) for bf16 x of whole tiles; `s` is a bf16 tile of ones (the
    reductions' scaler). Runtime values: `rows` (tile rows), `wt` (tiles per row)."""
    p = km.Program("softmax", machine)
    x, s, y = (
        p.input("x", "bfloat16"),
        p.input("s", "bfloat16"),
        p.output("y", "bfloat16"),
    )
    wt = p.runtime("wt")
    first, count = p.split(p.runtime("rows"))
    seg = p.sized("seg", of=wt, exact=True, limit=LIMIT)
    cb_x, cb_s = p.cb(0, like=x, pages=seg), p.cb(1, like=s, pages=1)
    cb_e = p.cb(2, dtype="bfloat16", pages=seg, name="exps")
    cb_m = p.cb(3, dtype="bfloat16", pages=1, name="max")
    cb_sum = p.cb(4, dtype="bfloat16", pages=1, name="sum")
    cb_y = p.cb(16, like=y, pages=2)

    k = p.kernel("dm0")
    slot = k.reserve(cb_s, 1)
    k.read(s.page(0), slot)
    k.reads_done()
    k.push(cb_s, 1)
    with k.loop(count, name="r") as r:
        slot = k.reserve(cb_x, seg)
        with k.loop(seg, name="j") as j:
            k.read(x.page((first + r) * wt + j), slot[j])
        k.reads_done()
        k.push(cb_x, seg)

    c = p.kernel("compute", fp32_dest=True, exact=True)
    c.startup(cb_x, cb_s, cb_y)
    c.wait(cb_s, 1)

    def reduced(kind, cb, out):
        """The row `kind` of the seg tiles of `cb` into a page of `out`."""
        c.acquire()
        r = c.reduce_init(kind, "row", cb, cb_s, out)
        with c.loop(seg, name="j") as j:
            c.reduce(r, j, 0, 0)
        if kind == "sum":
            c.init("recip")
            c.op("recip", 0)
        c.commit()
        c.reserve(out, 1)
        c.pack(0, out)
        c.release()
        c.reduce_done()
        c.push(out, 1)

    with c.loop(count, name="r"):
        c.wait(cb_x, seg)
        reduced("max", cb_x, cb_m)
        c.wait(cb_m, 1)
        c.reserve(cb_e, seg)
        with c.loop(seg, name="j") as j:
            c.acquire()
            c.tiles(c.tiles_init("sub", cb_x, cb_m, bcast="col"), j, 0, 0)
            c.init("exp")
            c.op("exp", 0)
            c.commit()
            c.pack(0, cb_e, at=j)
            c.release()
        c.push(cb_e, seg)
        c.pop(cb_m, 1)
        c.pop(cb_x, seg)
        c.wait(cb_e, seg)
        reduced("sum", cb_e, cb_sum)
        c.wait(cb_sum, 1)
        with c.loop(seg, name="j") as j:
            c.acquire()
            c.tiles(c.tiles_init("mul", cb_e, cb_sum, bcast="col"), j, 0, 0)
            c.commit()
            c.reserve(cb_y, 1)
            c.pack(0, cb_y)
            c.release()
            c.push(cb_y, 1)
        c.pop(cb_sum, 1)
        c.pop(cb_e, seg)
    c.pop(cb_s, 1)

    w = p.kernel("dm1")
    with w.loop(count, name="r") as r, w.loop(seg, name="j") as j:
        w.wait(cb_y, 1)
        w.write(cb_y, y.page((first + r) * wt + j))
        w.flushed()
        w.pop(cb_y, 1)
    w.writes_done()
    return p


def softmax(launcher, x, ones):
    """softmax(x, -1) of `launcher` (a km.Launcher of softmax_program) for tt x."""
    rows, wt = math.prod(x.shape[:-1]) // 32, x.shape[-1] // 32
    (y,) = launcher(
        {"x": x, "s": ones},
        outputs={"y": (list(x.shape), x.dtype)},
        runtime={"rows": rows, "wt": wt},
    )
    return y


def main():
    """Runs the program and prints its largest difference from torch.softmax."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()
    torch.manual_seed(0)
    x = (torch.randn(args.rows, args.width) * 4).bfloat16()
    dev = kohakutt.device(0)
    try:
        cols, rows = torch.ops.ktt.compute_grid()
        launcher = km.Launcher(softmax_program(km.Machine(cols, rows)))
        ones = torch.ones(32, 32).bfloat16().to(dev)
        got = softmax(launcher, x.to(dev), ones).cpu().float()
        want = torch.softmax(x.float(), -1)
        worst = (got - want).abs().max()
        print(f"softmax_km {tuple(x.shape)}: max difference {worst:.3e}", flush=True)
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
