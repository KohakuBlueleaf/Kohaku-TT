"""RMS norm at the metal level: the backend's own norm forward (ktt_norm, one core per
tile row) as a model; rows reach the compute and the writer as control words, and past
one row per core come from a counter on core 0.
Usage: python examples/kernels/dsl_metal/rmsnorm_km.py [--rows M] [--width W]"""

import argparse
import math
import os
import struct
import textwrap

import torch

import kohakutt
import kohakutt.dsl.metal as km

# Column tiles a row may have (W up to 2048), and tile rows of x resident per core.
LIMIT = 64
DEPTH = 2
# y tiles the writer sends per flush.
BLOCK = 4

f = km.sfpu


def y_fn():
    """y = (x - m) r gamma + beta as the backend's norm computes it: m and r column
    broadcasts in one slot (S), gamma / beta of pair tile t in row bands 2 t, 2 t + 1
    of another (P); over two tiles S and P load once per band."""
    x = f.operand(0)
    m, r = f.operand(1, along="col", half=0), f.operand(2, along="col", half=1)
    g = f.operand(3, along="row", band=0, step=2)
    b = f.operand(4, along="row", band=1, step=2)
    return km.sfpu_fn("norm_y", f.mad((x - m) * r, g, b))


Y = y_fn()

# Columns `cols` .. 31 of register slot `out` set to 0 (the last tile of a ragged row).
CUT = km.SfpuFn(
    "norm_cut",
    textwrap.dedent("""
        #ifdef TRISC_MATH
        inline void norm_cut(uint32_t, uint32_t, uint32_t out, uint32_t cols) {
            for (int u = 0; u < 32; ++u) {
                const int col = 16 * ((u / 8) % 2) + u % 2;
                v_if((sfpi::vConstTileId & 14) + col >= static_cast<int>(cols)) {
                    sfpi::dst_reg[out * 32 + u] = 0.0f;
                }
                v_endif;
            }
        }
        #endif
        """),
)


def rmsnorm_program(machine, dtype="bfloat16", gamma=None):
    """y = x * rsqrt(mean(x^2) + eps) (* gamma) per row, for x in format `dtype` and
    gamma in format `gamma` (None: no gamma). Runtime values: `rows` (tile rows),
    `wt` (tiles per row), `groups` (cores with work), `dynamic` (1: rows from the
    counter), `last` (valid columns of the last tile), `inv_w` and `eps` (fp32 bits)."""
    p = km.Program("rmsnorm" if gamma is None else "rmsnorm_gamma", machine)
    x = p.input("x", dtype)
    g = None if gamma is None else p.input("gamma", gamma)
    y = p.output("y", dtype)
    rows, wt = p.runtime("rows"), p.runtime("wt")
    groups, dynamic = p.runtime("groups"), p.runtime("dynamic")
    last, inv_w, eps = p.runtime("last"), p.runtime("inv_w"), p.runtime("eps")
    p.layout(cores=groups)
    seg = p.sized("seg", of=wt, exact=True, limit=LIMIT)
    pairs = p.sized("pairs", of=(wt + 1) // 2, exact=True, limit=LIMIT // 2)
    # y pages: a whole row, at least two flush blocks (the backend's ring).
    ring = p.sized("ring", of=wt, classes=range(2 * BLOCK, LIMIT + 1, 2))
    counter, first = p.semaphore(0, name="next"), p.core_ref(0)

    cb_x = p.cb(0, like=x, pages=DEPTH * seg)
    cb_g = None if g is None else p.cb(1, like=g, pages=seg)
    cb_p = p.cb(3, dtype="float32", pages=pairs, name="params")
    cb_s = p.cb(5, dtype="float32", pages=1, name="stats")
    cb_st = p.cb(6, dtype="float32", pages=2, name="columns")
    cb_y = p.cb(16, like=y, pages=ring)
    to_compute = p.words(24, name="to_compute")
    to_writer = p.words(25, name="to_writer")

    k = p.kernel("dm0")
    if g is not None:
        slot = k.reserve(cb_g, seg)
        with k.loop(seg, name="j") as j:
            k.read(g.page(j), slot[j])
        k.reads_done()
        k.push(cb_g, seg)
    r = k.var("r", p.core)
    with k.forever():
        k.put(to_compute, r)
        k.put(to_writer, r)
        k.exit_when(r >= rows)
        slot = k.reserve(cb_x, seg)
        with k.loop(seg, name="j") as j:
            k.read(x.page(r * wt + j), slot[j])
        k.reads_done()
        k.push(cb_x, seg)
        with k.when(dynamic.ne(0)):
            k.set(r, k.fetch_add(counter, at=first) + groups)
        with k.otherwise():
            k.set(r, r + groups)

    fp32 = [
        cb for cb in (cb_x, cb_g, cb_p, cb_s, cb_st) if cb and cb.dtype == "float32"
    ]
    c = p.kernel("compute", fp32_dest=True, unpack_fp32=fp32, exact=True)
    c.include("utility")
    c.include("api/compute/welford.h")
    c.include("api/compute/transpose_dest.h")
    c.startup(cb_x, None, cb_y)

    def fill(slot, value):
        c.init("fill")
        c.op("fill", slot, value=value)

    # P: per pair of column tiles, gamma's first row broadcast (1 without gamma) in
    # row band 2 t and beta (0) in band 2 t + 1 for pair tile t.
    if cb_g is not None:
        c.wait(cb_g, seg)
    c.reserve(cb_p, pairs)
    with c.loop(pairs, name="k") as k:
        c.acquire()
        fill(0, 0.0)
        for t in range(2):
            with c.when(2 * k + t < seg):
                if cb_g is not None:
                    c.format(srca=cb_g, srcb=cb_g)
                    c.init("bcast_row", cb_g)
                    c.bcast(cb_g, (2 * k + t).bounded(0, seg), 1, "row")
                else:
                    fill(1, 1.0)
                fill(2, 0.0)
                c.init("mul")
                c.run(f.put_rows(2 * t), 1, out=0)
                c.run(f.put_rows(2 * t + 1), 2, out=0)
        c.commit()
        c.pack_format(cb_p)
        c.pack(0, cb_p, at=k)
        c.release()
    c.push(cb_p, pairs)
    if cb_g is not None:
        c.pop(cb_g, seg)
    c.wait(cb_p, pairs)

    with c.forever():
        row = c.take(to_compute, "row")
        c.exit_when(row >= rows)
        c.wait(cb_x, seg)

        # Statistics: x tiles transposed, each lane one row; rstd as a row.
        c.acquire()
        fill(1, 0.0)
        fill(2, 0.0)
        c.srca_format(cb_x)
        c.init("transpose", cb_x)
        c.init("stats")
        c.stats_clear()
        with c.loop(seg, name="j") as j:
            c.transpose(cb_x, j, 0)
            n = km.select(j.eq(seg - 1), last, 32)
            c.stats_m2(0, count=n, dual=True)
        c.stats_finalize(1, inv_w, dual=True)
        c.init("transpose_dest")
        c.op("transpose_dest", 1)
        c.op("transpose_dest", 2)
        c.init("relu")
        c.op("relu", 2)
        c.init("add_scalar")
        c.op("add_scalar", 2, value=eps)
        c.init("rsqrt")
        c.op("rsqrt", 2)
        c.commit()
        c.reserve(cb_st, 2)
        c.pack_format(cb_st)
        c.pack(1, cb_st)
        c.pack(2, cb_st)
        c.release()
        c.push(cb_st, 2)

        # S: the mean and rstd columns broadcast, rstd's faces 1 / 3 merged in.
        c.wait(cb_st, 2)
        c.acquire()
        c.format(srca=cb_st, srcb=cb_st)
        c.init("bcast_col", cb_st)
        c.bcast(cb_st, 0, 1, "col")
        c.bcast(cb_st, 1, 2, "col")
        c.init("mul")
        c.run(f.join_cols(), 1, 2, out=1)
        c.commit()
        c.pop(cb_st, 2)
        c.reserve(cb_s, 1)
        c.pack_format(cb_s)
        c.pack(1, cb_s)
        c.release()
        c.push(cb_s, 1)

        # y, two column tiles per register scope, pairs from pair row mod pairs; the
        # last tile cut at the valid columns.
        c.wait(cb_s, 1)
        c.pack_format(cb_y)
        with c.loop(pairs, name="i") as i:
            k = c.let("k", (row + i) % pairs)
            # Pages the pair reads: 2 k + 1 only when the pair has two tiles.
            pk, x0, x1 = (
                k.bounded(0, pairs),
                (2 * k).bounded(0, seg),
                (2 * k + 1).bounded(0, seg),
            )
            two = c.let("two", km.select(2 * k + 1 < seg, 1, 0))
            c.acquire()
            c.srca_format(cb_s)
            c.init("copy", cb_s)
            c.copy(cb_s, 0, 2)
            c.copy(cb_p, pk, 3)
            c.srca_format(cb_x)
            c.init("copy", cb_x)
            c.copy(cb_x, x0, 0)
            with c.when(two.ne(0)):
                c.copy(cb_x, x1, 1)
            with c.otherwise():
                c.copy(cb_x, x0, 1)
            c.init("mul")
            with c.when(two.ne(0)):
                c.run(Y, 0, 2, 2, 3, 3, out=0, tiles=2)
            with c.otherwise():
                c.run(Y, 0, 2, 2, 3, 3, out=0, tiles=1)
            # The slot holding the cut last tile, 2 when there is none.
            at_end = km.select((2 * k + two).eq(seg - 1), two, 2)
            cut = c.let("cut", km.select(last < 32, at_end, 2))
            for t in range(2):
                with c.when(cut.eq(t)):
                    c.sfpu(CUT, 0, 1, t, mode="none", args=(last,))
            c.commit()
            # Two pages per pair; the writer skips the second of a single tile's.
            c.reserve(cb_y, 2)
            c.pack(0, cb_y)
            c.pack(1, cb_y)
            c.release()
            c.push(cb_y, 2)
        c.pop(cb_s, 1)
        c.pop(cb_x, seg)
    c.pop(cb_p, pairs)

    w = p.kernel("dm1")
    with w.forever():
        row = w.take(to_writer, "row")
        w.exit_when(row >= rows)
        with w.loop(pairs, name="i") as i:
            k = w.let("k", (row + i) % pairs)
            w.wait(cb_y, 2)
            w.write(cb_y, y.page(row * wt + 2 * k))
            with w.when(2 * k + 1 < seg):
                w.write(cb_y, y.page(row * wt + 2 * k + 1), 1)
            w.flushed()
            w.pop(cb_y, 2)
    w.writes_done()
    return p


def bits(value):
    """The fp32 bits of `value` (a Python float or a 0-d fp32 tensor)."""
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def runtime(shape, eps, cores):
    """The runtime values of rmsnorm_program for x of device shape `shape`."""
    w = shape[-1]
    rows = math.prod(shape[:-2]) * math.ceil(shape[-2] / 32)
    wt = math.ceil(w / 32)
    groups = min(rows, cores)
    one = torch.tensor(1.0, dtype=torch.float32)
    return {
        "rows": rows,
        "wt": wt,
        "groups": groups,
        "dynamic": int(rows > groups),
        "last": w - (wt - 1) * 32,
        "inv_w": bits(one / torch.tensor(float(w), dtype=torch.float32)),
        "eps": bits(eps),
    }


def rmsnorm(launcher, x, eps, gamma=None):
    """y of `launcher` (a km.Launcher of rmsnorm_program) for tt tensors x and gamma
    ([W], or None)."""
    cols, rows = torch.ops.ktt.compute_grid()
    inputs = {"x": x}
    if gamma is not None:
        inputs["gamma"] = gamma.reshape(1, -1)
    values = runtime(x.shape, eps, cols * rows)
    (out,) = launcher(inputs, outputs={"y": (list(x.shape), x.dtype)}, runtime=values)
    return out


def main():
    """Runs both norms on tt tensors; prints whether they agree bit for bit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--width", type=int, default=768)
    args = parser.parse_args()
    torch.manual_seed(0)
    x = torch.randn(args.rows, args.width).bfloat16()
    gamma = torch.randn(args.width).bfloat16()
    dev = kohakutt.device(0)
    try:
        cols, rows = torch.ops.ktt.compute_grid()
        program = rmsnorm_program(km.Machine(cols, rows), "bfloat16", "bfloat16")
        launcher = km.Launcher(program)
        x_tt, g_tt = x.to(dev), gamma.to(dev)
        got = rmsnorm(launcher, x_tt, 1e-6, g_tt).cpu()
        want = torch.nn.functional.rms_norm(x_tt, (args.width,), g_tt, 1e-6).cpu()
        equal = torch.equal(got, want)
        print(f"rmsnorm_km {tuple(x.shape)}: equal to the backend: {equal}")
    finally:
        kohakutt.close()


if __name__ == "__main__":
    # The backend's norm with one core per tile row, as this program lays it out.
    os.environ.setdefault("KTT_NORM_GROUP", "1")
    main()
