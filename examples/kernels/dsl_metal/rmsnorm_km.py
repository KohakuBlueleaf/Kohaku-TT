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

# The backend's norm vector-unit code. S (slot 2): mean in faces 0 / 2, rstd in 1 / 3;
# P (slot 3): gamma at units 4 t + e, beta at 4 t + 2 + e (docs: ktt_norm/fwd_compute).
TEMPLATES = km.CppFn(
    "norm_templates",
    """
#ifdef TRISC_MATH
template <int T, int H, int C, int D, bool kMask>
sfpi_inline void y_vec(sfpi::vFloat m, sfpi::vFloat r, uint32_t cols) {
    constexpr int O = 32 * T + 16 * H + 8 * C + D;
    constexpr int G = 96 + 8 * C + 4 * T + D % 2;
    sfpi::vFloat v = (sfpi::dst_reg[O] - m) * r;
    v = v * sfpi::dst_reg[G] + sfpi::dst_reg[G + 2];
    if constexpr (kMask) {
        v_if((sfpi::vConstTileId & 14) + (16 * C + D % 2) >= static_cast<int>(cols)) {
            v = 0.0f;
        }
        v_endif;
    }
    sfpi::dst_reg[O] = v;
}

template <bool kMask, int H, int D>
sfpi_inline void y_band(uint32_t cols) {
    const sfpi::vFloat m = sfpi::dst_reg[64 + 16 * H + D];
    const sfpi::vFloat r = sfpi::dst_reg[64 + 16 * H + 8 + D];
    y_vec<0, H, 0, D, kMask>(m, r, cols);
    y_vec<0, H, 1, D, kMask>(m, r, cols);
}

template <bool kMask, int... B>
sfpi_inline void y_bands(std::integer_sequence<int, B...>, uint32_t cols) {
    (y_band<kMask, B / 8, B % 8>(cols), ...);
}
#endif
""",
)


def _function(name, body, args="uint32_t, uint32_t, uint32_t"):
    """Vector-unit function `name` with statements `body`, on the math thread only."""
    lines = textwrap.indent(textwrap.dedent(body).strip("\n"), "    ")
    code = f"#ifdef TRISC_MATH\ninline void {name}({args}) {{\n{lines}\n}}\n#endif"
    return km.SfpuFn(name, code)


COLS = "uint32_t, uint32_t, uint32_t, uint32_t cols"
Y = _function(
    "norm_y", "y_bands<false>(std::make_integer_sequence<int, 16>{}, cols);", COLS
)
Y_CUT = _function(
    "norm_y_cut", "y_bands<true>(std::make_integer_sequence<int, 16>{}, cols);", COLS
)
P_PUT = _function(
    "norm_p",
    """
for (int c = 0; c < 2; ++c) {
    for (int e = 0; e < 2; ++e) {
        const sfpi::vFloat g = sfpi::dst_reg[32 + 8 * c + e];
        const sfpi::vFloat b = sfpi::dst_reg[64 + 8 * c + e];
        sfpi::dst_reg[8 * c + e] = g;
        sfpi::dst_reg[8 * c + 2 + e] = b;
    }
}""",
)
MERGE = _function(
    "norm_merge",
    """
for (int d = 0; d < 8; ++d) {
    const sfpi::vFloat upper = sfpi::dst_reg[64 + 8 + d];
    const sfpi::vFloat lower = sfpi::dst_reg[64 + 24 + d];
    sfpi::dst_reg[32 + 8 + d] = upper;
    sfpi::dst_reg[32 + 24 + d] = lower;
}""",
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
    counter, first = p.semaphore(0, name="next"), p.core_ref(0)

    cb_x = p.cb(0, like=x, pages=DEPTH * seg)
    cb_g = None if g is None else p.cb(1, like=g, pages=seg)
    cb_p = p.cb(3, dtype="float32", pages=seg, name="params")
    cb_s = p.cb(5, dtype="float32", pages=1, name="stats")
    cb_st = p.cb(6, dtype="float32", pages=2, name="columns")
    cb_y = p.cb(16, like=y, pages=2)
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
    c.helper(TEMPLATES)
    c.startup(cb_x, None, cb_y)

    def fill(slot, value):
        c.init("fill")
        c.op("fill", slot, value=value)

    # P: per column tile, gamma's first row broadcast (1 without gamma), beta 0.
    if cb_g is not None:
        c.wait(cb_g, seg)
    c.reserve(cb_p, seg)
    with c.loop(seg, name="j") as j:
        c.acquire()
        fill(0, 0.0)
        if cb_g is not None:
            c.format(srca=cb_g, srcb=cb_g)
            c.init("bcast_row", cb_g)
            c.bcast(cb_g, j, 1, "row")
        else:
            fill(1, 1.0)
        fill(2, 0.0)
        c.init("mul")
        c.sfpu(P_PUT, 0, 1, 0, mode="none")
        c.commit()
        c.pack_format(cb_p)
        c.pack(0, cb_p, at=j)
        c.release()
    c.push(cb_p, seg)
    if cb_g is not None:
        c.pop(cb_g, seg)
    c.wait(cb_p, seg)

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
        c.inline("two_pass_stats_init_shifted();\ntwo_pass_stats_clear();")
        with c.loop(seg, name="j") as j:
            c.transpose(cb_x, j, 0)
            n = km.select(j.eq(seg - 1), last, 32)
            c.inline("two_pass_stats_update_rows(0, 0, n);", args={"n": n})
        c.inline("two_pass_stats_finalize_to_row(1, inv_w);", args={"inv_w": inv_w})
        c.inline(
            "transpose_dest_init<DST_ACCUM_MODE>();\n"
            "transpose_dest<DST_ACCUM_MODE>(1);\n"
            "transpose_dest<DST_ACCUM_MODE>(2);"
        )
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
        c.sfpu(MERGE, 1, 2, 1, mode="none")
        c.commit()
        c.pop(cb_st, 2)
        c.reserve(cb_s, 1)
        c.pack_format(cb_s)
        c.pack(1, cb_s)
        c.release()
        c.push(cb_s, 1)

        # y per column tile; the last one cut at the valid columns.
        c.wait(cb_s, 1)
        c.pack_format(cb_y)
        with c.loop(seg, name="j") as j:
            c.acquire()
            c.srca_format(cb_s)
            c.init("copy", cb_s)
            c.copy(cb_s, 0, 2)
            c.copy(cb_p, j, 3)
            c.srca_format(cb_x)
            c.init("copy", cb_x)
            c.copy(cb_x, j, 0)
            c.init("mul")
            cols = c.let("cols", km.select(j.eq(seg - 1), last, 32))
            with c.when(cols < 32):
                c.sfpu(Y_CUT, 0, 1, 0, mode="none", args=(cols,))
            with c.otherwise():
                c.sfpu(Y, 0, 1, 0, mode="none", args=(cols,))
            c.commit()
            c.reserve(cb_y, 1)
            c.pack(0, cb_y)
            c.push(cb_y, 1)
            c.release()
        c.pop(cb_s, 1)
        c.pop(cb_x, seg)
    c.pop(cb_p, seg)

    w = p.kernel("dm1")
    with w.forever():
        row = w.take(to_writer, "row")
        w.exit_when(row >= rows)
        with w.loop(seg, name="j") as j:
            w.wait(cb_y, 1)
            w.write(cb_y, y.page(row * wt + j))
            w.flushed()
            w.pop(cb_y, 1)
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
