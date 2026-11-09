"""Sweep of the grouped / depthwise / strided-1-D conv grid on Kohaku-TT's ktt_conv2.

Run as a script on one card (KTT_HW=1):
    python tests/hardware/ops/test_conv_sweep.py --chip 0 --json out.json --md out.md
Per cell (kernel x stride x dilation x groups x 1-D / 2-D, plus model shapes): forward,
input gradient and weight gradient against CPU float64 next to CPU bf16 (cosine and
max relative error), the route (one bridge op, Kohaku-TT programs only), descriptor
cache hits in warm calls, device-synchronized warm times, and TT-NN conv2d's forward
time and cosine in the same process (KTT_GROUPED_CONV=ttnn) where it runs the cell.
Under pytest, a small grid runs as a smoke test.
"""

import argparse
import json
import os
import time

import pytest
import torch
import torch.nn.functional as F

KERNELS = (1, 2, 3, 5, 7, 11)
STRIDES = (1, 2, 3, 4)
DILATIONS = (1, 2)
GROUPS = (2, 16, "dw")

# (name, N, C, H, W, O, groups, kernel, stride, dilation, padding); H == 1: 1-D.
MODELS = [
    ("convnext-t s1 dw7", 1, 96, 56, 56, 96, 96, 7, 1, 1, 3),
    ("convnext-t s2 dw7", 1, 192, 28, 28, 192, 192, 7, 1, 1, 3),
    ("convnext-t s3 dw7", 1, 384, 14, 14, 384, 384, 7, 1, 1, 3),
    ("convnext-t s4 dw7", 1, 768, 7, 7, 768, 768, 7, 1, 1, 3),
    ("convnext-t s1 dw7 b8", 8, 96, 56, 56, 96, 96, 7, 1, 1, 3),
    ("mobilenetv2 dw3 s1", 1, 144, 56, 56, 144, 144, 3, 1, 1, 1),
    ("mobilenetv2 dw3 s2", 1, 144, 56, 56, 144, 144, 3, 2, 1, 1),
    ("mobilenetv2 dw3 s2 b8", 8, 96, 112, 112, 96, 96, 3, 2, 1, 1),
    ("mobilenetv3 dw5 s2", 1, 240, 28, 28, 240, 240, 5, 2, 1, 2),
    ("resnext50 g32 c128", 1, 128, 56, 56, 128, 32, 3, 1, 1, 1),
    ("resnext50 g32 c256 s2", 1, 256, 56, 56, 256, 32, 3, 2, 1, 1),
    ("resnext50 g32 c512 s2", 1, 512, 28, 28, 512, 32, 3, 2, 1, 1),
    ("resnext50 g32 c1024", 1, 1024, 14, 14, 1024, 32, 3, 1, 1, 1),
    ("strided 1-D audio", 1, 256, 1, 4096, 256, 1, 3, 2, 1, 1),
    ("depthwise 1-D", 1, 512, 1, 2048, 512, 512, 7, 1, 1, 3),
]


def grid_cells(small=False, dims=(2, 1)):
    """The cell grid: 2-D on [1, 64, 56, 56], 1-D on [1, 128, 2048]."""
    cells = []
    kernels = (1, 3, 7) if small else KERNELS
    strides = (1, 2) if small else STRIDES
    for one_d in [d == 1 for d in dims]:
        c = 128 if one_d else 64
        h, w = (1, 2048) if one_d else (56, 56)
        for k in kernels:
            for s in strides:
                for d in DILATIONS:
                    for g in GROUPS:
                        groups = c if g == "dw" else g
                        if k == 1 and d == 2:
                            continue
                        p = d * (k - 1) // 2
                        name = f"{'1d' if one_d else '2d'} k{k} s{s} d{d} g{g}"
                        cells.append((name, 1, c, h, w, c, groups, k, s, d, p))
    return cells


def cosine(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def max_rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


def tensors(cell):
    _, n, c, h, w, o, groups, k, s, d, p = cell
    one_d = h == 1
    gen = torch.Generator().manual_seed(0)
    x = torch.randn((n, c, w) if one_d else (n, c, h, w), generator=gen)
    shape = (o, c // groups, k) if one_d else (o, c // groups, k, k)
    wt = torch.randn(shape, generator=gen) * (1.0 / (c // groups * k * k) ** 0.5)
    conv = F.conv1d if one_d else F.conv2d
    out = conv(x, wt, None, s, p, d, groups)
    grad = torch.randn(out.shape, generator=gen)
    return x, wt, grad, conv, (s, p, d, groups)


def timed(fn, iters, batches=3):
    """Warm device-synchronized seconds per call of fn: two warm-up calls, then the
    fastest of `batches` batches of `iters` calls."""
    import kohakutt

    for _ in range(2):
        fn()
    kohakutt.synchronize()
    best = float("inf")
    for _ in range(batches):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        kohakutt.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters)
    return best


def backward(grad, x, w, args, mask):
    s, p, d, groups = args
    dims = x.dim() - 2
    return torch.ops.aten.convolution_backward(
        grad, x, w, None, [s] * dims, [p] * dims, [d] * dims, False, [0] * dims,
        groups, mask
    )


def run_cell(cell, iters=10, ttnn=True):
    import kohakutt

    x, wt, grad, conv, args = tensors(cell)
    xt, wtt, gt = (t.bfloat16().to("tt") for t in (x, wt, grad))
    row = {"cell": cell[0], "shape": list(cell[1:])}
    os.environ["KTT_GROUPED_CONV"] = "own"

    # Forward, input gradient, weight gradient: numerics and route.
    xd, wd = x.double().requires_grad_(), wt.double().requires_grad_()
    want = conv(xd, wd, None, *args)
    want_gi, want_gw = torch.autograd.grad(want, (xd, wd), grad.double())
    xb, wb = x.bfloat16().requires_grad_(), wt.bfloat16().requires_grad_()
    ref = conv(xb, wb, None, *args)
    ref_gi, ref_gw = torch.autograd.grad(ref, (xb, wb), grad.bfloat16())
    kohakutt.synchronize()
    calls = {
        "fwd": (lambda: conv(xt, wtt, None, *args), want, ref),
        "dgrad": (lambda: backward(gt, xt, wtt, args, [True, False, False])[0],
                  want_gi, ref_gi),
        "wgrad": (lambda: backward(gt, xt, wtt, args, [False, True, False])[1],
                  want_gw, ref_gw),
    }
    for name, (fn, w64, w16) in calls.items():
        try:
            before = kohakutt.counters()
            got = fn()
            kohakutt.synchronize()
            after = kohakutt.counters()
            got = got.cpu()
        except Exception as e:  # noqa: BLE001
            row[name] = {"error": str(e).splitlines()[0][:200]}
            continue
        entry = {
            "cos": cosine(got, w64),
            "cpu_cos": cosine(w16, w64),
            "rel": max_rel(got, w64),
            "cpu_rel": max_rel(w16, w64),
            "ops": after["ops"] - before["ops"],
            "programs": after["programs"] - before["programs"],
        }
        entry["own"] = entry["ops"] == 1 and entry["programs"] >= 3
        before = kohakutt.counters()
        entry["ms"] = timed(fn, iters) * 1e3
        after = kohakutt.counters()
        entry["warm_misses"] = (after["programs"] - before["programs"]) - (
            after["desc_hits"] - before["desc_hits"]
        )
        row[name] = entry

    # TT-NN conv2d's forward on the same tensors.
    if ttnn:
        os.environ["KTT_GROUPED_CONV"] = "ttnn"
        try:
            fn = lambda: conv(xt, wtt, None, *args)  # noqa: E731
            got = fn().cpu()
            row["ttnn"] = {"cos": cosine(got, want), "ms": timed(fn, iters) * 1e3}
        except Exception as e:  # noqa: BLE001
            row["ttnn"] = {"error": str(e).splitlines()[0][:200]}
        os.environ["KTT_GROUPED_CONV"] = "own"
    return row


def fmt(v, spec):
    return format(v, spec) if isinstance(v, (int, float)) else "-"


def passes(e):
    """Cosine to float64 at least CPU bf16's (within 1e-4, or >= 0.9999)."""
    return "cos" in e and e["cos"] >= min(0.9999, e["cpu_cos"] - 1e-4)


CORES, CLOCK, DRAM_BPS = 110, 1.35e9, 512e9
MATMUL_CYCLES = 64  # one 32 x 32 x 32 tile product at HiFi4: 4 phases x 16 MVMUL


def roofline(shape):
    """Forward lower bound (microseconds) and its resource: the gather's sparse tile
    products on all cores at HiFi4 ("FPU") or the three programs' DRAM bytes at
    512 GB/s ("DRAM"), whichever is larger."""
    n, c, h, w, o, groups, k, s, d, p = shape
    one_d = h == 1
    kh, sh, dh, ph = (1, 1, 1, 0) if one_d else (k, s, d, p)
    oh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (w + 2 * p - d * (k - 1) - 1) // s + 1
    hp, wp = h + 2 * ph, w + 2 * p
    cp, ot = -(-c // 32) * 32, -(-o // 32)
    cg, og = c // groups, o // groups
    nt, kt, read = min(4, ot), 0, 0  # tile products per tap; channel tiles read per row
    for j0 in range(0, ot, nt):
        union = set()
        for j in range(j0, min(j0 + nt, ot)):
            first = (32 * j // og) * cg
            last = (min(32 * j + 31, o - 1) // og) * cg + cg - 1
            kt += last // 32 - first // 32 + 1
            union.update(range(first // 32, last // 32 + 1))
        read += len(union)
    m = -(-n * oh * ow // 32)
    fpu = m * kt * kh * k * MATMUL_CYCLES / CORES / CLOCK
    rows = (32 + d * (k - 1)) * kh if s == 1 else 32 * kh * k
    gather = m * rows * read * 64 + m * 32 * ot * 64
    layout = n * c * h * w * 2 + n * hp * wp * cp * 2 + m * 32 * ot * 64 + n * o * oh * ow * 2
    dram = (gather + layout) / DRAM_BPS
    return max(fpu, dram) * 1e6, "FPU" if fpu >= dram else "DRAM"


def markdown(rows):
    """The sweep as a markdown summary and table."""
    lines = []
    stats = {"cells": len(rows), "own": 0, "numerics": 0, "misses": 0}
    faster, slower, ttnn_cells = [], [], 0
    for r in rows:
        es = [r.get(k, {}) for k in ("fwd", "dgrad", "wgrad")]
        stats["own"] += all(e.get("own", False) for e in es)
        stats["numerics"] += all(passes(e) for e in es)
        stats["misses"] += sum(e.get("warm_misses", 0) for e in es)
        tn, f = r.get("ttnn", {}), r.get("fwd", {})
        if "ms" in tn and "ms" in f:
            ttnn_cells += 1
            (faster if f["ms"] <= tn["ms"] else slower).append(r["cell"])
    lines.append(
        f"Cells: {stats['cells']}; on Kohaku-TT programs only (fwd, dgrad, wgrad): "
        f"{stats['own']}; numerics at least CPU bf16's: {stats['numerics']}; "
        f"descriptor misses in warm calls: {stats['misses']}; TT-NN ran: {ttnn_cells}, "
        f"ours as fast or faster: {len(faster)}, slower: {len(slower)}"
        + (f" ({', '.join(slower)})" if slower else "")
        + ".\n"
    )
    lines += [
        "| cell | fwd ms | fwd bound us (binds) | of bound | TT-NN ms | TT-NN / ours |"
        " fwd cos (cpu bf16) | dgrad ms | dgrad cos (cpu) | wgrad ms | wgrad cos (cpu) |"
        " own | warm misses |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        f, di, wg, tn = (r.get(k, {}) for k in ("fwd", "dgrad", "wgrad", "ttnn"))
        own = all(e.get("own", False) for e in (f, di, wg))
        misses = sum(e.get("warm_misses", 0) for e in (f, di, wg))
        tt = fmt(tn.get("ms"), ".3f") if "error" not in tn else "n/a"
        ratio = tn["ms"] / f["ms"] if "ms" in tn and "ms" in f else None
        bound, binds = roofline(r["shape"])
        frac = bound / (f["ms"] * 1e3) if "ms" in f else None
        lines.append(
            f"| {r['cell']} | {fmt(f.get('ms'), '.3f')} | {bound:.1f} ({binds}) | "
            f"{fmt(frac, '.0%')} | {tt} | {fmt(ratio, '.2f')} | "
            f"{fmt(f.get('cos'), '.6f')} ({fmt(f.get('cpu_cos'), '.6f')}) | "
            f"{fmt(di.get('ms'), '.3f')} | {fmt(di.get('cos'), '.6f')} "
            f"({fmt(di.get('cpu_cos'), '.6f')}) | {fmt(wg.get('ms'), '.3f')} | "
            f"{fmt(wg.get('cos'), '.6f')} ({fmt(wg.get('cpu_cos'), '.6f')}) | "
            f"{'yes' if own else 'NO'} | {misses} |"
        )
    return "\n".join(lines) + "\n"


@pytest.mark.hw
def test_conv_sweep_smoke(native):
    for cell in grid_cells(small=True)[:12]:
        row = run_cell(cell, iters=2, ttnn=False)
        for name in ("fwd", "dgrad", "wgrad"):
            e = row[name]
            assert "error" not in e, (cell, name, e)
            assert e["own"], (cell, name, e)
            assert e["warm_misses"] == 0, (cell, name, e)
            assert passes(e), (cell, name, e)


def main():
    import kohakutt

    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", type=int, default=0)
    ap.add_argument("--json", required=True)
    ap.add_argument("--md")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument(
        "--only", choices=("grid", "grid2d", "grid1d", "models", "all"), default="all"
    )
    ap.add_argument("--no-ttnn", action="store_true")
    ap.add_argument("--match", default="", help="only cells whose name contains this")
    a = ap.parse_args()
    cells = []
    if a.only in ("grid", "all"):
        cells += grid_cells()
    if a.only == "grid2d":
        cells += grid_cells(dims=(2,))
    if a.only == "grid1d":
        cells += grid_cells(dims=(1,))
    if a.only in ("models", "all"):
        cells += MODELS
    cells = [c for c in cells if a.match in c[0]]
    kohakutt.open(a.chip)
    rows = []
    try:
        for cell in cells:
            t0 = time.time()
            rows.append(run_cell(cell, a.iters, not a.no_ttnn))
            r = rows[-1]
            print(
                f"{cell[0]}: fwd {r['fwd'].get('ms', 'err')} ttnn "
                f"{r.get('ttnn', {}).get('ms', r.get('ttnn', {}).get('error', '-'))} "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )
            with open(a.json, "w") as f:
                json.dump(rows, f, indent=1)
    finally:
        kohakutt.close()
    if a.md:
        with open(a.md, "w") as f:
            f.write(markdown(rows))


if __name__ == "__main__":
    main()
