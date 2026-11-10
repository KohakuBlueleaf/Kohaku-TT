"""Kernel sweeps: correctness, speed and compile counts over a family's shape grid.

    python -m kohakutt.tools.sweep matmul --grid audit --dtypes bf16 bfp8
        --json out.json

Each cell runs on both routes (Kohaku-TT's kernels, and TT-NN's path with
`KTT_GEMM=0 KTT_CUSTOM_KERNELS=0`), each in its own process. Per cell and route: cosine
and relative RMS error against CPU float64 (and CPU at the same dtype), warm ms, TFLOPS,
and the kernel binaries compiled during the warm calls. `--report` writes a markdown
summary with the gate checks (curve drops, compiles, errors).
"""

import argparse
import json
import os
import subprocess
import sys
import time

GRIDS = {
    "matmul": {
        "smoke": [(256, 256, 256), (1024, 1000, 768), (2048, 4096, 4096)],
        "audit": (
            [
                (m, k, n)
                for m in (32, 128, 512, 2048, 8192)
                for k in (256, 1000, 4096)
                for n in (256, 1000, 4096, 11008)
            ]
            + [(s, s, s) for s in (512, 1024, 2048, 3072, 4096, 6144, 8192)]
        ),
        "curve_m": [
            (m, 4096, 4096)
            for m in (
                32,
                64,
                128,
                256,
                512,
                1024,
                1536,
                2048,
                3072,
                4096,
                6144,
                8192,
                12288,
                16384,
            )
        ],
        "curve_k": [
            (4096, k, 4096)
            for k in (
                128,
                256,
                512,
                1000,
                1024,
                2000,
                2048,
                3000,
                4096,
                6144,
                8192,
                9742,
                12288,
                16384,
            )
        ],
        "curve_n": [
            (4096, 4096, n)
            for n in (
                128,
                256,
                512,
                1000,
                1024,
                2048,
                3000,
                4096,
                6144,
                8192,
                11008,
                16384,
            )
        ],
        "curve_sq": [
            (s, s, s)
            for s in (
                256,
                512,
                1024,
                1536,
                2048,
                2560,
                3072,
                4096,
                5120,
                6144,
                7168,
                8192,
                9742,
                11585,
                13777,
                16384,
            )
        ],
    },
}

ROUTES = {
    "own": {"KTT_GEMM": "1"},
    "ttnn": {"KTT_GEMM": "0", "KTT_CUSTOM_KERNELS": "0"},
}


def _kernel_variants():
    """Compiled kernel binaries in the TT-Metal caches (every `tt-metal-cache*`
    directory under the cache root; each holds build directories with one hash directory
    per kernel variant)."""
    parent = os.environ.get("TT_METAL_CACHE", os.path.expanduser("~/.cache"))
    if os.path.basename(parent.rstrip("/")).startswith("tt-metal-cache"):
        parent = os.path.dirname(parent.rstrip("/"))
    count = 0
    if not os.path.isdir(parent):
        return 0
    for root in os.scandir(parent):
        if not (root.is_dir() and root.name.startswith("tt-metal-cache")):
            continue
        for build in os.scandir(root.path):
            kernels = os.path.join(build.path, "kernels")
            if not os.path.isdir(kernels):
                continue
            for kernel in os.scandir(kernels):
                if kernel.is_dir():
                    count += sum(1 for _ in os.scandir(kernel.path))
    return count


def _errors(y, ref):
    import torch

    y = y.double().flatten()
    ref = ref.flatten()
    cos = torch.nn.functional.cosine_similarity(y, ref, dim=0).item()
    rel = ((y - ref).norm() / ref.norm()).item()
    return cos, rel, bool(torch.isfinite(y).all())


def run_matmul_cells(cells, dtype, reps):
    """Runs every (M, K, N) cell as F.linear on this process's route; yields records."""
    import torch
    import torch.nn.functional as F

    import kohakutt

    dev = kohakutt.device(0)
    for m, k, n in cells:
        g = torch.Generator().manual_seed(m * 7 + k * 13 + n)
        x = torch.randn(m, k, generator=g).to(torch.bfloat16)
        w = (torch.randn(n, k, generator=g) / k**0.5).to(torch.bfloat16)
        ref = x.double() @ w.double().t()
        rec = {"family": "matmul", "shape": [m, k, n], "dtype": dtype}
        try:
            xt, wt = x.to(dev), w.to(dev)
            if dtype != "bf16":
                kohakutt.block_float_(wt, dtype)
            y = F.linear(xt, wt)
            out = y.cpu()
            F.linear(xt, wt)
            kohakutt.synchronize()
            before = _kernel_variants()
            programs = kohakutt.counters()["programs"]
            t0 = time.perf_counter()
            for _ in range(reps):
                F.linear(xt, wt)
            kohakutt.synchronize()
            ms = (time.perf_counter() - t0) / reps * 1e3
            rec["warm_compiles"] = _kernel_variants() - before
            rec["own_programs_per_call"] = (
                kohakutt.counters()["programs"] - programs
            ) / reps
            cos, rel, finite = _errors(out, ref)
            rec.update(
                ms=round(ms, 4),
                tflops=round(2 * m * k * n / ms / 1e9, 2),
                cos=round(cos, 7),
                rel_rms=float(f"{rel:.3e}"),
                finite=finite,
            )
            if dtype == "bf16":
                ccos, crel, _ = _errors(x @ w.t(), ref)
                rec.update(cpu_cos=round(ccos, 7), cpu_rel_rms=float(f"{crel:.3e}"))
            del xt, wt, y
        except Exception as e:  # noqa: BLE001 - a failing cell is a record, not a stop
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        yield rec


def _worker(args):
    cells = []
    for grid in args.grid:
        cells += GRIDS[args.family][grid]
    for dtype in args.dtypes:
        for rec in run_matmul_cells(cells, dtype, args.reps):
            rec["route"] = args.route
            print(json.dumps(rec), flush=True)


def _drops(records, key):
    """Pairs of neighbouring cells along a curve whose TFLOPS fall more than 15 %."""
    out = []
    for a, b in zip(records, records[1:]):
        if (
            "tflops" in a
            and "tflops" in b
            and 2 * a["shape"][0] * a["shape"][1] * a["shape"][2] > 1e9
            and b["tflops"] < 0.85 * a["tflops"]
        ):
            out.append((a["shape"], b["shape"], a["tflops"], b["tflops"]))
    return out


def report(records, path):
    """Markdown summary: per dtype and route, medians, worst error, compiles, drops."""
    lines = ["# Sweep report", ""]
    by = {}
    for r in records:
        by.setdefault((r["dtype"], r["route"]), []).append(r)
    lines += [
        "| dtype | route | cells | errors | median TFLOPS (>0.5 TFLOP) | min cos | "
        "max warm compiles |",
        "|---|---|---|---|---|---|---|",
    ]
    for (dtype, route), rs in sorted(by.items()):
        ok = [r for r in rs if "error" not in r]
        big = sorted(
            r["tflops"]
            for r in ok
            if 2 * r["shape"][0] * r["shape"][1] * r["shape"][2] > 5e11
        )
        med = big[len(big) // 2] if big else float("nan")
        lines.append(
            f"| {dtype} | {route} | {len(rs)} | {len(rs) - len(ok)} | {med:.1f} | "
            f"{min((r['cos'] for r in ok), default=float('nan')):.6f} | "
            f"{max((r['warm_compiles'] for r in ok), default=0)} |"
        )
    lines += [
        "",
        "## Cells",
        "",
        "| shape | dtype | own ms | own TFLOPS | TT-NN ms | TT-NN TFLOPS | "
        "own / TT-NN | own cos | TT-NN cos | CPU cos |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    cells = {}
    for r in records:
        cells.setdefault((tuple(r["shape"]), r["dtype"]), {})[r["route"]] = r
    for (shape, dtype), rr in cells.items():
        o, t = rr.get("own", {}), rr.get("ttnn", {})
        ratio = (t["ms"] / o["ms"]) if "ms" in o and "ms" in t else float("nan")
        lines.append(
            f"| {'x'.join(map(str, shape))} | {dtype} | "
            f"{o.get('ms', o.get('error', '-'))} | "
            f"{o.get('tflops', '-')} | {t.get('ms', t.get('error', '-'))} | "
            f"{t.get('tflops', '-')} | "
            f"{ratio:.2f} | {o.get('cos', '-')} | {t.get('cos', '-')} | "
            f"{o.get('cpu_cos', '-')} |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("family", choices=sorted(GRIDS))
    p.add_argument("--grid", nargs="+", default=["smoke"])
    p.add_argument("--dtypes", nargs="+", default=["bf16"])
    p.add_argument("--routes", nargs="+", default=["own", "ttnn"])
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--json", default=None)
    p.add_argument("--report", default=None)
    p.add_argument("--route", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.route:
        _worker(args)
        return
    records = []
    for route in args.routes:
        env = dict(os.environ, **ROUTES[route])
        cmd = [
            sys.executable,
            "-m",
            "kohakutt.tools.sweep",
            args.family,
            "--route",
            route,
            "--grid",
            *args.grid,
            "--dtypes",
            *args.dtypes,
            "--reps",
            str(args.reps),
        ]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        for line in proc.stdout.splitlines():
            if line.startswith("{"):
                records.append(json.loads(line))
                print(line, flush=True)
        if proc.returncode != 0:
            print(
                f"route {route} exited {proc.returncode}: {proc.stderr[-2000:]}",
                flush=True,
            )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(records, f, indent=1)
    if args.report:
        report(records, args.report)
        for grid in args.grid:
            if grid.startswith("curve"):
                cells = [tuple(c) for c in GRIDS[args.family][grid]]
                for dtype in args.dtypes:
                    own = [
                        r
                        for r in records
                        if r["route"] == "own"
                        and r["dtype"] == dtype
                        and tuple(r["shape"]) in cells
                    ]
                    own.sort(key=lambda r: cells.index(tuple(r["shape"])))
                    for d in _drops(own, grid):
                        print(f"DROP {grid} {dtype}: {d}", flush=True)


if __name__ == "__main__":
    main()
