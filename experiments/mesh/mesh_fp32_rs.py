"""fp32 P(sum) completion on the 1x2 mesh, checked bit for bit: a column -> row Linear
pair in fp32 leaves P(sum) of shape [rows, dim] ([1, 1, rows, dim] with --view4); each
chip's partial is read back, and the sum of the two in fp32 on the CPU is the
reference. Completes to rows sharded and to replicated, compares each result with the
reference exactly, and times each completion eagerly and per call inside a captured
trace of --loop back-to-back completions. Prints one JSON line.

    KTT_MESH_FP32_RS=1 python experiments/mesh/mesh_fp32_rs.py --rows 4096 --dim 4096
    KTT_MESH_FP32_RS=0 python experiments/mesh/mesh_fp32_rs.py --rows 4096 --dim 4096
"""

import argparse
import json
import statistics
import time

import torch

import kohakutt


def _eager_ms(fn, x, n):
    fn(x)
    torch.tt.synchronize()
    times = []
    for _ in range(n):
        start = time.perf_counter()
        fn(x)
        torch.tt.synchronize()
        times.append((time.perf_counter() - start) * 1e3)
    return round(statistics.median(times), 4)


def _trace_ms(fn, x, loop, n):
    def many(t):
        out = None
        for _ in range(loop):
            out = fn(t)
        return out

    step = kohakutt.capture(many, x, warmup=1)
    try:
        step(x)
        torch.tt.synchronize()
        times = []
        for _ in range(n):
            start = time.perf_counter()
            step(x)
            torch.tt.synchronize()
            times.append((time.perf_counter() - start) * 1e3 / loop)
    finally:
        step.release()
    return round(statistics.median(times), 4)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rows", type=int, default=512)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--view4", action="store_true")
    p.add_argument("--timed", type=int, default=5)
    p.add_argument("--loop", type=int, default=10)
    args = p.parse_args()
    mesh = kohakutt.mesh([0, 1], shape=(1, 2))
    rec = {"shape": [args.rows, args.dim], "view4": args.view4}
    try:
        torch.manual_seed(0)
        up = kohakutt.colwise(torch.nn.Linear(args.dim, 4 * args.dim, bias=False))
        down = kohakutt.rowwise(torch.nn.Linear(4 * args.dim, args.dim, bias=False))
        up, down = up.to(mesh), down.to(mesh)
        g = torch.Generator().manual_seed(1)
        lead = (1, 1) if args.view4 else ()
        x_host = torch.randn(*lead, args.rows, args.dim, generator=g)
        x = kohakutt.replicate(x_host.to(mesh))
        with torch.no_grad():
            partial = down(torch.relu(up(x)))
            rec["partial"] = kohakutt.placement(partial)
            parts = [torch.ops.ktt.local_shard(partial, c) for c in range(2)]
            ref = (parts[0] + parts[1])[..., : args.rows, : args.dim]
            row_dim = len(lead)
            specs = {
                "sharded": ",".join(["R", f"S({row_dim})"]),
                "replicated": "R,R",
            }
            for name, spec in specs.items():
                complete = lambda t, s=spec: torch.ops.ktt.distribute(t, s)  # noqa: E731
                torch.tt.synchronize()
                before = torch.tt.mesh_stats()
                y = complete(partial)
                torch.tt.synchronize()
                after = torch.tt.mesh_stats()
                got = y.cpu()
                diff = (got - ref).abs()
                rec[name] = {
                    "placement": kohakutt.placement(y),
                    "conversions": {
                        k: after[k] - before[k] for k in after if after[k] != before[k]
                    },
                    "equal": bool(torch.equal(got, ref)),
                    "max_abs_diff": float(diff.max()),
                    "eager_ms": _eager_ms(complete, partial, args.timed),
                    "trace_ms": _trace_ms(complete, partial, args.loop, args.timed),
                }
    finally:
        kohakutt.close()
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
