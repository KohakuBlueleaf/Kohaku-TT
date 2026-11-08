"""Device-profiler trace of the MLP-stack forward (4 blocks of Linear d -> 4d -> d,
GELU, residual, bf16) on 1 card or a 1x2 mesh, for one shape.

Run with the tt-metal device profiler on; the per-core CSV is written when the device
closes:

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR=out/devprof_2c_2048 \
        python experiments/mesh/mesh_devprof.py --cards 2 --dim 2048 --rows 1024

The forwards are separated by 30 ms of host sleep so each one shows up as its own
burst on the device timeline. Host side, each forward's enqueue time (return of the
async forward) and wall time (after synchronize) are printed as JSON lines, so the
device report (mesh_devprof_report.py) can set device busy time against host
dispatch time.
"""

import argparse
import json
import statistics
import time

import torch

import kohakutt


class MLP(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = torch.nn.Linear(dim, 4 * dim)
        self.down = torch.nn.Linear(4 * dim, dim)

    def forward(self, x):
        return x + self.down(torch.nn.functional.gelu(self.up(x)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", type=int, choices=(1, 2), required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--forwards", type=int, default=6)
    parser.add_argument(
        "--loop", type=int, default=0, help="back-to-back forwards, no profiler read"
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="--loop / --timed replay a captured trace",
    )
    parser.add_argument(
        "--timed", type=int, default=0, help="forwards, each synchronized: median ms"
    )
    parser.add_argument(
        "--plan",
        choices=("auto", "megatron", "dp"),
        default="auto",
        help="weights: automatic placement; up S(0) + down S(1); all replicated",
    )
    args = parser.parse_args()
    mesh = args.cards == 2
    dev = kohakutt.mesh([0, 1], shape=(1, 2)) if mesh else kohakutt.device(0)
    try:
        torch.manual_seed(0)
        g = torch.Generator().manual_seed(0)
        model = torch.nn.Sequential(*[MLP(args.dim) for _ in range(4)])
        model = model.to(torch.bfloat16).to(dev)
        if mesh and args.plan != "auto":
            for block in model:
                for name, p in block.named_parameters():
                    if args.plan == "dp":
                        p.data = kohakutt.replicate(p.data)
                    elif name.startswith("up."):
                        p.data = kohakutt.shard(p.data, 0)
                    elif name == "down.weight":
                        p.data = kohakutt.shard(p.data, 1)
                    else:
                        p.data = kohakutt.replicate(p.data)
        x = torch.randn(args.rows, args.dim, generator=g, dtype=torch.bfloat16).to(dev)
        if mesh:
            print(json.dumps({"placement_x": str(kohakutt.placement(x))}), flush=True)
        with torch.no_grad():
            for _ in range(3):
                model(x)
            torch.tt.synchronize()
            if args.timed:
                step = kohakutt.capture(model, x) if args.capture else model
                for _ in range(3):
                    step(x)
                torch.tt.synchronize()
                samples = []
                for _ in range(args.timed):
                    start = time.perf_counter()
                    step(x)
                    torch.tt.synchronize()
                    samples.append((time.perf_counter() - start) * 1e3)
                med = statistics.median(samples)
                print(json.dumps({"timed_ms_median": round(med, 3)}), flush=True)
                return
            if args.loop:
                step = kohakutt.capture(model, x) if args.capture else model
                step(x)
                torch.tt.synchronize()
                start = time.perf_counter()
                for _ in range(args.loop):
                    step(x)
                torch.tt.synchronize()
                per = (time.perf_counter() - start) * 1e3 / args.loop
                print(json.dumps({"loop_ms_per_forward": round(per, 3)}), flush=True)
                return
            # Drain the warm-up markers so the forwards below start a fresh read.
            torch.ops.ktt.read_device_profiler()
            stats0 = torch.tt.mesh_stats() if mesh else {}
            for i in range(args.forwards):
                time.sleep(0.03)
                start = time.perf_counter()
                y = model(x)
                enq = time.perf_counter()
                torch.tt.synchronize()
                end = time.perf_counter()
                print(
                    json.dumps(
                        {
                            "forward": i,
                            "enqueue_ms": round((enq - start) * 1e3, 3),
                            "wall_ms": round((end - start) * 1e3, 3),
                        }
                    ),
                    flush=True,
                )
            if mesh:
                stats1 = torch.tt.mesh_stats()
                per = {
                    k: (stats1[k] - stats0[k]) / args.forwards
                    for k in stats1
                    if isinstance(stats1[k], (int, float))
                    and stats1[k] != stats0.get(k)
                }
                print(json.dumps({"mesh_stats_per_forward": per}), flush=True)
                print(
                    json.dumps({"placement_y": str(kohakutt.placement(y))}), flush=True
                )
            torch.ops.ktt.read_device_profiler()
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
