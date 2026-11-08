"""Placement marks set in a model's __init__ (kohakutt.colwise / rowwise / replicate):
4 MLP blocks (Linear d -> 4d -> d, GELU, residual) with the up projection marked
column-parallel, the down projection row-parallel and the norm replicated, on 1 card or
a 1x2 mesh. Prints each parameter's placement, the conversions per forward, the median
of synchronized warm forwards, and the output against CPU float64 beside CPU in
the same dtype.

    python experiments/mesh/mesh_marks.py --cards 2 --dim 4096 --rows 2048
    python experiments/mesh/mesh_marks.py --cards 1 --dim 4096 --rows 2048
"""

import argparse
import statistics
import time

import torch
from torch import nn

import kohakutt

CONVERSIONS = (
    "all_gathers",
    "reduce_scatters",
    "all_reduces",
    "partitions",
    "all_to_alls",
    "gathers",
)


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = kohakutt.replicate(nn.LayerNorm(d))
        self.up = kohakutt.colwise(nn.Linear(d, 4 * d))
        self.down = kohakutt.rowwise(nn.Linear(4 * d, d))

    def forward(self, x):
        return x + self.down(nn.functional.gelu(self.up(self.norm(x))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", type=int, choices=(1, 2), required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--timed", type=int, default=10)
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    cpu_model = nn.Sequential(*[Block(args.dim) for _ in range(4)]).to(dtype)
    x_cpu = torch.randn(args.rows, args.dim, dtype=dtype)
    with torch.no_grad():
        ref = nn.Sequential(*[Block(args.dim) for _ in range(4)]).double()
        ref.load_state_dict({k: v.double() for k, v in cpu_model.state_dict().items()})
        want = ref(x_cpu.double())
        cpu_same = cpu_model(x_cpu)
    dev = kohakutt.mesh([0, 1], shape=(1, 2)) if args.cards == 2 else kohakutt.device(0)
    try:
        model = cpu_model.to(dev)
        if args.cards == 2:
            for name, p in list(model.named_parameters())[:6]:
                print(f"param {name:16s} {tuple(p.shape)} {kohakutt.placement(p)}")
        x = x_cpu.to(dev)
        with torch.no_grad():
            for _ in range(3):
                model(x)
            torch.tt.synchronize()
            before = torch.tt.mesh_stats() if args.cards == 2 else {}
            out = model(x)
            torch.tt.synchronize()
            if args.cards == 2:
                after = torch.tt.mesh_stats()
                conv = {
                    k: after[k] - before[k]
                    for k in CONVERSIONS
                    if after[k] != before[k]
                }
                print("conversions per forward", conv)
            samples = []
            for _ in range(args.timed):
                start = time.perf_counter()
                model(x)
                torch.tt.synchronize()
                samples.append((time.perf_counter() - start) * 1e3)
        print(
            f"{args.cards} card(s) d={args.dim} rows={args.rows} {args.dtype}: "
            f"{statistics.median(samples):.3f} ms (median of {args.timed})"
        )
        print("device ", kohakutt.format_stats(kohakutt.diff_stats(out.cpu(), want)))
        print("cpu    ", kohakutt.format_stats(kohakutt.diff_stats(cpu_same, want)))
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
