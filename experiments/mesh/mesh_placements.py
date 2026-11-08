"""Placement trace of one MLP-stack forward on a 1x2 mesh: per aten op, the input and
output placements, local shard shapes and the mesh conversions it triggered.

    python experiments/mesh/mesh_placements.py --dim 2048 --rows 1024
"""

import argparse

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import kohakutt

CONVERSIONS = (
    "all_gathers",
    "reduce_scatters",
    "all_reduces",
    "partitions",
    "all_to_alls",
)


class MLP(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = torch.nn.Linear(dim, 4 * dim)
        self.down = torch.nn.Linear(4 * dim, dim)

    def forward(self, x):
        return x + self.down(torch.nn.functional.gelu(self.up(x)))


def describe(t):
    if not isinstance(t, torch.Tensor):
        return None
    return f"{tuple(t.shape)}:{kohakutt.placement(t)}"


class PlacementTrace(TorchDispatchMode):
    """Prints each aten op with argument and result placements and its conversions."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        before = torch.tt.mesh_stats()
        out = func(*args, **(kwargs or {}))
        after = torch.tt.mesh_stats()
        conv = {k: after[k] - before[k] for k in CONVERSIONS if after[k] != before[k]}
        ins = [d for d in map(describe, args) if d]
        outs = describe(out) if isinstance(out, torch.Tensor) else None
        print(f"{func!s:28s} {ins} -> {outs} {conv if conv else ''}", flush=True)
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True)
    args = parser.parse_args()
    dev = kohakutt.mesh([0, 1], shape=(1, 2))
    try:
        torch.manual_seed(0)
        model = torch.nn.Sequential(*[MLP(args.dim) for _ in range(2)])
        model = model.to(torch.bfloat16).to(dev)
        for name, p in model.named_parameters():
            print(f"param {name:16s} {describe(p)}")
        x = torch.randn(args.rows, args.dim, dtype=torch.bfloat16).to(dev)
        with torch.no_grad():
            model(x)
            torch.tt.synchronize()
            with PlacementTrace():
                model(x)
            torch.tt.synchronize()
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
