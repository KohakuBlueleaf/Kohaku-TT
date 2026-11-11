"""GroupNorm (NCHW) at the tile level: one unit per (sample, group), a contiguous run
of tiles; mean and variance as exact fp32 sums over the run, then the affine chain.
Compared with float64 and F.group_norm at the SDXL UNet's shapes.
Usage: python examples/kernels/dsl_tile/group_norm.py [--shape N C H W] [--groups G]"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import kohakutt
import kohakutt.dsl as kd

sys.path.insert(0, str(Path(__file__).parent))
from common import timed


@kd.kernel
def group_norm(
    x: kd.Tensor,
    weight: kd.Tensor,
    bias: kd.Tensor,
    out: kd.Out,
    eps: kd.Runtime[float],
    groups: kd.Static[int] = 32,
):
    """x [N, C, H, W]; weight and bias as per-channel tiles ([C, 32, 32], see
    `channel_tiles`). Each (n, g) normalises its C / groups channels."""
    per_channel = (
        kd.num_tile_rows(x) // kd.size(x, 1) // kd.size(x, 0) * kd.num_tiles(x, -1)
    )
    cpg = kd.size(x, 1) // groups
    run = cpg * per_channel
    elements = kd.size(x, 2) * kd.size(x, 3) * cpg
    for u in kd.grid(kd.size(x, 0) * groups):
        blk = kd.load(x.tiles[u * run : (u + 1) * run]).cast(kd.f32)
        g0 = u % groups * cpg
        w = kd.load(weight.tiles[g0 : g0 + cpg]).each(per_channel)
        b = kd.load(bias.tiles[g0 : g0 + cpg]).each(per_channel)
        mean = kd.reduce_sum(blk, exact=True) / elements
        d = blk - mean.all()
        var = kd.reduce_sum(d * d, exact=True) / elements
        r = kd.rsqrt(var + eps)
        kd.store(
            out.tiles[u * run : (u + 1) * run], (d * r.all() * w + b).cast(out.dtype)
        )


def channel_tiles(v):
    """A per-channel vector [C] as one constant tile per channel ([C, 32, 32])."""
    return v.reshape(-1, 1, 1).expand(-1, 32, 32).contiguous()


def inputs(shape, seed=0):
    """x (bf16) and weight, bias ([C], bf16) on the open device, plus their channel
    tiles."""
    g = torch.Generator().manual_seed(seed)
    dev = kohakutt.device(0)
    x = (torch.randn(shape, generator=g) * 2 + 0.3).bfloat16()
    w = (torch.randn(shape[1], generator=g) * 0.5 + 1).bfloat16()
    b = (torch.randn(shape[1], generator=g) * 0.1).bfloat16()
    tiles = channel_tiles(w).to(dev), channel_tiles(b).to(dev)
    return x.to(dev), w.to(dev), b.to(dev), tiles


def errors(got, want):
    diff = (got.cpu().double() - want).abs()
    return diff.max().item(), diff.mean().item()


def compare(shape, groups=32, eps=1e-5):
    """Errors against float64 of the tile kernel and of F.group_norm on the device."""
    x, w, b, (wt, bt) = inputs(shape)
    want = F.group_norm(
        x.cpu().double(), groups, w.cpu().double(), b.cpu().double(), eps
    )
    return {
        "tile error": errors(group_norm(x, wt, bt, eps, groups=groups), want),
        "eager error": errors(F.group_norm(x, groups, w, b, eps), want),
    }


def bench(shape, groups=32, eps=1e-5):
    """{name: seconds}: the tile kernel and F.group_norm."""
    x, w, b, (wt, bt) = inputs(shape)
    return {
        "tile": timed(lambda: group_norm(x, wt, bt, eps, groups=groups)),
        "eager": timed(lambda: F.group_norm(x, groups, w, b, eps), reps=5, rounds=3),
    }


def main():
    """Runs the norm at one shape; prints errors, times and the plan."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=int, nargs=4, default=(2, 320, 64, 64))
    parser.add_argument("--groups", type=int, default=32)
    args = parser.parse_args()
    shape = tuple(args.shape)
    try:
        for key, value in {
            **compare(shape, args.groups),
            **bench(shape, args.groups),
        }.items():
            print(f"  {key}: {value}")
        x, w, b, (wt, bt) = inputs(shape)
        print(group_norm.plan(x, wt, bt, 1e-5, groups=args.groups).report())
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
