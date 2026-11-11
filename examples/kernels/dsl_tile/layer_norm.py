"""LayerNorm written at the tile level: one launch, the row read once and held in L1,
the mean and then the variance about it as exact fp32 sums, the affine chain staged
over two functions (it reads five blocks). Compared with float64 and F.layer_norm.
Usage: python examples/kernels/dsl_tile/layer_norm.py [--tokens T] [--width W]"""

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
def layer_norm(
    x: kd.Tensor,
    weight: kd.Tensor,
    bias: kd.Tensor,
    out: kd.Out,
    eps: kd.Runtime[float],
    group: kd.Static[int] = 4,
):
    """out = (x - mean) / sqrt(var + eps) * weight + bias along the last dim. With
    more rows than cores, each row may be split over `group` cores."""
    g = kd.resident(weight.tiles[0, :])
    b = kd.resident(bias.tiles[0, :])
    for t in kd.grid(kd.num_tile_rows(x), group=group):
        row = kd.load(x.tiles[t, :], pad=0).cast(kd.f32)
        mean = kd.reduce_sum(row, exact=True) / kd.size(x, -1)
        d = row - mean.col()
        var = kd.reduce_sum(d * d, exact=True) / kd.size(x, -1)
        r = kd.rsqrt(var + eps)
        kd.store(out.tiles[t, :], (d * r.col() * g.row() + b.row()).cast(out.dtype))


def inputs(shape, seed=0):
    """x, weight and bias ([W]) in bf16 on the open device."""
    g = torch.Generator().manual_seed(seed)
    dev = kohakutt.device(0)
    x = (torch.randn(shape, generator=g) * 2 + 0.5).bfloat16()
    w = (torch.randn(shape[-1], generator=g) * 0.5 + 1).bfloat16()
    b = (torch.randn(shape[-1], generator=g) * 0.1).bfloat16()
    return x.to(dev), w.to(dev), b.to(dev)


def errors(got, want):
    """(max, mean) absolute error of `got` against float64 `want`."""
    diff = (got.cpu().double() - want).abs()
    return diff.max().item(), diff.mean().item()


def compare(shape, eps=1e-5):
    """Errors against float64 of the tile kernel and of F.layer_norm on the device,
    for the fp32 and the bf16 output."""
    x, w, b = inputs(shape)
    want = F.layer_norm(
        x.cpu().double(), (shape[-1],), w.cpu().double(), b.cpu().double(), eps
    )
    eager = F.layer_norm(x.float(), (shape[-1],), w.float(), b.float(), eps)
    return {
        "tile fp32 error": errors(
            layer_norm(x, w, b, eps, out_dtype=torch.float32), want
        ),
        "eager fp32 error": errors(eager, want),
        "tile bf16 error": errors(layer_norm(x, w, b, eps), want),
        "eager bf16 error": errors(eager.bfloat16(), want),
    }


def bench(shape, eps=1e-5):
    """{name: seconds}: the tile kernel and F.layer_norm (bf16 in, bf16 out)."""
    x, w, b = inputs(shape)
    return {
        "tile": timed(lambda: layer_norm(x, w, b, eps)),
        "eager": timed(lambda: F.layer_norm(x, (shape[-1],), w, b, eps)),
    }


def main():
    """Runs the norm at one shape; prints errors, times and the plan."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--width", type=int, default=1024)
    args = parser.parse_args()
    shape = (1, args.tokens, args.width)
    try:
        for key, value in {**compare(shape), **bench(shape)}.items():
            print(f"  {key}: {value}")
        x, w, b = inputs(shape)
        print(layer_norm.plan(x, w, b, 1e-5).report())
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
