"""RMS norm's backward pass written by hand at the tile level, as two kernels: dx
(rows: the row statistic and the exact sum of x * w * dy) and dweight (columns of
tiles: exact sums over rows). Checked against torch autograd in float64.
Usage: python examples/kernels/dsl_tile/rms_norm_backward.py [--tokens T] [--width W]"""

import argparse
import sys
from pathlib import Path

import torch

import kohakutt
import kohakutt.dsl as kd

sys.path.insert(0, str(Path(__file__).parent))
from common import timed


@kd.kernel
def rms_norm_backward(
    x: kd.Tensor,
    dy: kd.Tensor,
    weight: kd.Tensor,
    dx: kd.Out,
    rstd: kd.Out,
    eps: kd.Runtime[float],
):
    """y = x * r * w with r = rsqrt(mean(x^2) + eps): dx = r * w * dy - x * r^3 *
    mean(x * w * dy); rstd = r per row ([..., R, 1], fp32) for the dweight kernel."""
    g = kd.resident(weight.tiles[0, :])
    for t in kd.grid(kd.num_tile_rows(x)):
        row = kd.load(x.tiles[t, :], pad=0).cast(kd.f32)
        d = kd.load(dy.tiles[t, :], pad=0).cast(kd.f32)
        r = kd.rsqrt(kd.reduce_sum(row * row, exact=True) / kd.size(x, -1) + eps)
        gd = d * g.row()
        c = kd.reduce_sum(row * gd, exact=True) / kd.size(x, -1)
        kd.store(
            dx.tiles[t, :], (r.col() * gd - row * (r * r * r * c).col()).cast(dx.dtype)
        )
        kd.store(rstd.tiles[t], r)


@kd.kernel
def rms_norm_dweight(x: kd.Tensor, dy: kd.Tensor, rstd: kd.Tensor, dweight: kd.Out):
    """dweight = sum over rows of dy * x * r ([D], fp32): one unit per column of
    tiles, its tiles read with a stride of the row's tiles."""
    rows, wt = kd.num_tile_rows(x), kd.num_tiles(x, -1)
    for c in kd.grid(wt):
        xc = kd.load(x.tiles[c : rows * wt : wt]).cast(kd.f32)
        dc = kd.load(dy.tiles[c : rows * wt : wt]).cast(kd.f32)
        r = kd.load(rstd.tiles[0:rows])
        kd.store(
            dweight.tiles[c], kd.reduce_sum(dc * (xc * r.col()), dim=-2, exact=True)
        )


def backward(x, dy, w, eps=1e-6, out_dtype=None):
    """(dx, dweight) by the two kernels; x, dy [T, D] (or [..., T, D]), w [D]; dx in
    x's dtype unless `out_dtype`."""
    dx, rstd = rms_norm_backward(x, dy, w, eps, out_dtype=out_dtype)
    return dx, rms_norm_dweight(x, dy, rstd)


def eager_backward(x, dy, w, eps=1e-6):
    """The same formulas as torch ops on the device (fp32): (dx, dweight)."""
    xf, gd = x.float(), dy.float() * w
    r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    c = (xf * gd).mean(-1, keepdim=True)
    return r * gd - xf * (r * r * r * c), (dy.float() * xf * r).sum(0)


def autograd(x, dy, w, eps=1e-6):
    """(dx, dweight) of torch autograd in float64 on the CPU."""
    x = x.cpu().double().requires_grad_()
    w = w.cpu().double().requires_grad_()
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
    y.backward(dy.cpu().double())
    return x.grad, w.grad


def inputs(shape, seed=0):
    """x, dy (bf16) and weight ([D], fp32) on the open device."""
    g = torch.Generator().manual_seed(seed)
    dev = kohakutt.device(0)
    x = torch.randn(shape, generator=g).bfloat16()
    dy = torch.randn(shape, generator=g).bfloat16()
    w = torch.rand(shape[-1], generator=g) + 0.5
    return x.to(dev), dy.to(dev), w.to(dev)


def compare(shape, eps=1e-6):
    """Relative errors of dx and dweight against float64 autograd, for the tile
    kernels and for eager autograd on the device."""
    x, dy, w = inputs(shape)
    want_dx, want_dw = autograd(x, dy, w, eps)
    dx, dw = backward(x, dy, w, eps, out_dtype=torch.float32)
    xe = x.float().requires_grad_()
    we = w.clone().requires_grad_()
    ye = xe * torch.rsqrt(xe.pow(2).mean(-1, keepdim=True) + eps) * we
    ye.backward(dy.float())

    def rel(got, want):
        got = got.cpu().double()
        return ((got - want).abs().max() / want.abs().max()).item()

    return {
        "tile dx error": rel(dx, want_dx),
        "tile dweight error": rel(dw, want_dw),
        "eager dx error": rel(xe.grad, want_dx),
        "eager dweight error": rel(we.grad, want_dw),
    }


def bench(shape, eps=1e-6):
    """{name: seconds}: each kernel, both, and the same formulas as eager ops."""
    x, dy, w = inputs(shape)
    _, rstd = rms_norm_backward(x, dy, w, eps)
    return {
        "tile dx kernel": timed(lambda: rms_norm_backward(x, dy, w, eps)),
        "tile dweight kernel": timed(lambda: rms_norm_dweight(x, dy, rstd)),
        "tile both": timed(lambda: backward(x, dy, w, eps)),
        "eager formulas": timed(
            lambda: eager_backward(x, dy, w, eps), reps=5, rounds=3
        ),
    }


def main():
    """Runs the backward at one shape; prints errors and times."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--width", type=int, default=1536)
    args = parser.parse_args()
    shape = (args.tokens, args.width)
    try:
        for key, value in {**compare(shape), **bench(shape)}.items():
            print(f"  {key}: {value}")
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
