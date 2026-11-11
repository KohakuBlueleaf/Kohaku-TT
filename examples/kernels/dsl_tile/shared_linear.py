"""Linear layers whose weight is shared down core columns (`share=g.col`): the top
core of each column reads a block of w once and multicasts it to the cores below,
each core reading only its own rows of a. Compared with `dsl_metal/gemm_km.py` (the
same layout, bit for bit), the unshared tile kernels in linear.py and eager torch."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import kohakutt
import kohakutt.dsl as kd

sys.path.insert(0, str(Path(__file__).parent))
import linear as unshared
from common import metal_example, timed


@kd.kernel(name="linear_shared")
def linear(a: kd.Tensor, w: kd.Tensor, out: kd.Out, nt: kd.Static[int] = 4):
    """out = a @ w for a [..., M, K] and w [K, N]."""
    g = kd.grid2d(rows=kd.num_tile_rows(a), cols=kd.cdiv(kd.num_tiles(w, -1), nt))
    for i, j in g.units():
        acc = kd.accumulator((1, nt))
        for k in kd.range(kd.num_tiles(a, -1)):
            wb = kd.load(w.tiles[k, j * nt : (j + 1) * nt], share=g.col)
            acc += kd.load(a.tiles[i, k]) @ wb
        kd.store(out.tiles[i, j * nt : (j + 1) * nt], acc.cast(out.dtype))


@kd.kernel(name="linear_gelu_shared")
def linear_gelu(
    a: kd.Tensor, w: kd.Tensor, b: kd.Tensor, out: kd.Out, nt: kd.Static[int] = 2
):
    """out = gelu(a @ w + b), GELU in its tanh form; b is [N]."""
    g = kd.grid2d(rows=kd.num_tile_rows(a), cols=kd.cdiv(kd.num_tiles(w, -1), nt))
    for i, j in g.units():
        acc = kd.accumulator((1, nt))
        for k in kd.range(kd.num_tiles(a, -1)):
            wb = kd.load(w.tiles[k, j * nt : (j + 1) * nt], share=g.col)
            acc += kd.load(a.tiles[i, k]) @ wb
        x = acc + kd.load(b.tiles[0, j * nt : (j + 1) * nt]).row()
        inner = unshared.GELU_C * (x + 0.044715 * x * x * x)
        kd.store(
            out.tiles[i, j * nt : (j + 1) * nt], (0.5 * x * (1.0 + kd.tanh(inner)))
        )


@kd.kernel(name="linear_2d")
def linear_2d(a: kd.Tensor, w: kd.Tensor, out: kd.Out, nt: kd.Static[int] = 4):
    """out = a @ w with a shared along core rows and w down core columns."""
    g = kd.grid2d(rows=kd.num_tile_rows(a), cols=kd.cdiv(kd.num_tiles(w, -1), nt))
    for i, j in g.units():
        acc = kd.accumulator((1, nt))
        for k in kd.range(kd.num_tiles(a, -1)):
            ab = kd.load(a.tiles[i, k], share=g.row)
            wb = kd.load(w.tiles[k, j * nt : (j + 1) * nt], share=g.col)
            acc += ab @ wb
        kd.store(out.tiles[i, j * nt : (j + 1) * nt], acc.cast(out.dtype))


@kd.kernel(name="linear_gelu_2d")
def linear_gelu_2d(
    a: kd.Tensor, w: kd.Tensor, b: kd.Tensor, out: kd.Out, nt: kd.Static[int] = 2
):
    """out = gelu(a @ w + b) with both operands shared."""
    g = kd.grid2d(rows=kd.num_tile_rows(a), cols=kd.cdiv(kd.num_tiles(w, -1), nt))
    for i, j in g.units():
        acc = kd.accumulator((1, nt))
        for k in kd.range(kd.num_tiles(a, -1)):
            ab = kd.load(a.tiles[i, k], share=g.row)
            wb = kd.load(w.tiles[k, j * nt : (j + 1) * nt], share=g.col)
            acc += ab @ wb
        x = acc + kd.load(b.tiles[0, j * nt : (j + 1) * nt]).row()
        inner = unshared.GELU_C * (x + 0.044715 * x * x * x)
        kd.store(
            out.tiles[i, j * nt : (j + 1) * nt], (0.5 * x * (1.0 + kd.tanh(inner)))
        )


def gemm_km():
    """The metal-level block matmul (gemm_km.py) as a callable (a, w)."""
    metal_example("gemm_km")
    return lambda a, w: torch.ops.kohakutt_example.gemm_km(a, w, False)


def inputs(tokens, inner, cols, seed=0):
    """a [1, T, K], w [K, N], b [N] bf16 on the open device."""
    g = torch.Generator().manual_seed(seed)
    dev = kohakutt.device(0)
    a = torch.randn(1, tokens, inner, generator=g).bfloat16()
    w = (torch.randn(inner, cols, generator=g) / inner**0.5).bfloat16()
    b = (torch.randn(cols, generator=g) / 4).bfloat16()
    return a.to(dev), w.to(dev), b.to(dev)


def compare(tokens, inner, cols, seed=0):
    """{"linear = metal gemm": bool, "gelu tile error", "gelu eager error"}: errors
    (max, mean) against float64; the tile kernel in the layout gemm_km's plan takes."""
    a, w, b = inputs(tokens, inner, cols, seed)
    one_d = metal_example("gemm_km").values_for(a, w, False)["one_d"]
    tile = (linear if one_d else linear_2d)(a, w).cpu()
    out = {"linear = metal gemm": torch.equal(tile, gemm_km()(a, w).cpu())}
    x = a.cpu().double() @ w.cpu().double() + b.cpu().double()
    want = F.gelu(x, approximate="tanh")
    eager = F.gelu(F.linear(a, w.t(), b), approximate="tanh")
    out["gelu tile error"] = unshared.errors(linear_gelu(a, w, b).cpu(), want)
    out["gelu eager error"] = unshared.errors(eager.cpu(), want)
    return out


def bench(tokens, inner, cols):
    """{name: seconds} of the shared and unshared tile kernels, gemm_km and eager."""
    a, w, b = inputs(tokens, inner, cols)
    return {
        "linear shared": timed(lambda: linear(a, w)),
        "linear unshared": timed(lambda: unshared.linear(a, w)),
        "gemm_km": timed(lambda: gemm_km()(a, w)),
        "a @ w": timed(lambda: a @ w),
        "gelu shared": timed(lambda: linear_gelu(a, w, b)),
        "gelu unshared": timed(lambda: unshared.linear_gelu(a, w, b)),
        "gelu eager": timed(lambda: F.gelu(F.linear(a, w.t(), b), approximate="tanh")),
    }


def main():
    """Runs both kernels at one shape; prints agreement, errors, times and the plan."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--inner", type=int, default=768)
    parser.add_argument("--cols", type=int, default=3072)
    args = parser.parse_args()
    shape = (args.tokens, args.inner, args.cols)
    try:
        for key, value in compare(*shape).items():
            print(f"  {key}: {value}")
        for key, value in bench(*shape).items():
            print(f"  {key}: {value * 1e6:.0f} us")
        a, w, b = inputs(*shape)
        print(linear_gelu.plan(a, w, b).report())
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
