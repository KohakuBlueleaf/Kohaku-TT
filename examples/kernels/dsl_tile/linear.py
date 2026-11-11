"""A linear layer with its activation at the tile level: a @ w summed on the matrix
unit, bias and activation applied on the registers before the pack, one launch.
Compared with the metal-level matmul and with F.linear + the activation run eagerly.
Usage: python examples/kernels/dsl_tile/linear.py [--tokens T] [--inner K] [--cols N]"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import kohakutt
import kohakutt.dsl as kd

sys.path.insert(0, str(Path(__file__).parent))
from common import metal_example, timed

# sqrt(2 / pi), the constant of GELU's tanh form.
GELU_C = math.sqrt(2 / math.pi)


@kd.kernel
def linear(a: kd.Tensor, w: kd.Tensor, out: kd.Out, nt: kd.Static[int] = 4):
    """out = a @ w for a [..., M, K] and w [K, N], in blocks of nt output tiles."""
    g = kd.grid2d(rows=kd.num_tile_rows(a), cols=kd.cdiv(kd.num_tiles(w, -1), nt))
    for i, j in g.units():
        acc = kd.accumulator((1, nt))
        for k in kd.range(kd.num_tiles(a, -1)):
            acc += kd.load(a.tiles[i, k]) @ kd.load(w.tiles[k, j * nt : (j + 1) * nt])
        kd.store(out.tiles[i, j * nt : (j + 1) * nt], acc.cast(out.dtype))


@kd.kernel
def linear_silu(a: kd.Tensor, w: kd.Tensor, out: kd.Out, nt: kd.Static[int] = 4):
    """out = silu(a @ w)."""
    g = kd.grid2d(rows=kd.num_tile_rows(a), cols=kd.cdiv(kd.num_tiles(w, -1), nt))
    for i, j in g.units():
        acc = kd.accumulator((1, nt))
        for k in kd.range(kd.num_tiles(a, -1)):
            acc += kd.load(a.tiles[i, k]) @ kd.load(w.tiles[k, j * nt : (j + 1) * nt])
        kd.store(out.tiles[i, j * nt : (j + 1) * nt], kd.silu(acc).cast(out.dtype))


@kd.kernel
def linear_gelu(
    a: kd.Tensor, w: kd.Tensor, b: kd.Tensor, out: kd.Out, nt: kd.Static[int] = 2
):
    """out = gelu(a @ w + b), GELU in its tanh form; b is [N]."""
    g = kd.grid2d(rows=kd.num_tile_rows(a), cols=kd.cdiv(kd.num_tiles(w, -1), nt))
    for i, j in g.units():
        acc = kd.accumulator((1, nt))
        for k in kd.range(kd.num_tiles(a, -1)):
            acc += kd.load(a.tiles[i, k]) @ kd.load(w.tiles[k, j * nt : (j + 1) * nt])
        x = acc + kd.load(b.tiles[0, j * nt : (j + 1) * nt]).row()
        inner = GELU_C * (x + 0.044715 * x * x * x)
        kd.store(
            out.tiles[i, j * nt : (j + 1) * nt], (0.5 * x * (1.0 + kd.tanh(inner)))
        )


@kd.kernel
def norm_linear(
    x: kd.Tensor,
    scale: kd.Tensor,
    w: kd.Tensor,
    out: kd.Out,
    eps: kd.Runtime[float],
    nt: kd.Static[int] = 4,
):
    """out = rms_norm(x, scale, eps) @ w: each row normalized in L1 (exact fp32
    statistic), packed as bf16 and multiplied, one launch."""
    g = kd.grid2d(rows=kd.num_tile_rows(x), cols=kd.cdiv(kd.num_tiles(w, -1), nt))
    gamma = kd.resident(scale.tiles[0, :])
    for i, j in g.units():
        row = kd.load(x.tiles[i, :], pad=0).cast(kd.f32)
        ms = kd.reduce_sum(row * row, dim=-1, exact=True) / kd.size(x, -1)
        h = (row * kd.rsqrt(ms + eps).col() * gamma.row()).cast(kd.bf16)
        acc = kd.accumulator((1, nt))
        for k in kd.range(kd.num_tiles(x, -1)):
            acc += h.tile(k) @ kd.load(w.tiles[k, j * nt : (j + 1) * nt])
        kd.store(out.tiles[i, j * nt : (j + 1) * nt], acc.cast(out.dtype))


@torch.no_grad()
def compare_norm(tokens, inner, cols, eps=1e-6, seed=0):
    """norm_linear against the two tile kernels it fuses (bit for bit) and against
    F.rms_norm + matmul eagerly: agreement, errors against float64 and times (s)."""
    import rms_norm

    g = torch.Generator().manual_seed(seed)
    dev = kohakutt.device(0)
    x = torch.randn(1, tokens, inner, generator=g).bfloat16()
    gamma = (1 + 0.1 * torch.randn(inner, generator=g)).bfloat16()
    w = (torch.randn(inner, cols, generator=g) / inner**0.5).bfloat16()
    x64 = x.double()
    h64 = x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + eps) * gamma.double()
    want = h64 @ w.double()
    x_tt, g_tt, w_tt = x.to(dev), gamma.to(dev), w.to(dev)
    fused = norm_linear(x_tt, g_tt, w_tt, eps).cpu()
    two = linear(rms_norm.rms_norm(x_tt, g_tt, eps), w_tt).cpu()
    eager = F.rms_norm(x_tt, (inner,), g_tt, eps) @ w_tt
    return {
        "fused = rms_norm then linear": torch.equal(fused, two),
        "fused error": errors(fused, want),
        "eager error": errors(eager.cpu(), want),
        "fused ms": timed(lambda: norm_linear(x_tt, g_tt, w_tt, eps)) * 1e3,
        "two tile kernels ms": timed(
            lambda: linear(rms_norm.rms_norm(x_tt, g_tt, eps), w_tt)
        )
        * 1e3,
        "eager F.rms_norm + @ ms": timed(
            lambda: F.rms_norm(x_tt, (inner,), g_tt, eps) @ w_tt
        )
        * 1e3,
    }


def matmul_km():
    """The metal-level matmul example's op (a @ w, w [K, N] for w_nk False)."""
    os.environ.setdefault("KTT_OWN_MATMUL_MIN_ROWS", "1")
    return metal_example("matmul_km").matmul.op


def errors(got, want):
    """(max, mean) absolute error of `got` against float64 `want`."""
    diff = (got.double() - want).abs()
    return float(diff.max()), float(diff.mean())


@torch.no_grad()
def compare(tokens, inner, cols, seed=0):
    """Agreement, errors against float64 and times (s) at [1, T, K] @ [K, N]."""
    g = torch.Generator().manual_seed(seed)
    dev = kohakutt.device(0)
    x = torch.randn(1, tokens, inner, generator=g).bfloat16()
    layer = torch.nn.Linear(inner, cols).bfloat16()
    with torch.no_grad():
        layer.weight.copy_(torch.randn(cols, inner, generator=g) / inner**0.5)
        layer.bias.copy_(torch.randn(cols, generator=g) * 0.1)
    w = layer.weight.t().contiguous()
    want_silu = F.silu(x.double() @ w.double())
    want_gelu = F.gelu(
        x.double() @ w.double() + layer.bias.double(), approximate="tanh"
    )
    x_tt, w_tt, b_tt = x.to(dev), w.to(dev), layer.bias.detach().to(dev)
    layer_tt = layer.to(dev)
    twin = matmul_km()
    out = {
        # The metal-level example takes K and N up to 1024.
        "linear = metal matmul": (
            torch.equal(linear(x_tt, w_tt).cpu(), twin(x_tt, w_tt, False).cpu())
            if max(inner, cols) <= 1024
            else None
        ),
        "silu tile error": errors(linear_silu(x_tt, w_tt).cpu(), want_silu),
        "silu eager error": errors(F.silu(x_tt @ w_tt).cpu(), want_silu),
        "gelu tile error": errors(linear_gelu(x_tt, w_tt, b_tt).cpu(), want_gelu),
        "gelu eager error": errors(
            F.gelu(layer_tt(x_tt), approximate="tanh").cpu(), want_gelu
        ),
        "linear ms": timed(lambda: linear(x_tt, w_tt)) * 1e3,
        "metal matmul ms": (
            timed(lambda: twin(x_tt, w_tt, False)) * 1e3
            if max(inner, cols) <= 1024
            else None
        ),
        "backend a @ w ms": timed(lambda: x_tt @ w_tt) * 1e3,
        "linear_silu ms": timed(lambda: linear_silu(x_tt, w_tt)) * 1e3,
        "eager silu(a @ w) ms": timed(lambda: F.silu(x_tt @ w_tt)) * 1e3,
        "linear_gelu ms": timed(lambda: linear_gelu(x_tt, w_tt, b_tt)) * 1e3,
        "eager gelu(linear) ms": timed(
            lambda: F.gelu(layer_tt(x_tt), approximate="tanh")
        )
        * 1e3,
    }
    return out


def main():
    """Runs the kernels at one shape; prints agreement, errors, times and the plan."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--inner", type=int, default=768)
    parser.add_argument("--cols", type=int, default=3072)
    args = parser.parse_args()
    try:
        print(f"linear [1, {args.tokens}, {args.inner}] @ [{args.inner}, {args.cols}]")
        for key, value in compare(args.tokens, args.inner, args.cols).items():
            print(f"  {key}: {value}", flush=True)
        print("norm + linear")
        for key, value in compare_norm(args.tokens, args.inner, args.cols).items():
            print(f"  {key}: {value}", flush=True)
        a = kd.like((1, args.tokens, args.inner))
        w = kd.like((args.inner, args.cols))
        print(linear_gelu.plan(a, w, kd.like((args.cols,))).report())
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
