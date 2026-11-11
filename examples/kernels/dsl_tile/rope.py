"""Rotary embedding at the tile level in the two forms models write: rotate-half
(DINOv3, Qwen) and complex pairs (PixAI tagger), against the same torch code run
eagerly on the tt device; the complex form also against the metal level.
Usage: python examples/kernels/dsl_tile/rope.py [--heads H] [--tokens N] [--dim D]"""

import argparse
import sys
from pathlib import Path

import torch

import kohakutt
import kohakutt.dsl as kd

sys.path.insert(0, str(Path(__file__).parent))
from common import metal_example, timed


@kd.kernel
def rope_half(x: kd.Tensor, cos: kd.Tensor, sin: kd.Tensor, out: kd.Out):
    """out = x * cos + rotate_half(x) * sin for x [..., N, D] and cos, sin [N, D]."""
    for t in kd.grid(kd.num_tile_rows(x)):
        row = kd.load(x.tiles[t, :]).cast(kd.f32)
        p = t % kd.num_tile_rows(cos)
        c = kd.load(cos.tiles[p, :])
        s = kd.load(sin.tiles[p, :])
        half = kd.size(x, -1) // 2
        rotated = kd.cat([-row[..., half:], row[..., :half]])
        kd.store(out.tiles[t, :], (row * c + rotated * s).cast(out.dtype))


@kd.kernel
def rope_complex(
    x: kd.Tensor, table: kd.Tensor, out: kd.Out, block: kd.Static[int] = 4
):
    """x [..., N, D] as (re, im) pairs times table [N, D] (fp32 cos, sin pairs)."""
    for t in kd.grid(kd.num_tiles(out), block=block):
        a = kd.load(x.tiles[t]).cast(kd.f32)
        b = kd.load(table.tiles[t % kd.num_tiles(table)])
        re = a.even * b.even - a.odd * b.odd
        im = a.even * b.odd + a.odd * b.even
        kd.store(out.tiles[t], kd.pairs(re, im).cast(out.dtype))


def rotate_half(x):
    """The rotation of transformers' DINOv3 and Qwen2 code."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def eager_half(q, cos, sin):
    """`(q * cos) + (rotate_half(q) * sin)`, as the models write it."""
    return (q * cos) + (rotate_half(q) * sin)


def eager_complex(xq, freqs_cis):
    """The PixAI tagger's apply_rotary_enc for one tensor."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    shape = [d if i >= xq_.ndim - 2 else 1 for i, d in enumerate(xq_.shape)]
    xq_out = torch.view_as_real(xq_ * freqs_cis.view(*shape)).flatten(3)
    return xq_out.type_as(xq)


def tables(tokens, dim, seed=0):
    """(cos, sin) [N, D] in the rotate-half layout and the complex table freqs_cis
    [N, D / 2] of the same angles."""
    g = torch.Generator().manual_seed(seed)
    angles = torch.rand(tokens, dim // 2, generator=g) * 6.2831853
    cos = torch.cat([angles.cos()] * 2, -1)
    sin = torch.cat([angles.sin()] * 2, -1)
    return cos, sin, torch.polar(torch.ones_like(angles), angles)


def errors(got, want):
    diff = (got.double() - want.double()).abs()
    return float(diff.max()), float(diff.mean())


def compare_half(heads, tokens, dim, seed=0):
    """Times and errors (against float64) of rotate-half, tile kernel and eager."""
    dev = kohakutt.device(0)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, heads, tokens, dim, generator=g).bfloat16()
    cos, sin, _ = tables(tokens, dim, seed)
    cos, sin = cos.bfloat16(), sin.bfloat16()
    want = eager_half(x.double(), cos.double(), sin.double())
    x_tt, c_tt, s_tt = x.to(dev), cos.to(dev), sin.to(dev)
    tile = rope_half(x_tt, c_tt, s_tt).cpu()
    eager = eager_half(x_tt, c_tt, s_tt).cpu()
    return {
        "tile ms": timed(lambda: rope_half(x_tt, c_tt, s_tt)) * 1e3,
        "eager ms": timed(lambda: eager_half(x_tt, c_tt, s_tt)) * 1e3,
        "tile error": errors(tile, want),
        "eager error": errors(eager, want),
        "cpu error": errors(eager_half(x, cos, sin), want),
    }


def compare_complex(heads, tokens, dim, seed=0):
    """Times and errors of the complex form; agreement with the metal level."""
    dev = kohakutt.device(0)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, heads, tokens, dim, generator=g).bfloat16()
    _, _, freqs = tables(tokens, dim, seed)
    table = torch.view_as_real(freqs).reshape(tokens, dim).contiguous()
    want = eager_complex(x.double(), freqs.to(torch.complex128)).double()
    x_tt, t_tt, f_tt = x.to(dev), table.to(dev), freqs.to(dev)
    rope_km = metal_example("rope_km").rope.op
    tile = rope_complex(x_tt, t_tt).cpu()
    eager = eager_complex(x_tt, f_tt).cpu()
    return {
        "tile = metal": torch.equal(tile, rope_km(x_tt, t_tt).cpu()),
        "tile ms": timed(lambda: rope_complex(x_tt, t_tt)) * 1e3,
        "metal ms": timed(lambda: rope_km(x_tt, t_tt)) * 1e3,
        "eager ms": timed(lambda: eager_complex(x_tt, f_tt)) * 1e3,
        "tile error": errors(tile, want),
        "eager error": errors(eager, want),
        "cpu error": errors(eager_complex(x, freqs), want),
    }


def main():
    """Runs both forms at one shape; prints agreement, times and errors."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=64)
    args = parser.parse_args()
    shape = (args.heads, args.tokens, args.dim)
    try:
        for name, fn in (("rotate-half", compare_half), ("complex", compare_complex)):
            print(f"rope {name} [1, {', '.join(map(str, shape))}]")
            for key, value in fn(*shape).items():
                print(f"  {key}: {value}")
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
