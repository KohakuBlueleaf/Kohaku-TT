"""Card tests of the tile-level examples: each kernel equals its metal-level version
(or the same arithmetic on the host) bit for bit, and a sweep of sizes inside one size
class compiles no kernel and builds no program after its first call.
Run: python -m pytest examples/kernels/dsl_tile/test_dsl_tile.py"""

import sys
from pathlib import Path

import pytest
import torch

import kohakutt

sys.path.insert(0, str(Path(__file__).parent))
import rms_norm
import rope
import swiglu
from common import builds, counted


@pytest.fixture(scope="module")
def dev():
    yield kohakutt.device(0)
    kohakutt.close()


def _randn(shape, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=g) * scale).bfloat16()


def _sweep(fn, shapes):
    """(compiles, builds) of each call of fn(shape) after the first."""
    fn(shapes[0])
    out = []
    for shape in shapes[1:]:
        _, grown = counted(lambda shape=shape: fn(shape))
        out.append(builds(grown))
    return out


@pytest.mark.parametrize("shape", [(1, 64, 4096), (3, 100, 160)])
def test_silu_and_swiglu_equal_the_metal_level(dev, shape):
    a, b = _randn(shape, 1, 3).to(dev), _randn(shape, 2).to(dev)
    assert torch.equal(swiglu.silu_fp32(a).cpu(), swiglu.silu_km.op(a).cpu())
    twin = swiglu.swiglu_km()
    want = twin(a, b).cpu()
    assert torch.equal(swiglu.swiglu_fp32(a, b).cpu(), want)
    # bf16 registers: one rounding to nearest even instead of the packer's.
    got = swiglu.swiglu(a, b).cpu().float()
    ulp = torch.finfo(torch.bfloat16).eps * want.float().abs()
    assert ((got - want.float()).abs() <= ulp).all()


def test_swiglu_sweep_builds_nothing(dev):
    def call(shape):
        a = torch.ones(shape, dtype=torch.bfloat16).to(dev)
        return swiglu.swiglu(a, a).cpu()

    # 4 * T tiles stay above the grid's 110 cores x 8: one core count.
    shapes = [(1, t, 4096) for t in (512, 640, 1024, 2048, 4096)]
    assert _sweep(call, shapes) == [(0, 0)] * 4


@pytest.mark.parametrize("shape", [(1, 300, 1536), (2, 64, 1000), (1, 96, 8, 256)])
def test_rms_norm_equals_the_metal_level(dev, shape):
    assert rms_norm.same_as_metal(shape)


def test_rms_norm_sweep_builds_nothing(dev):
    scale = torch.ones(1536, dtype=torch.bfloat16).to(dev)

    def call(shape):
        x = _randn(shape, 3).to(dev)
        return rms_norm.rms_norm(x, scale, 1e-6).cpu()

    # 35 .. 48 tile rows: the 48-core class; 48 tiles per row: one row class.
    shapes = [(1, t, 1536) for t in (1100, 1200, 1300, 1400, 1536)]
    assert _sweep(call, shapes) == [(0, 0)] * 4


def test_rope_complex_equals_the_metal_level(dev):
    out = rope.compare_complex(4, 197, 64)
    assert out["tile = metal"]


@pytest.mark.parametrize("dim", [64, 128])
def test_rope_half_equals_one_rounding_of_the_fp32_formula(dev, dim):
    x = _randn((1, 3, 197, dim), 4)
    cos, sin, _ = rope.tables(197, dim)
    cos, sin = cos.bfloat16(), sin.bfloat16()
    args = (x.to(dev), cos.to(dev), sin.to(dev))
    want = rope.eager_half(x.float(), cos.float(), sin.float())
    got32 = rope.rope_half(*args, out_dtype=torch.float32).cpu()
    assert torch.equal(got32, want)
    got = rope.rope_half(*args).cpu()
    assert torch.equal(got, packed_bf16(want))


def packed_bf16(x):
    """fp32 `x` rounded to bf16 as the packer does: to nearest, ties away from 0."""
    bits = x.view(torch.int32) + 0x8000
    return (bits & ~0xFFFF).view(torch.float32).bfloat16()


def test_rope_half_sweep_builds_nothing(dev):
    sizes = [1024, 1500, 2048, 4096, 5184]
    inputs = {}
    for tokens in sizes:
        cos, sin, _ = rope.tables(tokens, 64)
        x = torch.ones(1, 16, tokens, 64, dtype=torch.bfloat16)
        inputs[tokens] = [t.bfloat16().to(dev) for t in (x, cos, sin)]

    def call(tokens):
        return rope.rope_half(*inputs[tokens]).cpu()

    assert _sweep(call, sizes) == [(0, 0)] * 4
