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
import flash_attention
import group_norm
import in_use
import layer_norm
import linear
import online_softmax
import rms_norm
import rms_norm_backward
import rope
import shared_linear
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


@pytest.mark.parametrize("shape", [(1, 512, 768, 1024), (3, 100, 256, 288)])
def test_linear_equals_the_metal_level_matmul(dev, shape):
    *lead, k, n = shape
    a = _randn((*lead, k), 6).to(dev)
    w = _randn((k, n), 7, 1 / k**0.5).to(dev)
    got = linear.linear(a, w).cpu()
    assert torch.equal(got, linear.matmul_km()(a, w, False).cpu())


def test_linear_with_activation_is_no_less_accurate_than_eager(dev):
    out = linear.compare(256, 768, 1024)
    assert out["linear = metal matmul"]
    for act in ("silu", "gelu"):
        tile, eager = out[f"{act} tile error"], out[f"{act} eager error"]
        # One rounding of the fp32 sum instead of eager's two: no larger on average.
        assert tile[1] <= eager[1] and tile[0] <= 1.25 * eager[0], act


def test_norm_linear_equals_its_two_kernels(dev):
    out = linear.compare_norm(256, 1536, 512)
    assert out["fused = rms_norm then linear"]
    assert out["fused error"][1] <= 1.01 * out["eager error"][1]


def test_linear_sweep_builds_nothing(dev):
    w = _randn((768, 1024), 8, 1 / 768**0.5).to(dev)
    b = torch.zeros(1024, dtype=torch.bfloat16).to(dev)
    inputs = {t: torch.ones(1, t, 768, dtype=torch.bfloat16).to(dev) for t in SIZES}

    def call(tokens):
        return linear.linear_gelu(inputs[tokens], w, b).cpu()

    # 16 column groups of 6 cores: one core count from 6 row tiles on.
    assert _sweep(call, SIZES) == [(0, 0)] * 3


SIZES = [512, 700, 1024, 1500]


@pytest.mark.parametrize(
    "shape, group", [((1, 32, 4096), 8), ((2, 32, 2000), 4), ((1, 1, 8192), 16)]
)
def test_a_row_split_over_cores_equals_one_core_per_row(dev, shape, group):
    x, s = _randn(shape, 11).to(dev), (_randn(shape[-1:], 12).float() + 2).to(dev)
    split = rms_norm.rms_norm_split(x, s, 1e-6, group=group).cpu().float()
    whole = rms_norm.rms_norm(x, s, 1e-6, group=1).cpu().float()
    # The same sum of squares in another order: at most one bf16 ulp apart.
    ulp = torch.finfo(torch.bfloat16).eps * whole.abs().clamp(min=1e-30)
    assert ((split - whole).abs() <= ulp).all()
    plan = rms_norm.rms_norm_split.plan(x, s, 1e-6, group=group)
    assert plan.mode == "wide" and plan.cores == group * plan.units


def test_a_row_split_over_cores_sweep_builds_nothing(dev):
    s = torch.ones(4096).to(dev)
    # 9 .. 12 tile rows: the 12-unit class, 96 cores in groups of 8.
    shapes = [(1, t, 4096) for t in (288, 320, 352, 384)]
    inputs = {sh: _randn(sh, 13).to(dev) for sh in shapes}

    def call(shape):
        return rms_norm.rms_norm_split(inputs[shape], s, 1e-6).cpu()

    assert _sweep(call, shapes) == [(0, 0)] * 3


@pytest.mark.parametrize("shape", [(300, 1000), (2048, 1536), (64, 4096)])
def test_rms_norm_backward_matches_autograd(dev, shape):
    out = rms_norm_backward.compare(shape)
    # Sequential fp32 sums over the row (dx) and over the rows (dweight).
    assert out["tile dx error"] <= 1e-5
    assert out["tile dweight error"] <= 1e-5


def test_autograd_function_gradients_match_float64(dev):
    out = in_use.gradients()
    assert out["dx"] <= 1e-5 and out["dweight"] <= 1e-5


def test_tile_kernels_replay_in_a_capture_and_compile(dev):
    replay = in_use.captured()
    assert replay["replay vs eager"] == 0 and replay["new input"] == 0
    assert replay["built"] == (0, 0)
    fast = in_use.compiled()
    assert fast["compiled vs eager"] == 0 and fast["built"] == (0, 0)


def test_training_with_tile_norms_follows_torch_ops(dev):
    losses = in_use.train(steps=6)
    for a, b in zip(losses["tile"], losses["torch ops"], strict=True):
        assert abs(a - b) <= 1e-3 * abs(b)


@pytest.mark.parametrize(
    "shape, causal, kv_heads",
    [((1, 2, 256, 64), False, None), ((1, 2, 100, 64), False, None),
     ((1, 2, 300, 128), True, None), ((1, 4, 256, 64), False, 2)],  # fmt: skip
)
def test_flash_attention_is_as_accurate_as_the_backend(dev, shape, causal, kv_heads):
    out = flash_attention.compare(shape, causal, kv_heads)
    # Relative RMS against float64; the backend's flash kernel is about 5e-3.
    assert out["tile"] <= 1.25 * out["backend"] + 1e-4


@pytest.mark.parametrize(
    "shape, causal", [((1, 2, 256, 64), False), ((1, 2, 300, 128), True)]
)
def test_flash_attention_in_one_scope_per_step_is_as_accurate(dev, shape, causal):
    # Blocks of 3 key tiles: scores, row max and p in the registers.
    q, k, v = flash_attention.inputs(shape)
    want = flash_attention.reference(q, k, v, causal)
    fused = flash_attention.relative_rms(
        flash_attention.attention(q, k, v, causal, keys=3), want
    )
    blocks = flash_attention.relative_rms(
        flash_attention.attention(q, k, v, causal), want
    )
    assert fused <= 1.1 * blocks + 1e-4


def test_flash_attention_sweep_builds_nothing(dev):
    # 36 .. 42 query blocks of 2 tile rows: one core-count class.
    shapes = [(1, 2, s, 64) for s in (1100, 1200, 1300)]
    inputs = {s: flash_attention.inputs(s) for s in shapes}

    def call(shape):
        return flash_attention.attention(*inputs[shape]).cpu()

    assert _sweep(call, shapes) == [(0, 0)] * 2


def test_rms_norm_backward_sweep_builds_nothing(dev):
    w = torch.ones(1536).to(dev)
    shapes = [(t, 1536) for t in (1100, 1200, 1300, 1536)]
    inputs = {s: (_randn(s, 15).to(dev), _randn(s, 16).to(dev)) for s in shapes}

    def call(shape):
        x, dy = inputs[shape]
        dx, dw = rms_norm_backward.backward(x, dy, w)
        return dx.cpu(), dw.cpu()

    assert _sweep(call, shapes) == [(0, 0)] * 3


@pytest.mark.parametrize(
    "shape",
    [(2, 320, 64, 64), (2, 1280, 16, 16), (1, 512, 128, 128), (1, 64, 32, 96),
     (2, 96, 8, 8)],  # fmt: skip
)
def test_group_norm_is_as_accurate_as_eager(dev, shape):
    out = group_norm.compare(shape)
    tile, eager = out["tile error"], out["eager error"]
    assert tile[0] <= max(eager[0], 2**-7) and tile[1] <= 1.01 * eager[1]


def test_group_norm_sweep_builds_nothing(dev):
    # 40 .. 48 tiles per group: one run class.
    shapes = [(2, c, 64, 64) for c in (320, 352, 384)]
    inputs = {}
    for shape in shapes:
        x, _, _, (wt, bt) = group_norm.inputs(shape)
        inputs[shape] = (x, wt, bt)

    def call(shape):
        return group_norm.group_norm(*inputs[shape], 1e-5).cpu()

    assert _sweep(call, shapes) == [(0, 0)] * 2


@pytest.mark.parametrize("shape", [(1, 300, 1000), (2, 64, 768), (1, 128, 4096)])
def test_layer_norm_is_as_accurate_as_eager_in_bf16(dev, shape):
    out = layer_norm.compare(shape)
    tile, eager = out["tile bf16 error"], out["eager bf16 error"]
    assert tile[0] <= eager[0] and tile[1] <= 1.01 * eager[1]
    # fp32: sequential fp32 sums for the mean and the variance.
    assert out["tile fp32 error"][0] <= 3e-5


def test_layer_norm_sweep_builds_nothing(dev):
    w = torch.ones(1024, dtype=torch.bfloat16).to(dev)
    b = torch.zeros(1024, dtype=torch.bfloat16).to(dev)
    shapes = [(1, t, 1024) for t in (1100, 1200, 1300, 1536)]
    inputs = {s: _randn(s, 5).to(dev) for s in shapes}

    def call(shape):
        return layer_norm.layer_norm(inputs[shape], w, b, 1e-5).cpu()

    assert _sweep(call, shapes) == [(0, 0)] * 3


@pytest.mark.parametrize(
    "shape", [(512, 768, 1024), (300, 256, 288), (2048, 1536, 1536)]
)
def test_shared_linear_equals_the_metal_level_block_matmul(dev, shape):
    out = shared_linear.compare(*shape)
    assert out["linear = metal gemm"]
    tile, eager = out["gelu tile error"], out["gelu eager error"]
    assert tile[1] <= eager[1] and tile[0] <= 1.25 * eager[0]


@pytest.mark.parametrize("shape", [(512, 768, 1024), (2048, 768, 3072)])
def test_2d_linear_equals_the_backend_block_matmul(dev, shape):
    # N > K: the backend's own block matmul, in its 2-D plan (the same plan here).
    a, w, b = shared_linear.inputs(*shape)
    assert torch.equal(shared_linear.linear_2d(a, w).cpu(), (a @ w).cpu())
    x = a.cpu().double() @ w.cpu().double() + b.cpu().double()
    want = torch.nn.functional.gelu(x, approximate="tanh")
    eager = torch.nn.functional.gelu(
        torch.nn.functional.linear(a, w.t(), b), approximate="tanh"
    )
    tile = linear.errors(shared_linear.linear_gelu_2d(a, w, b).cpu(), want)
    assert tile[0] <= linear.errors(eager.cpu(), want)[0]


def test_2d_linear_sweep_builds_nothing(dev):
    w = _randn((768, 1024), 14, 1 / 768**0.5).to(dev)
    # 31 .. 40 row tiles: 4 per core row, the same blocks.
    sizes = [1000, 1100, 1200, 1280]
    inputs = {t: torch.ones(1, t, 768, dtype=torch.bfloat16).to(dev) for t in sizes}

    def call(tokens):
        return shared_linear.linear_2d(inputs[tokens], w).cpu()

    assert _sweep(call, sizes) == [(0, 0)] * 3


def test_shared_linear_sweep_builds_nothing(dev):
    w = _randn((768, 1024), 9, 1 / 768**0.5).to(dev)
    inputs = {t: torch.ones(1, t, 768, dtype=torch.bfloat16).to(dev) for t in SIZES}

    def call(tokens):
        return shared_linear.linear(inputs[tokens], w).cpu()

    assert _sweep(call, SIZES) == [(0, 0)] * 3


@pytest.mark.parametrize("shape", [(64, 96, 1), (256, 1024, 8)])
def test_online_softmax_step_is_one_rounding_of_the_float64_step(dev, shape):
    out = online_softmax.compare(*shape)
    assert out["tile m error"] == 0
    # p: the fp32 exp rounded once to bf16; den: the fp32 sum of the fp32 exps.
    assert out["tile p error (half ulps)"] <= 2.5
    assert out["tile den error"] <= max(1e-5, out["eager den error"])
    # p / den with p rounded to bf16: within half a bf16 ulp and den's error.
    assert out["softmax from the empty state error (relative)"] <= 2**-8 + 1e-5


def test_online_softmax_sweep_builds_nothing(dev):
    shapes = [(256, 1024), (320, 800), (512, 992), (288, 1024)]
    inputs = {s: online_softmax.inputs(*s, dev, heads=16) for s in shapes}

    def call(shape):
        return online_softmax.softmax_step(*inputs[shape], 0.125)[2].cpu()

    # 128 .. 256 tile rows: the whole grid; 25 .. 32 tiles per row: one row class.
    assert _sweep(call, shapes) == [(0, 0)] * 3
