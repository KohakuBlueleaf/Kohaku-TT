"""Regression: ktt_fusion.rope (RoPE as one program, rotate-half and interleaved pairs)
over decode and prefill shapes, every even head dim up to 256 (not whole tiles too),
cos / sin broadcast over heads (bf16 or fp32): no worse than CPU's bf16 chain."""

import pytest
import torch

import kohakutt
import kohakutt.fusion.rope  # registers the op

# tt's relative RMS may exceed CPU's by this factor (as tests/hardware/ops/test_norm:
# where CPU rounds once too, the fp32 operation order differs).
SLACK = 1.02


def _rel(a, ref):
    a, ref = a.double().flatten(), ref.double().flatten()
    return ((a - ref).norm() / ref.norm().clamp_min(1e-300)).item()


def rotate(x, form):
    if form == 0:
        h = x.shape[-1] // 2
        return torch.cat((-x[..., h:], x[..., :h]), dim=-1)
    return torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)


@pytest.mark.hw
@pytest.mark.parametrize("form", [0, 1])
@pytest.mark.parametrize(
    "x_shape, cos_shape, cos_dtype",
    [
        ((1, 14, 1, 64), (1, 1, 1, 64), torch.bfloat16),
        ((1, 16, 1, 128), (1, 1, 1, 128), torch.bfloat16),
        ((2, 4, 77, 64), (2, 1, 77, 64), torch.bfloat16),
        ((1, 4, 40, 32), (1, 1, 40, 32), torch.bfloat16),
        ((1, 4, 40, 96), (1, 1, 40, 96), torch.bfloat16),
        ((1, 3, 33, 256), (33, 256), torch.float32),
        ((1, 4, 9, 80), (1, 1, 9, 80), torch.bfloat16),
        ((2, 3, 5, 72), (2, 3, 5, 72), torch.bfloat16),
        ((1, 2, 3, 2), (1, 1, 3, 2), torch.float32),
    ],
)
def test_rope(native, form, x_shape, cos_shape, cos_dtype):
    g = torch.Generator().manual_seed(sum(x_shape) + form)
    x = torch.randn(x_shape, generator=g).bfloat16()
    angle = torch.rand(cos_shape, generator=g) * 6
    cos, sin = angle.cos().to(cos_dtype), angle.sin().to(cos_dtype)
    ref = x.double() * cos.double() + rotate(x.double(), form) * sin.double()
    cpu = (x * cos + rotate(x, form) * sin).bfloat16()
    args = (x.to("tt"), cos.to("tt"), sin.to("tt"), form)
    torch.ops.ktt_fusion.rope(*args)
    kohakutt.synchronize()
    before = kohakutt.counters()["programs"]
    got = torch.ops.ktt_fusion.rope(*args)
    kohakutt.synchronize()
    assert kohakutt.counters()["programs"] - before == 1
    assert got.dtype == torch.bfloat16 and got.shape == x.shape
    assert _rel(got.cpu(), ref) <= SLACK * _rel(cpu, ref)
