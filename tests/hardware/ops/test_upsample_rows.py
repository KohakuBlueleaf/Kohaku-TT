"""Nearest upsampling by integer factors (row copies, one program) is bit-exact against
PyTorch for bf16 and fp32, Nchw and Nhwc storage (kept as stored), ragged sizes, and
factors dividing 32 or not (with activations in L1: test_upsample_l1.py)."""

import pytest
import torch
import torch.nn.functional as F

import kohakutt

CASES = [
    ((2, 64, 32, 32), (2, 2)),
    ((1, 3, 17, 23), (2, 2)),
    ((1, 4, 7, 16), (2, 2)),
    ((2, 32, 16, 16), (4, 4)),
    ((1, 4, 33, 7), (1, 2)),
    ((1, 4, 7, 33), (2, 1)),
    ((2, 40, 6, 8), (2, 4)),
    ((1, 5, 9, 40), (3, 3)),
    ((1, 40, 5, 48), (3, 2)),
    ((1, 8, 10, 12), (5, 3)),
]


@pytest.mark.hw
@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("shape,factor", CASES, ids=lambda v: "x".join(map(str, v)))
def test_upsample_nearest_factors(native, shape, factor, dtype, layout):
    g = torch.Generator().manual_seed(sum(shape) + factor[0])
    x = torch.randn(shape, generator=g).to(dtype)
    want = F.interpolate(x, scale_factor=factor, mode="nearest")
    t = x.to("tt")
    if layout == "nhwc":
        torch.ops.ktt._debug_to_nhwc(t)
    torch.tt.synchronize()
    before = kohakutt.counters()["programs"]
    got = F.interpolate(t, scale_factor=factor, mode="nearest")
    torch.tt.synchronize()
    assert kohakutt.counters()["programs"] - before == 1
    assert torch.ops.ktt._debug_layout(got) == layout
    assert torch.equal(got.cpu(), want)
