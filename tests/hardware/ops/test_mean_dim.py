"""Regression: mean over the last two dims is one reduction that scales its sum (no
separate division); other dims keep sum then divide. Values match PyTorch's."""

import pytest
import torch

import kohakutt

TOL = {torch.bfloat16: 1e-2, torch.float32: 1e-5}


def counted(fn):
    """(result, ops queued by fn())."""
    kohakutt.synchronize()
    before = kohakutt.counters()
    out = fn()
    kohakutt.synchronize()
    return out, kohakutt.counters()["ops"] - before["ops"]


@pytest.mark.hw
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize(
    "shape, dims, ops",
    [
        ((1, 1, 896), (-1,), 1),
        ((1, 7, 896), (2,), 1),
        ((2, 7, 64), (-1, -2), 1),
        ((2, 7, 64), (-2,), 1),
        ((5, 1000), (1,), 1),
        ((2, 3, 40, 100), (1,), 2),
        ((2, 3, 40, 100), (0, 2), 2),
    ],
)
def test_mean_dim(native, dtype, keepdim, shape, dims, ops):
    g = torch.Generator().manual_seed(len(shape) + dims[0])
    x = (torch.randn(shape, generator=g) + 0.5).to(dtype)
    t = x.to("tt")
    t.mean(dims, keepdim)
    got, n = counted(lambda: t.mean(dims, keepdim))
    want = x.double().mean(dims, keepdim).to(dtype)
    assert got.dtype == dtype
    torch.testing.assert_close(got.cpu(), want, rtol=TOL[dtype], atol=TOL[dtype])
    assert n == ops
