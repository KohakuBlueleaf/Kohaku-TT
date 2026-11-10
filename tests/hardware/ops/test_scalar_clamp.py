"""maximum / minimum / clamp_min / clamp_max against a scalar on tt equal CPU."""

import pytest
import torch

SCALARS = [0.25, -1.0, 3.0]
DTYPES = [torch.float32, torch.bfloat16, torch.int32]


def _input(dtype):
    g = torch.Generator().manual_seed(0)
    if dtype == torch.int32:
        return torch.randint(-20, 20, (64, 96), generator=g, dtype=dtype)
    return torch.randn(64, 96, generator=g).to(dtype)


def _same(got, want):
    assert got.device.type == "tt" and got.dtype == want.dtype
    assert torch.equal(got.cpu(), want)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).replace("torch.", ""))
@pytest.mark.parametrize("scalar", SCALARS)
def test_clamp_min_max(native, dtype, scalar):
    s = int(scalar) if dtype == torch.int32 else scalar
    x = _input(dtype)
    t = x.to("tt")
    _same(torch.clamp_min(t, s), torch.clamp_min(x, s))
    _same(torch.clamp_max(t, s), torch.clamp_max(x, s))
    _same(torch.clamp(t, min=s), torch.clamp(x, min=s))
    _same(torch.clamp(t, max=s), torch.clamp(x, max=s))


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).replace("torch.", ""))
@pytest.mark.parametrize("scalar", SCALARS)
def test_maximum_minimum_with_scalar_tensor(native, dtype, scalar):
    s = torch.tensor(int(scalar) if dtype == torch.int32 else scalar, dtype=dtype)
    x = _input(dtype)
    t = x.to("tt")
    _same(torch.maximum(t, s.to("tt")), torch.maximum(x, s))
    _same(torch.minimum(t, s.to("tt")), torch.minimum(x, s))
