"""Regression: argmax / argmin of bf16 / fp32 tensors are one own program (long rows
split over cores, combined on the device), equal to PyTorch: first index on ties, NaN
first."""

import pytest
import torch

import kohakutt


def sample(shape, seed, dtype):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g).to(dtype)


def counted(fn):
    """(ops, own programs) queued by fn() and its result."""
    kohakutt.synchronize()
    before = kohakutt.counters()
    out = fn()
    kohakutt.synchronize()
    after = kohakutt.counters()
    return (after["ops"] - before["ops"], after["programs"] - before["programs"]), out


@pytest.mark.hw
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape",
    [(1, 1, 151936), (3, 5000), (2, 7, 300), (70, 40, 100), (1, 33)],
    ids=lambda s: "x".join(map(str, s)),
)
@pytest.mark.parametrize("op", ["argmax", "argmin"])
def test_last_dim_one_program(native, dtype, shape, op):
    x = sample(shape, 0, dtype)
    xt = x.to("tt")
    counts, out = counted(lambda: getattr(xt, op)(-1))
    assert counts == (1, 1)
    assert torch.equal(out.cpu(), getattr(x, op)(-1))


@pytest.mark.hw
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_ties_and_specials(native, dtype):
    w = 151936
    x = torch.zeros(4, w, dtype=dtype)
    x[0, 100000] = 5
    x[0, 7] = 5  # first of two maxima, in another chunk
    x[1, 90000] = float("inf")
    x[1, 120000] = float("nan")  # NaN wins over an earlier inf
    x[1, 140000] = float("nan")
    x[2] = -float("inf")
    x[2, w - 1] = -1e30
    x[3, 3] = -0.0  # -0.0 ties +0.0: index 0 stays first
    xt = x.to("tt")
    for op in ("argmax", "argmin"):
        assert torch.equal(getattr(xt, op)(-1).cpu(), getattr(x, op)(-1)), op


@pytest.mark.hw
def test_other_dims_and_keepdim(native):
    x = sample((5, 300, 3), 1, torch.float32)
    xt = x.to("tt")
    assert torch.equal(xt.argmax(1).cpu(), x.argmax(1))
    assert torch.equal(xt.argmin(1, keepdim=True).cpu(), x.argmin(1, keepdim=True))
    assert torch.equal(xt.argmax().cpu(), x.argmax())
    assert torch.equal(xt.argmax(keepdim=True).cpu(), x.argmax(keepdim=True))
    values, index = xt.max(-1)
    assert torch.equal(index.cpu(), x.max(-1).indices)
    assert torch.equal(values.cpu(), x.max(-1).values)
    assert torch.equal(xt.min(0).indices.cpu(), x.min(0).indices)


@pytest.mark.hw
def test_int_input_keeps_cast_route(native):
    x = torch.randint(-1000, 1000, (3, 400))
    assert torch.equal(x.to("tt").argmax(-1).cpu(), x.argmax(-1))
