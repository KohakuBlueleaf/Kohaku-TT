"""Regression: where.self with one-element operands (the decode mask's scalar branches)
reads them unexpanded in one TT-NN where, equal to PyTorch; other broadcasts expand."""

import pytest
import torch

import kohakutt


def counted(fn):
    """(ops, own programs) queued by fn() and its result."""
    kohakutt.synchronize()
    before = kohakutt.counters()
    out = fn()
    kohakutt.synchronize()
    after = kohakutt.counters()
    return (after["ops"] - before["ops"], after["programs"] - before["programs"]), out


def mask(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=g) > 0.5


@pytest.mark.hw
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape", [(1, 1, 1, 125), (2, 4, 40, 50), (3, 64, 1), (77,)], ids=str
)
def test_scalar_branches(native, dtype, shape):
    cond = mask(shape, 0)
    low = torch.tensor(torch.finfo(dtype).min, dtype=dtype)
    zero = torch.tensor(0.0, dtype=dtype)
    want = torch.where(cond, zero, low)
    c, z, lo = cond.to("tt"), zero.to("tt"), low.to("tt")
    counts, out = counted(lambda: torch.where(c, z, lo))
    # fp32: the bool condition is cast first (one own program), as before.
    assert counts == ((1, 0) if dtype == torch.bfloat16 else (2, 1))
    assert torch.equal(out.cpu(), want)


@pytest.mark.hw
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_one_scalar_branch_and_scalar_cond(native, dtype):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 33, 70, generator=g).to(dtype)
    cond = mask((2, 33, 70), 2)
    s = torch.tensor(-3.5, dtype=dtype)
    xt, ct, st = x.to("tt"), cond.to("tt"), s.to("tt")
    assert torch.equal(torch.where(ct, xt, st).cpu(), torch.where(cond, x, s))
    assert torch.equal(torch.where(ct, st, xt).cpu(), torch.where(cond, s, x))
    one = torch.tensor(True)
    assert torch.equal(
        torch.where(one.to("tt"), xt, -xt).cpu(), torch.where(one, x, -x)
    )


@pytest.mark.hw
def test_row_broadcast_keeps_expansion(native):
    g = torch.Generator().manual_seed(3)
    x = torch.randn(2, 4, 40, 50, generator=g)
    cond = mask((1, 1, 40, 1), 4)
    out = torch.where(cond.to("tt"), x.to("tt"), torch.tensor(0.0).to("tt")).cpu()
    assert torch.equal(out, torch.where(cond, x, torch.tensor(0.0)))
