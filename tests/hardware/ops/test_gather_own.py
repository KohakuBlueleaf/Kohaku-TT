"""Kohaku-TT's gather along the last dim: bit-exact against PyTorch for every dtype,
ragged / 1-D / batched shapes, take_along_dim, flat sizes up to 1M under a time
ceiling, no compile for a new shape, and the error for large uncovered calls."""

import time

import pytest
import torch

import kohakutt

DTYPES = [torch.float32, torch.bfloat16, torch.int32]


def _data(shape, dtype):
    if dtype == torch.int32:
        return torch.randint(-(2**31), 2**31 - 1, shape, dtype=torch.int32)
    return torch.randn(shape).to(dtype)


def _bits(t):
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32)


@pytest.mark.hw
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("xs", "ishape"),
    [
        ((5,), (9,)),
        ((1000,), (77,)),
        ((3, 45, 70), (3, 40, 90)),
        ((2, 3, 33, 65), (2, 3, 17, 10)),
        ((64, 32), (64, 32)),
        ((7, 1), (7, 5)),
    ],
)
def test_gather_last_dim(native, dtype, xs, ishape):
    x = _data(xs, dtype)
    idx = torch.randint(0, xs[-1], ishape)
    idx.view(-1)[0] = xs[-1] - 1
    out = torch.gather(x.to("tt"), -1, idx.to("tt")).cpu()
    assert torch.equal(_bits(out), _bits(torch.gather(x, -1, idx)))


@pytest.mark.hw
@pytest.mark.parametrize("dtype", DTYPES)
def test_take_along_last_dim(native, dtype):
    x = _data((4, 64, 1000), dtype)
    idx = torch.randint(0, 1000, (4, 64, 1))
    out = torch.take_along_dim(x.to("tt"), idx.to("tt"), -1).cpu()
    assert torch.equal(_bits(out), _bits(torch.take_along_dim(x, idx, -1)))


@pytest.mark.hw
def test_new_shape_compiles_nothing(native):
    for n in (300, 4100):
        x = torch.randn(n)
        idx = torch.randint(0, n, (n,))
        before = kohakutt.counters()["kernel_compiles"]
        out = torch.gather(x.to("tt"), 0, idx.to("tt")).cpu()
        assert torch.equal(out, x[idx])
    assert kohakutt.counters()["kernel_compiles"] == before


@pytest.mark.hw
@pytest.mark.parametrize("n", [131072, 262144, 270336, 524288, 1048576])
@pytest.mark.parametrize("order", ["reversed", "random"])
def test_flat_sizes_time_ceiling(native, n, order):
    x = torch.arange(n, dtype=torch.float32)
    idx = (
        torch.arange(n - 1, -1, -1)
        if order == "reversed"
        else torch.randint(0, n, (n,))
    )
    xt, it = x.to("tt"), idx.to("tt")
    torch.gather(xt, 0, it)
    kohakutt.synchronize()
    took = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        out = torch.gather(xt, 0, it)
        kohakutt.synchronize()
        took = min(took, time.perf_counter() - start)
    print(f"gather {order} {n}: {took * 1e3:.2f} ms")
    assert torch.equal(out.cpu(), x[idx])
    assert took < 1.0


@pytest.mark.hw
def test_uncovered_small_runs_and_large_raises(native):
    x = torch.randn(40, 30)
    idx = torch.randint(0, 40, (20, 30))
    assert torch.equal(
        torch.gather(x.to("tt"), 0, idx.to("tt")).cpu(), torch.gather(x, 0, idx)
    )
    big = torch.randn(600, 600).to("tt")
    with pytest.raises(Exception, match="KTT_TTNN_GATHER_LARGE"):
        torch.gather(big, 0, torch.zeros(4, 600, dtype=torch.long).to("tt")).cpu()
