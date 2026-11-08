"""Regression: row selections on Kohaku-TT's row gather (index_select along the last two
dims, flip, strided max-pool taps, small bf16 embedding lookups) equal PyTorch's."""

import pytest
import torch
import torch.nn.functional as F

DTYPES = [torch.float32, torch.bfloat16]


@pytest.mark.hw
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", [-2, -1])
def test_index_select_last_dims(native, dtype, dim):
    x = torch.randn(3, 45, 70).to(dtype)
    idx = torch.randint(0, x.size(dim), (37,))
    idx[0] = x.size(dim) - 1
    out = x.to("tt").index_select(dim, idx.to("tt")).cpu()
    assert torch.equal(out, x.index_select(dim, idx))


@pytest.mark.hw
@pytest.mark.parametrize("dtype", DTYPES)
def test_flip_last_dim(native, dtype):
    x = torch.randn(2, 33, 47).to(dtype)
    assert torch.equal(x.to("tt").flip(-1).cpu(), x.flip(-1))


@pytest.mark.hw
@pytest.mark.parametrize("dtype", DTYPES)
def test_strided_max_pool_indices(native, dtype):
    x = torch.randn(2, 8, 37, 29).to(dtype)
    value, index = F.max_pool2d(x.to("tt"), 3, 2, 1, return_indices=True)
    want_value, want_index = F.max_pool2d(x, 3, 2, 1, return_indices=True)
    assert torch.equal(value.cpu(), want_value)
    assert torch.equal(index.cpu(), want_index)


@pytest.mark.hw
def test_small_bf16_embedding(native):
    table = torch.randn(500, 77).bfloat16()
    idx = torch.randint(0, 500, (3, 7))
    out = F.embedding(idx.to("tt"), table.to("tt")).cpu()
    assert torch.equal(out, F.embedding(idx, table))
