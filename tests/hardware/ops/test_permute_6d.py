"""6-D permutes, size-2 dims moved last included, of whole and reshaped bases run on the
device: equal to PyTorch's, and no index is uploaded for them."""

import pytest
import torch

import kohakutt

CASES = [
    ((2, 8, 2, 8, 2, 32), None, (0, 1, 3, 5, 2, 4)),
    ((2, 2, 3, 4, 5, 6), None, (2, 3, 4, 5, 0, 1)),
    ((2, 16, 16, 64), (2, 8, 2, 8, 2, 64), (0, 1, 3, 5, 2, 4)),
    ((2, 64, 16, 16), (2, 64, 8, 2, 8, 2), (0, 1, 3, 5, 2, 4)),
    ((2, 16, 16, 64), (2, 8, 2, 8, 2, 64), (0, 1, 3, 4, 2, 5)),
]


@pytest.mark.hw
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("base, view, perm", CASES)
def test_permute_6d_on_device(native, dtype, base, view, perm):
    x = torch.randn(*base).to(dtype)
    xt = x.to("tt")
    torch.tt.synchronize()
    uploads = kohakutt.counters()["uploads"]
    got = (xt.view(*view) if view else xt).permute(*perm).contiguous().cpu()
    assert kohakutt.counters()["uploads"] == uploads  # no host-built index
    assert torch.equal(got, (x.view(*view) if view else x).permute(*perm).contiguous())
