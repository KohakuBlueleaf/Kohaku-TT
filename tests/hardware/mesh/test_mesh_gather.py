"""Mesh: gather along the last dim of a replicated, a sharded and a pending-sum input;
values are small integers, so every result is exact."""

import pytest
import torch
from mesh_support import place, read

pytestmark = [pytest.mark.hw, pytest.mark.dual]


@pytest.mark.parametrize("spec", ["R", "S0", "S1", "P"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_gather_last_dim(native_module, mesh, spec, dtype):
    gen = torch.Generator().manual_seed(3)
    value = torch.randint(-4, 5, (64, 96), generator=gen).to(dtype)
    index = torch.randint(0, 96, (64, 40), generator=gen)
    x = place(native_module, mesh, value, spec)
    want = torch.gather(read(x), 1, index)
    got = torch.gather(x, 1, index.to(mesh))
    assert torch.equal(read(got), want)
