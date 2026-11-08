"""Regression: integer elementwise ops (true division's dtype, remainder / fmod, shifts,
gcd) equal PyTorch's on CPU."""

import pytest
import torch

A = torch.tensor([5, 7, 11, -3, 0, 23, -17])
B = torch.tensor([3, 4, -4, 5, 2, 6, 5])


@pytest.mark.hw
def test_int_true_division(native):
    got = A.to("tt") / B.to("tt")
    assert got.dtype == torch.float32
    torch.testing.assert_close(got.cpu(), A / B)
    assert torch.equal((A.to("tt") - torch.floor(got)).cpu(), A - torch.floor(A / B))


@pytest.mark.hw
@pytest.mark.parametrize("op", [torch.remainder, torch.fmod, torch.gcd])
def test_int_modulo(native, op):
    assert torch.equal(op(A.to("tt"), B.to("tt")).cpu(), op(A, B))


@pytest.mark.hw
def test_int_shifts(native):
    s = B.abs() % 5
    assert torch.equal((A.to("tt") << s.to("tt")).cpu(), A << s)
    assert torch.equal((A.to("tt") >> s.to("tt")).cpu(), A >> s)
