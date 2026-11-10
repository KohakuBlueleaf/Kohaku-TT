"""Complex64 arithmetic on interleaved fp32 pairs: the multiply as one program, bit for
bit against the composite on fp32 parts (re = ar br - ai bi, im = ar bi + ai br); add,
sub, neg and real-scalar mul / div against CPU; conjugated operands against CPU."""

import pytest
import torch

# Relative bound (max error over max magnitude) of the paths that are not bit-exact.
BOUND = 1e-6

# (a's shape, b's shape) of a * b.
MUL_CASES = {
    "heads": ((2, 4, 96, 32), (1, 1, 96, 32)),
    "2-d b": ((2, 4, 96, 32), (96, 32)),
    "same shape": ((2, 4, 96, 32), (2, 4, 96, 32)),
    "ragged": ((3, 2, 50, 13), (50, 13)),
    "model, 9 crops": ((9, 16, 576, 32), (1, 1, 576, 32)),
    "model, 1 image": ((1, 16, 5184, 32), (1, 1, 5184, 32)),
}


def _operands(a_shape, b_shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.view_as_complex(torch.randn(*a_shape, 2, generator=g))
    b = torch.view_as_complex(torch.randn(*b_shape, 2, generator=g))
    return a, b


def _composite(a, b):
    """a * b from fp32 products and sums of the parts, each rounded on its own."""
    ar, ai = a.real, a.imag
    br, bi = b.real, b.imag
    return torch.stack([ar * br - ai * bi, ar * bi + ai * br], -1)


def _pairs(z):
    """The fp32 pairs [..., 2] of a complex tt tensor, on CPU."""
    assert z.dtype == torch.complex64 and z.device.type == "tt"
    return torch.view_as_real(z.cpu())


def _error(got, want):
    """Max error of complex tt `got` against CPU `want`, relative to want's max."""
    assert got.dtype == want.dtype and got.shape == want.shape
    return ((got.cpu() - want).abs().max() / want.abs().max()).item()


@pytest.mark.hw
@pytest.mark.parametrize("case", list(MUL_CASES))
def test_mul_matches_composite_bits(native, case):
    a, b = _operands(*MUL_CASES[case])
    got = a.to("tt") * b.to("tt")
    assert got.shape == a.shape
    assert torch.equal(_pairs(got), _composite(a, b))


@pytest.mark.hw
def test_mul_operands_swapped(native):
    a, b = _operands(*MUL_CASES["heads"])
    got = b.to("tt") * a.to("tt")
    assert got.shape == a.shape
    assert torch.equal(_pairs(got), _composite(b, a))


@pytest.mark.hw
def test_mul_is_one_program(native):
    a, b = _operands(*MUL_CASES["heads"])
    at, bt = a.to("tt"), b.to("tt")
    at * bt
    native.synchronize()
    before = native.counters()
    got = at * bt
    native.synchronize()
    after = native.counters()
    assert after["programs"] - before["programs"] == 1
    assert torch.equal(_pairs(got), _composite(a, b))


@pytest.mark.hw
def test_add_sub_neg_exact(native):
    a, b = _operands((2, 4, 96, 32), (2, 4, 96, 32), seed=1)
    at, bt = a.to("tt"), b.to("tt")
    assert torch.equal(_pairs(at + bt), torch.view_as_real(a + b))
    assert torch.equal(_pairs(at - bt), torch.view_as_real(a - b))
    assert torch.equal(_pairs(-at), torch.view_as_real(-a))


@pytest.mark.hw
def test_real_scalar_mul_div(native):
    a, _ = _operands((2, 4, 96, 32), (1,), seed=2)
    at = a.to("tt")
    assert torch.equal(_pairs(at * 1.7), torch.view_as_real(a * 1.7))
    assert _error(at / 1.7, a / 1.7) <= BOUND


@pytest.mark.hw
def test_conjugated_operands(native):
    a, b = _operands(*MUL_CASES["heads"], seed=3)
    at, bt = a.to("tt"), b.to("tt")
    assert _error(at.conj() * bt, a.conj() * b) <= BOUND
    assert _error(at * bt.conj(), a * b.conj()) <= BOUND
