"""F.gelu(x, approximate="tanh") on tt as one program, against CPU fp32; the exact form
and the tanh form's backward against CPU.

Bounds: bf16 max abs error <= 2e-2 and rel RMS <= 4e-3; fp32 rel RMS <= 1e-4."""

import pytest
import torch
import torch.nn.functional as F

MAX_ABS = 2e-2
BOUND = {torch.bfloat16: 4e-3, torch.float32: 1e-4}
# The input gradient against float64 autograd, rel RMS.
GRAD_BOUND = 2e-2

# (shape, dtype)
CASES = [
    ((1, 128, 256), torch.bfloat16),
    ((1, 128, 6144), torch.bfloat16),
    ((1, 128, 12288), torch.bfloat16),
    ((3, 50, 13), torch.bfloat16),
    ((2, 96, 64), torch.float32),
]


def _input(shape, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * 3).to(dtype)


def _reference(x, approximate):
    """gelu of `x` in CPU fp32, cast to x's dtype."""
    return F.gelu(x.float(), approximate=approximate).to(x.dtype)


def _errors(got, want):
    """(max abs error, rel RMS error) of tt `got` against CPU `want`."""
    assert (
        got.device.type == "tt" and got.dtype == want.dtype and got.shape == want.shape
    )
    diff = got.cpu().double() - want.double()
    return diff.abs().max().item(), (diff.norm() / want.double().norm()).item()


@pytest.mark.parametrize(
    "shape,dtype", CASES, ids=lambda v: str(v).replace("torch.", "")
)
def test_tanh_form(native, shape, dtype):
    x = _input(shape, dtype)
    want = _reference(x, "tanh")
    got = F.gelu(x.to("tt"), approximate="tanh")
    max_abs, rel_rms = _errors(got, want)
    differ = got.cpu() != want
    print(
        f"\ngelu tanh {tuple(shape)} {dtype}: max abs {max_abs:.3e}, "
        f"rel RMS {rel_rms:.3e}, "
        f"{differ.sum().item()} of {want.numel()} elements differ, "
        f"{(differ & (want.abs() >= 1e-4)).sum().item()} of them "
        "with |reference| >= 1e-4"
    )
    assert rel_rms <= BOUND[dtype]
    if dtype == torch.bfloat16:
        assert max_abs <= MAX_ABS


@pytest.mark.parametrize(
    "shape,dtype", CASES, ids=lambda v: str(v).replace("torch.", "")
)
def test_tanh_form_is_one_program(native, shape, dtype):
    x = _input(shape, dtype, 1).to("tt")
    want = F.gelu(x, approximate="tanh").cpu()
    native.synchronize()
    before = native.counters()
    y = F.gelu(x, approximate="tanh")
    native.synchronize()
    after = native.counters()
    assert after["ops"] - before["ops"] == 1
    # Kohaku-TT's own unary kernel runs from 512 tiles; TT-NN's op below.
    own = x.numel() >= 512 * 1024
    assert after["programs"] - before["programs"] == (1 if own else 0)
    assert torch.equal(y.cpu(), want)


@pytest.mark.parametrize(
    "shape,dtype", CASES, ids=lambda v: str(v).replace("torch.", "")
)
def test_exact_form(native, shape, dtype):
    x = _input(shape, dtype, 2)
    max_abs, rel_rms = _errors(F.gelu(x.to("tt")), _reference(x, "none"))
    assert rel_rms <= BOUND[dtype]
    if dtype == torch.bfloat16:
        assert max_abs <= MAX_ABS


def test_tanh_form_backward(native):
    x = _input((1, 128, 256), torch.bfloat16, 3)
    grad = _input((1, 128, 256), torch.bfloat16, 4)
    ref = x.double().requires_grad_()
    (want,) = torch.autograd.grad(F.gelu(ref, approximate="tanh"), ref, grad.double())
    leaf = x.to("tt").requires_grad_()
    (got,) = torch.autograd.grad(F.gelu(leaf, approximate="tanh"), leaf, grad.to("tt"))
    assert got.device.type == "tt" and got.dtype == torch.bfloat16
    error = ((got.cpu().double() - want).norm() / want.norm()).item()
    assert error <= GRAD_BOUND
