"""Regression: FFTs (direct, four-step, Bluestein lengths; multi-dim; inverse) and
complex64 tensors run on tt with the right shapes and dtypes and no host transfers;
precision is reported by the coverage stats, not asserted here."""

import pytest
import torch

import kohakutt


def transfers():
    """Host uploads + downloads so far."""
    c = kohakutt.counters()
    return c["uploads"] + c["downloads"]


@pytest.mark.hw
@pytest.mark.parametrize("n", [100, 1024, 2048, 1031])
def test_rfft_irfft_lengths(native, n):
    x = torch.randn(3, n).to("tt")
    torch.fft.irfft(torch.fft.rfft(x), n=n)
    before = transfers()
    spectrum = torch.fft.rfft(x)
    back = torch.fft.irfft(spectrum, n=n)
    assert transfers() == before
    assert spectrum.shape == (3, n // 2 + 1) and spectrum.dtype == torch.complex64
    assert back.shape == (3, n) and back.dtype == torch.float32


@pytest.mark.hw
def test_fft2_ortho_and_shift(native):
    z = torch.randn(2, 16, 24, dtype=torch.complex64).to("tt")
    got = torch.fft.fftshift(torch.fft.fft2(z, norm="ortho"))
    assert got.shape == z.shape and got.dtype == torch.complex64


@pytest.mark.hw
def test_complex_elementwise_and_conj(native):
    a = torch.randn(4, 8, dtype=torch.complex64)
    at = a.to("tt")
    assert (at * at.conj() + at / at).dtype == torch.complex64
    assert at.abs().dtype == torch.float32
    assert torch.equal(at.conj().resolve_conj().cpu(), a.conj())


@pytest.mark.hw
def test_spectral_loss_grad(native):
    got = torch.randn(4, 128).to("tt").requires_grad_()
    torch.fft.rfft(got).abs().sum().backward()
    assert got.grad.shape == got.shape and got.grad.device.type == "tt"
