"""Regression: bf16 convolutions with one channel over a single row (TT-NN's conv2d is
wrong there), 3-D convolutions, quantile and fractional max pooling agree with PyTorch
on CPU."""

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _cosine(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float((a @ b) / (a.norm() * b.norm()))


@pytest.mark.hw
@pytest.mark.parametrize("padding", ["same", "valid", 1])
def test_single_row_single_channel_conv(native, padding):
    x = torch.randn(2, 1, 10).bfloat16()
    w = torch.randn(1, 1, 4).bfloat16()
    got = F.conv1d(x.to("tt"), w.to("tt"), padding=padding).cpu()
    assert _cosine(got, F.conv1d(x.double(), w.double(), padding=padding)) > 0.999


@pytest.mark.hw
@pytest.mark.parametrize("transposed", [False, True])
def test_conv3d(native, transposed):
    x = torch.randn(2, 4, 5, 6, 7)
    if transposed:
        w = torch.randn(4, 3, 3, 3, 3)
        fn = lambda a, b: F.conv_transpose3d(a, b, stride=2, padding=1)  # noqa: E731
    else:
        w = torch.randn(3, 4, 3, 3, 3)
        fn = lambda a, b: F.conv3d(a, b, stride=(1, 2, 1), padding=1)  # noqa: E731
    got = fn(x.to("tt"), w.to("tt")).cpu()
    want = fn(x.double(), w.double())
    assert got.shape == want.shape
    assert _cosine(got, want) > 0.999


@pytest.mark.hw
def test_quantile(native):
    x = torch.randn(3, 101)
    q = torch.tensor([0.0, 0.25, 0.5, 0.9, 1.0])
    for mode in ("linear", "lower", "higher", "midpoint", "nearest"):
        got = torch.quantile(x.to("tt"), q.to("tt"), dim=-1, interpolation=mode)
        torch.testing.assert_close(
            got.cpu(), torch.quantile(x, q, dim=-1, interpolation=mode)
        )


@pytest.mark.hw
def test_fractional_max_pool_given_samples(native):
    x = torch.randn(2, 3, 9, 9).bfloat16()
    s = torch.rand(2, 3, 2).bfloat16()
    want = F.fractional_max_pool2d(
        x, 2, output_size=4, return_indices=True, _random_samples=s
    )
    got = F.fractional_max_pool2d(
        x.to("tt"), 2, output_size=4, return_indices=True, _random_samples=s.to("tt")
    )
    assert torch.equal(got[0].cpu(), want[0])
    assert torch.equal(got[1].cpu(), want[1])
