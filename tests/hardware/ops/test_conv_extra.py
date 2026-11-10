"""Regression: bf16 convolutions with one channel over a single row (TT-NN's conv2d is
wrong there), convolutions and matmuls whose last output column group is narrower than
the others, 3-D convolutions, quantile and fractional max pooling agree with PyTorch on
CPU."""

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


def _worst_column_tile(got, want):
    """The lowest cosine over the 32-column tiles of the last dim."""
    return min(
        _cosine(got[..., c : c + 32], want[..., c : c + 32])
        for c in range(0, want.shape[-1], 32)
    )


@pytest.mark.hw
@pytest.mark.parametrize("out_channels", [160, 192, 224])
def test_conv_narrow_last_column_group(native, out_channels):
    """5, 6 and 7 output channel tiles in groups of 4: the last group has 1, 2 and 3;
    every core takes several output row tiles."""
    x = torch.randn(1, 32, 96, 96).bfloat16()
    w = (torch.randn(out_channels, 32, 3, 3) / 17).bfloat16()
    got = F.conv2d(x.to("tt"), w.to("tt"), padding=1).cpu()
    want = F.conv2d(x.double(), w.double(), padding=1)
    assert _worst_column_tile(got.movedim(1, -1), want.movedim(1, -1)) > 0.999


@pytest.mark.hw
@pytest.mark.parametrize("n", [800, 832, 864])
def test_matmul_narrow_last_column_group(native, n):
    """25, 26 and 27 output column tiles in groups of 4, as one launch of Kohaku-TT's
    matmul (not TT-NN's)."""
    x = torch.randn(16384, 64).bfloat16()
    w = (torch.randn(64, n) / 8).bfloat16()
    xt, wt = x.to("tt"), w.to("tt")
    torch.tt.synchronize()
    before = native.counters()["programs"]
    got = (xt @ wt).cpu()
    assert native.counters()["programs"] - before == 1
    assert _worst_column_tile(got, x.double() @ w.double()) > 0.999


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
