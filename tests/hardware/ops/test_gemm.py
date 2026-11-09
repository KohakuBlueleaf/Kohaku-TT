"""Kohaku-TT's block matmul (ktt_gemm): error statistics against CPU float64 next to CPU
at the same precision, over ragged, batched and block-float cases."""

import pytest
import torch
import torch.nn.functional as F

import kohakutt


def _rel(y, ref):
    return ((y.double() - ref).norm() / ref.norm()).item()


def _case(m, k, n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator=g).to(torch.bfloat16)
    w = (torch.randn(n, k, generator=g) / k**0.5).to(torch.bfloat16)
    return x, w


@pytest.mark.parametrize(
    "m,k,n",
    [(32, 32, 32), (256, 256, 256), (77, 1000, 333), (1024, 1000, 768), (2048, 4096, 4096),
     (64, 8192, 128), (4096, 128, 64), (1, 4096, 4096)],
)
@pytest.mark.parametrize("layout", ["linear", "mm"])
def test_bf16_matches_cpu_bf16(native, m, k, n, layout):
    x, w = _case(m, k, n, m + k + n)
    ref = x.double() @ w.double().t()
    cpu = _rel(x @ w.t(), ref)
    xt = x.to("tt")
    if layout == "linear":
        y = F.linear(xt, w.to("tt")).cpu()
    else:
        y = (xt @ w.t().contiguous().to("tt")).cpu()
    assert torch.isfinite(y).all()
    # HiFi2 multiplies (the bridge default) against CPU's exact bf16 products.
    assert _rel(y, ref) < 4 * cpu


def test_batched_input_and_bias(native):
    g = torch.Generator().manual_seed(3)
    x = torch.randn(3, 65, 300, generator=g).to(torch.bfloat16)
    w = (torch.randn(200, 300, generator=g) / 300**0.5).to(torch.bfloat16)
    b = torch.randn(200, generator=g).to(torch.bfloat16)
    ref = x.double() @ w.double().t() + b.double()
    cpu = _rel(F.linear(x, w, b), ref)
    y = F.linear(x.to("tt"), w.to("tt"), b.to("tt")).cpu()
    assert y.shape == (3, 65, 200)
    assert _rel(y, ref) < 4 * cpu


@pytest.mark.parametrize("fmt", ["bfp8", "bfp4"])
@pytest.mark.parametrize("m,k,n", [(1024, 1000, 768), (2048, 4096, 4096)])
def test_block_float_weights(native, fmt, m, k, n):
    x, w = _case(m, k, n, 11)
    ref = x.double() @ w.double().t()
    wt = w.to("tt")
    kohakutt.block_float_(wt, fmt)
    y = F.linear(x.to("tt"), wt).cpu()
    # Weight quantization dominates: bfp8 about 8e-3, bfp4 about 0.21 relative RMS.
    assert _rel(y, ref) < (0.012 if fmt == "bfp8" else 0.25)


def test_repeat_calls_are_bit_identical(native):
    x, w = _case(1024, 1000, 768, 5)
    xt, wt = x.to("tt"), w.to("tt")
    first = F.linear(xt, wt).cpu()
    for _ in range(3):
        assert torch.equal(F.linear(xt, wt).cpu(), first)
