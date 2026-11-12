"""Kohaku-TT's weight-bound matmul (ktt_gemv, few rows of x, output columns over the
grid): error against CPU float64 next to CPU bf16, ragged and batched shapes, block
floats in both storages, the program it builds and its reuse."""

import os

import pytest
import torch
import torch.nn.functional as F

import kohakutt

ON = os.environ.get("KTT_GEMV") != "0" and os.environ.get("KTT_CUSTOM_KERNELS") != "0"


def _rel(y, ref):
    return ((y.double() - ref).norm() / ref.norm()).item()


def _case(shape, n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*shape, generator=g).to(torch.bfloat16)
    k = shape[-1]
    w = (torch.randn(n, k, generator=g) / k**0.5).to(torch.bfloat16)
    return x, w


@pytest.mark.parametrize(
    "shape,n",
    [
        ((1, 1, 896), 4864),
        ((1, 1, 1000), 1600),  # K and N ragged
        ((1, 2, 896), 32 * 49 + 5),  # the last core one column short, ragged N
        ((1, 3, 300), 2000),
        ((4, 1, 2048), 2048),  # a batch of 4 rows
        ((1, 64, 512), 1536),
        ((1, 128, 256), 3072),
        ((1, 1, 4864), 896),  # few columns: the block matmul's route
    ],
)
@pytest.mark.parametrize("layout", ["linear", "mm"])
def test_bf16_matches_cpu_bf16(native, shape, n, layout):
    x, w = _case(shape, n, sum(shape) + n)
    ref = x.double() @ w.double().t()
    cpu = _rel(x @ w.t(), ref)
    xt = x.to("tt")
    if layout == "linear":
        y = F.linear(xt, w.to("tt")).cpu()
    else:
        y = (xt @ w.t().contiguous().to("tt")).cpu()
    assert y.shape == ref.shape
    assert torch.isfinite(y).all()
    # HiFi2 multiplies (the bridge default) against CPU's exact bf16 products.
    assert _rel(y, ref) < 4 * cpu


def test_bias_and_rank3(native):
    x, w = _case((2, 1, 640), 3000, 3)
    b = torch.randn(3000, generator=torch.Generator().manual_seed(4)).to(torch.bfloat16)
    ref = x.double() @ w.double().t() + b.double()
    cpu = _rel(F.linear(x, w, b), ref)
    y = F.linear(x.to("tt"), w.to("tt"), b.to("tt")).cpu()
    assert y.shape == (2, 1, 3000)
    assert _rel(y, ref) < 4 * cpu


@pytest.mark.parametrize(
    "shape,n",
    [
        ((1, 1, 896), 896),  # the block matmul
        ((1, 1, 896), 128),
        ((1, 1, 896), 4864),  # ktt_gemv
        ((1, 3, 300), 2000),  # ragged K and N
        ((3, 1, 2048), 32 * 49 + 5),  # a batch of 3 rows, the last core short
        ((1, 64, 1000), 1000),  # the block matmul over K blocks
    ],
)
def test_bias_is_added_in_the_program(native, shape, n):
    x, w = _case(shape, n, n + 7)
    b = torch.randn(n, generator=torch.Generator().manual_seed(n)).to(torch.bfloat16)
    ref = x.double() @ w.double().t() + b.double()
    cpu = _rel(F.linear(x, w, b), ref)
    xt, wt, bt = x.to("tt"), w.to("tt"), b.to("tt")
    F.linear(xt, wt, bt)
    torch.tt.synchronize()
    c0 = kohakutt.counters()
    y = F.linear(xt, wt, bt)
    torch.tt.synchronize()
    c1 = kohakutt.counters()
    if ON:
        assert c1["programs"] - c0["programs"] == 1
    assert _rel(y.cpu(), ref) < 4 * cpu


@pytest.mark.parametrize("kn", [False, True])
@pytest.mark.parametrize("shape,n", [((1, 1, 896), 896), ((1, 128, 512), 1000)])
def test_bias_with_block_float_weights(native, kn, shape, n):
    x, w = _case(shape, n, 13)
    b = torch.randn(n, generator=torch.Generator().manual_seed(14)).to(torch.bfloat16)
    stored = w.t().contiguous().to("tt") if kn else w.to("tt")
    kohakutt.block_float_(stored, "bfp8")
    wt = stored.t() if kn else stored
    ref = x.double() @ wt.cpu().double().t() + b.double()
    cpu = _rel(F.linear(x, wt.cpu(), b), ref)
    y = F.linear(x.to("tt"), wt, b.to("tt")).cpu()
    assert _rel(y, ref) < 4 * cpu


@pytest.mark.parametrize("fmt", ["bfp8", "bfp4"])
@pytest.mark.parametrize("kn", [False, True])
@pytest.mark.parametrize("shape,n", [((1, 1, 896), 896), ((1, 2, 1024), 4000)])
def test_block_float_weights(native, fmt, kn, shape, n):
    x, w = _case(shape, n, 11)
    stored = w.t().contiguous().to("tt") if kn else w.to("tt")
    kohakutt.block_float_(stored, fmt)
    wt = stored.t() if kn else stored
    ref = x.double() @ wt.cpu().double().t()  # the stored (rounded) weights
    cpu = _rel(x @ wt.cpu().t(), ref)
    y = F.linear(x.to("tt"), wt).cpu()
    assert _rel(y, ref) < 4 * cpu


def test_rows_of_one_tile_share_a_program(native):
    # Rows 1 and 7 of x fall in one row tile: the second call builds nothing, and
    # row 0's sums run in the same order (equal bit for bit).
    x, w = _case((1, 7, 768), 3200, 5)
    wt = w.to("tt")
    one = F.linear(x[:, :1].to("tt"), wt).cpu()
    torch.tt.synchronize()
    c0 = kohakutt.counters()
    seven = F.linear(x.to("tt"), wt).cpu()
    torch.tt.synchronize()
    c1 = kohakutt.counters()
    if ON:
        assert c1["programs"] - c0["programs"] == 1
    assert c1["program_entries"] == c0["program_entries"]
    assert c1["kernel_compiles"] == c0["kernel_compiles"]
    assert torch.equal(seven[:, :1], one)


def test_repeat_calls_are_bit_identical(native):
    x, w = _case((1, 1, 2048), 6144, 9)
    xt, wt = x.to("tt"), w.to("tt")
    first = F.linear(xt, wt).cpu()
    for _ in range(3):
        assert torch.equal(F.linear(xt, wt).cpu(), first)
