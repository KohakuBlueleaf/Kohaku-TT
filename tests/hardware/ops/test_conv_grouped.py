"""Grouped, depthwise and one-group strided 1-D convolutions on Kohaku-TT's ktt_conv2:
forward, input gradient (any stride) and weight gradient match CPU float64 at least as
closely as CPU bf16 does; block-float weights (bfp8 / bfp4) against the same weights
dequantized; the work runs as Kohaku-TT programs (one bridge op, no composite)."""

import pytest
import torch
import torch.nn.functional as F

import kohakutt


def cosine(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def sample(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g)


def delta(before):
    """Counter increments since `before` (a kohakutt.counters() dict)."""
    kohakutt.synchronize()
    now = kohakutt.counters()
    return {k: now[k] - before[k] for k in ("ops", "programs")}


# (N, C, H, W, O, groups, kernel, stride, dilation, padding); H == 1: a 1-D conv.
CASES_2D = [
    (1, 96, 56, 56, 96, 96, 7, 1, 1, 3),  # ConvNeXt-T stage 1 depthwise
    (1, 768, 7, 7, 768, 768, 7, 1, 1, 3),  # ConvNeXt-T stage 4 depthwise
    (2, 48, 23, 37, 48, 48, 3, 2, 1, 1),  # ragged C / H / W, stride 2
    (1, 144, 28, 28, 144, 144, 5, 2, 1, 2),  # MobileNet-style ragged C
    (1, 64, 30, 30, 128, 64, 3, 1, 2, 2),  # channel multiplier 2, dilation 2
    (1, 64, 56, 56, 64, 2, 3, 1, 1, 1),  # groups 2 (32 channels each)
    (1, 64, 56, 56, 64, 16, 3, 1, 1, 1),  # groups 16 (4 channels each)
    (1, 64, 56, 56, 64, 16, 7, 4, 1, 3),  # grouped, stride 4
    (1, 128, 33, 41, 128, 32, 11, 3, 1, 5),  # kernel 11, stride 3
    (1, 128, 40, 40, 256, 4, 3, 1, 4, 4),  # groups C/32, dilation 4
    (1, 96, 20, 20, 96, 24, 1, 1, 1, 0),  # 1x1 grouped
    (1, 64, 19, 19, 64, 2, 11, 1, 1, 5),  # groups 2, kernel 11
    (1, 32, 17, 17, 32, 32, 2, 3, 2, 0),  # kernel 2 < stride 3: empty phases
]
CASES_1D = [
    (1, 256, 1, 4096, 256, 1, 3, 2, 1, 1),  # strided 1-D
    (2, 64, 1, 999, 128, 1, 7, 4, 2, 3),  # strided 1-D, ragged length
    (1, 128, 1, 512, 128, 128, 5, 1, 1, 2),  # depthwise 1-D
    (1, 64, 1, 300, 64, 16, 11, 3, 1, 5),  # grouped 1-D, kernel 11, stride 3
]
CASES = CASES_2D + CASES_1D


def tensors(case):
    n, c, h, w, o, groups, k, s, d, p = case
    one_d = h == 1
    x = sample((n, c, w) if one_d else (n, c, h, w), 0)
    wt = sample((o, c // groups, k) if one_d else (o, c // groups, k, k), 1) * 0.2
    conv = F.conv1d if one_d else F.conv2d
    return x, wt, conv, (s, p, d, groups)


@pytest.mark.hw
@pytest.mark.parametrize("case", CASES)
def test_grouped_conv_matches_float64(native, case):
    x, wt, conv, args = tensors(case)
    xt, wtt = x.bfloat16().to("tt"), wt.bfloat16().to("tt")
    kohakutt.synchronize()
    before = kohakutt.counters()
    got = conv(xt, wtt, None, *args)
    counts = delta(before)
    got = got.cpu()
    want = conv(x.double(), wt.double(), None, *args)
    cpu16 = conv(x.bfloat16(), wt.bfloat16(), None, *args)
    assert counts["ops"] == 1 and counts["programs"] >= 3, counts
    assert got.shape == want.shape
    assert torch.isfinite(got).all()
    tt_cos, cpu_cos = cosine(got, want), cosine(cpu16, want)
    print(f"{case}: tt cos {tt_cos:.7f}  cpu bf16 cos {cpu_cos:.7f}")
    assert tt_cos >= min(0.9999, cpu_cos - 1e-4)


@pytest.mark.hw
@pytest.mark.parametrize("case", [CASES[0], CASES[2], CASES[9], CASES[11], CASES[12]])
def test_grouped_conv_k_blocks(native, case, monkeypatch):
    """The K-blocked variant (at most 5 taps per block, fp32 sums in L1), forward and
    input gradient."""
    monkeypatch.setenv("KTT_CONV2_BLOCK", "5")
    x, wt, conv, args = tensors(case)
    grad = sample(conv(x, wt, None, *args).shape, 2)
    xt = x.bfloat16().to("tt").requires_grad_()
    out = conv(xt, wt.bfloat16().to("tt"), None, *args)
    (gi,) = torch.autograd.grad(out, xt, grad.bfloat16().to("tt"))
    xd = x.double().requires_grad_()
    want = conv(xd, wt.double(), None, *args)
    (want_gi,) = torch.autograd.grad(want, xd, grad.double())
    xb = x.bfloat16().requires_grad_()
    ref = conv(xb, wt.bfloat16(), None, *args)
    (ref_gi,) = torch.autograd.grad(ref, xb, grad.bfloat16())
    for got, w64, w16 in ((out, want, ref), (gi, want_gi, ref_gi)):
        tt_cos, cpu_cos = cosine(got.cpu(), w64), cosine(w16, w64)
        print(f"blocks {case}: tt cos {tt_cos:.7f}  cpu bf16 cos {cpu_cos:.7f}")
        assert tt_cos >= min(0.9999, cpu_cos - 1e-4)


@pytest.mark.hw
@pytest.mark.parametrize("split", ["1", "3"])
@pytest.mark.parametrize("blocks", [None, "2"])
@pytest.mark.parametrize("case", [CASES[1], CASES[5], CASES[6], CASES[8]])
def test_grouped_conv_tap_row_split(native, case, split, blocks, monkeypatch):
    """The tap-row split (fp32 partial sums per kernel-row range, summed into the output),
    forced on, alone and with K blocks; forward and input gradient."""
    monkeypatch.setenv("KTT_CONV2_SPLIT", split)
    if blocks:
        monkeypatch.setenv("KTT_CONV2_BLOCK", blocks)
    x, wt, conv, args = tensors(case)
    grad = sample(conv(x, wt, None, *args).shape, 2)
    xt = x.bfloat16().to("tt").requires_grad_()
    out = conv(xt, wt.bfloat16().to("tt"), None, *args)
    (gi,) = torch.autograd.grad(out, xt, grad.bfloat16().to("tt"))
    xd = x.double().requires_grad_()
    want = conv(xd, wt.double(), None, *args)
    (want_gi,) = torch.autograd.grad(want, xd, grad.double())
    xb = x.bfloat16().requires_grad_()
    ref = conv(xb, wt.bfloat16(), None, *args)
    (ref_gi,) = torch.autograd.grad(ref, xb, grad.bfloat16())
    for got, w64, w16 in ((out, want, ref), (gi, want_gi, ref_gi)):
        tt_cos, cpu_cos = cosine(got.cpu(), w64), cosine(w16, w64)
        print(f"split {split} {case}: tt cos {tt_cos:.7f}  cpu bf16 cos {cpu_cos:.7f}")
        assert tt_cos >= min(0.9999, cpu_cos - 1e-4)


@pytest.mark.hw
@pytest.mark.parametrize("fmt", ["bfp8", "bfp4"])
@pytest.mark.parametrize("case", [CASES[0], CASES[4], CASES[5], CASES[6]])
def test_grouped_conv_block_float_weights(native, case, fmt):
    x, wt, conv, args = tensors(case)
    wtt = wt.bfloat16().to("tt")
    kohakutt.block_float_(wtt, fmt)
    got = conv(x.bfloat16().to("tt"), wtt, None, *args).cpu()
    stored = wtt.cpu()
    want = conv(x.double(), wt.double(), None, *args)
    # The weights as stored (dequantized) through a float64 conv.
    same = conv(x.double(), stored.double(), None, *args)
    tt_cos, quant_cos = cosine(got, want), cosine(same, want)
    print(f"{fmt} {case}: tt cos {tt_cos:.6f}  cpu {fmt}-weights cos {quant_cos:.6f}")
    assert tt_cos >= quant_cos - 1e-4


@pytest.mark.hw
@pytest.mark.parametrize("case", CASES)
def test_grouped_conv_input_grad(native, case):
    x, wt, conv, args = tensors(case)
    grad = sample(conv(x, wt, None, *args).shape, 2)
    xt = x.bfloat16().to("tt").requires_grad_()
    out = conv(xt, wt.bfloat16().to("tt"), None, *args)
    gt = grad.bfloat16().to("tt")
    kohakutt.synchronize()
    before = kohakutt.counters()
    (got,) = torch.autograd.grad(out, xt, gt)
    counts = delta(before)
    got = got.cpu()

    def input_grad(dtype):
        xr = x.to(dtype).requires_grad_()
        conv(xr, wt.to(dtype), None, *args).backward(grad.to(dtype))
        return xr.grad

    want, cpu16 = input_grad(torch.float64), input_grad(torch.bfloat16)
    assert counts["ops"] == 1 and counts["programs"] >= 3, counts
    assert got.shape == want.shape
    tt_cos, cpu_cos = cosine(got, want), cosine(cpu16, want)
    print(f"dgrad {case}: tt cos {tt_cos:.7f}  cpu bf16 cos {cpu_cos:.7f}")
    assert tt_cos >= min(0.9999, cpu_cos - 1e-4)


@pytest.mark.hw
@pytest.mark.parametrize("case", CASES)
def test_grouped_conv_weight_grad(native, case):
    x, wt, conv, args = tensors(case)
    grad = sample(conv(x, wt, None, *args).shape, 2)
    wtt = wt.bfloat16().to("tt").requires_grad_()
    out = conv(x.bfloat16().to("tt"), wtt, None, *args)
    gt = grad.bfloat16().to("tt")
    kohakutt.synchronize()
    before = kohakutt.counters()
    (got,) = torch.autograd.grad(out, wtt, gt)
    counts = delta(before)
    got = got.cpu()

    def weight_grad(dtype):
        wr = wt.to(dtype).requires_grad_()
        conv(x.to(dtype), wr, None, *args).backward(grad.to(dtype))
        return wr.grad

    want, cpu16 = weight_grad(torch.float64), weight_grad(torch.bfloat16)
    assert counts["ops"] == 1 and counts["programs"] >= 3, counts
    tt_cos, cpu_cos = cosine(got, want), cosine(cpu16, want)
    print(f"wgrad {case}: tt cos {tt_cos:.7f}  cpu bf16 cos {cpu_cos:.7f}")
    assert tt_cos >= min(0.9999, cpu_cos - 1e-4)
