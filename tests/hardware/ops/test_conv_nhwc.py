"""Kohaku-TT's channels-last conv (one group, stride 1, dilation 1, odd K x K kernel,
padding (K - 1) / 2; 1x1 as a matmul) against float64, with KTT_NHWC=1: Nchw and Nhwc
inputs, with and without bias. The output is stored as Nhwc with zero tile padding,
reads back as NCHW and feeds the next op.

Bounds on the relative RMS error against float64: at HiFi4 at most 1.05x CPU bf16's
(products exact, summed in fp32); at the default fidelity (HiFi2) at most 3x CPU bf16's,
since at HiFi2 the activations (srcB) keep 1+6 of bf16's 1+7 mantissa bits, about 2.6x
CPU bf16's error. The route adds the bias to the bf16 conv output, rounding twice (as
conv2d()'s other routes do), so the CPU bf16 reference rounds the conv to bf16 and then
adds the bias in bf16; against a single rounding the error would be about sqrt(2) times
larger.
"""

import math

import pytest
import torch
import torch.nn.functional as F

# (N, C, H, W, O, K)
SMALL = [
    (1, 3, 17, 17, 8, 3),
    (2, 33, 7, 9, 40, 3),
    (1, 32, 9, 11, 3, 3),  # three output channels
    (1, 32, 12, 10, 32, 5),
    (1, 32, 12, 10, 32, 7),
    (1, 64, 15, 17, 64, 3),  # two Cout groups, planes feeders
    (1, 64, 15, 17, 128, 3),  # four Cout groups, planes feeders
    (1, 32, 12, 10, 32, 1),  # 1x1: matmul on the Nhwc rows
    (1, 40, 9, 11, 24, 1),
    (2, 33, 7, 9, 40, 1),
]
MODELS = [
    (1, 64, 56, 56, 64, 3),  # ResNet
    (1, 512, 64, 64, 512, 3),  # VAE decoder
    (2, 320, 32, 32, 320, 3),  # SD UNet
    (1, 512, 64, 64, 256, 1),  # VAE decoder 1x1 shortcut
]


def shape_id(s):
    n, c, h, w, o, k = s
    return f"{n}x{c}x{h}x{w}-{o}k{k}"


@pytest.fixture(autouse=True)
def _nhwc(monkeypatch):
    monkeypatch.setenv("KTT_NHWC", "1")


def runs():
    """Launches of the channels-last conv's program so far."""
    torch.tt.synchronize()
    return torch.ops.ktt._debug_nhwc_conv_runs()


def fallback_count():
    """Channels-last convs sent to conv2d()'s route so far."""
    torch.tt.synchronize()
    return torch.ops.ktt._debug_nhwc_conv_fallbacks()


def layout(t):
    return torch.ops.ktt._debug_layout(t)


def sample(shape, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=g) * scale).to(torch.bfloat16)


def rel(got, ref):
    """Relative RMS error of got against float64 ref."""
    return ((got.double() - ref).norm() / ref.norm()).item()


def problem(shape, bias, seed=0):
    """x, w, b (or None) in bf16, the float64 reference and CPU bf16's relative error:
    the conv rounded to bf16, then the bias added in bf16, as the route does."""
    n, c, h, w, o, k = shape
    x = sample((n, c, h, w), seed)
    wt = sample((o, c, k, k), seed + 1, 1 / math.sqrt(c * k * k))
    b = sample((o,), seed + 2) if bias else None
    p = (k - 1) // 2
    ref = F.conv2d(x.double(), wt.double(), None if b is None else b.double(), padding=p)
    cpu = F.conv2d(x.float(), wt.float(), padding=p).bfloat16()
    if b is not None:
        cpu = cpu + b.reshape(1, o, 1, 1)
    return x, wt, b, ref, rel(cpu, ref)


def as_nhwc(x):
    t = x.to("tt")
    torch.ops.ktt._debug_to_nhwc(t)
    assert layout(t) == "nhwc"
    return t


def conv(native, x, wt, b, k, fidelity=None):
    """F.conv2d on tt tensors, under `fidelity` when given."""
    bt = None if b is None else b.to("tt")
    if fidelity is None:
        return F.conv2d(x, wt.to("tt"), bt, padding=(k - 1) // 2)
    with native.numerics(fidelity=fidelity):
        return F.conv2d(x, wt.to("tt"), bt, padding=(k - 1) // 2)


def check(native, shape, bias, nhwc_input):
    x, wt, b, ref, cpu_rel = problem(shape, bias)
    k = shape[5]
    xt = as_nhwc(x) if nhwc_input else x.to("tt")
    before = runs()
    y4 = conv(native, xt, wt, b, k, "HiFi4")
    y2 = conv(native, xt, wt, b, k)
    assert runs() - before == 2, "the channels-last conv did not run"
    assert layout(y4) == "nhwc" and layout(y2) == "nhwc"
    assert torch.ops.ktt._debug_padding_max(y2) == 0, "nonzero tile padding"
    rel4, rel2 = rel(y4.cpu(), ref), rel(y2.cpu(), ref)
    print(f"{shape_id(shape)}: rel HiFi4 {rel4:.3e}, default {rel2:.3e}, cpu bf16 {cpu_rel:.3e}")
    assert rel4 <= 1.05 * cpu_rel, (rel4, cpu_rel)
    assert rel2 <= 3 * cpu_rel, (rel2, cpu_rel)
    return y2


@pytest.mark.hw
@pytest.mark.parametrize("bias", [False, True], ids=["nobias", "bias"])
@pytest.mark.parametrize("shape", SMALL, ids=shape_id)
def test_small(native, shape, bias):
    check(native, shape, bias, nhwc_input=False)
    check(native, shape, bias, nhwc_input=True)


@pytest.mark.hw
@pytest.mark.parametrize("shape", MODELS, ids=shape_id)
def test_models(native, shape):
    check(native, shape, True, nhwc_input=True)


@pytest.mark.hw
@pytest.mark.parametrize("shape", [(1, 64, 15, 17, 64, 3), (1, 40, 9, 11, 24, 1)], ids=shape_id)
def test_trace_reads_current_weight(native, shape):
    """A captured conv replays with the weight prepared before the capture, and with the
    new weight after the weight is written in place between replays."""
    x, wt, _, ref, cpu_rel = problem(shape, False)
    k = shape[5]
    xt, wtt = as_nhwc(x), wt.to("tt")
    fallbacks = fallback_count()
    step = native.Captured(lambda a: F.conv2d(a, wtt, padding=(k - 1) // 2), (xt,), 2)
    try:
        assert rel(step(xt).cpu(), ref) <= 3 * cpu_rel
        assert rel(step(xt).cpu(), ref) <= 3 * cpu_rel
        w2 = sample(wt.shape, 9, 1 / math.sqrt(shape[1] * k * k))
        wtt.copy_(w2.to("tt"))
        ref2 = F.conv2d(x.double(), w2.double(), padding=(k - 1) // 2)
        cpu2 = F.conv2d(x.float(), w2.float(), padding=(k - 1) // 2).bfloat16()
        assert rel(step(xt).cpu(), ref2) <= 3 * rel(cpu2, ref2)
    finally:
        step.release()
    assert fallback_count() == fallbacks


GRAD = [
    (2, 33, 7, 9, 40, 3),
    (1, 32, 12, 10, 32, 5),
    (1, 64, 15, 17, 128, 3),
    (1, 40, 9, 11, 24, 1),
    (2, 320, 32, 32, 320, 3),
]


@pytest.mark.hw
@pytest.mark.parametrize("shape", GRAD, ids=shape_id)
def test_input_grad(native, shape):
    """The input gradient runs on the channels-last conv (taps reversed, channels swapped):
    at HiFi4 within 1.05x CPU bf16's error against float64, at the default fidelity within
    3x. The weight gradient (another route) is checked against float64 too."""
    n, c, h, w, o, k = shape
    p = (k - 1) // 2
    x, wt, _, _, _ = problem(shape, False)
    gy = sample((n, o, h, w), 5)
    ref_gx = torch.nn.grad.conv2d_input(x.shape, wt.double(), gy.double(), padding=p)
    cpu_gx = torch.nn.grad.conv2d_input(x.shape, wt.float(), gy.float(), padding=p).bfloat16()
    ref_gw = torch.nn.grad.conv2d_weight(x.double(), wt.shape, gy.double(), padding=p)
    cpu_gw = torch.nn.grad.conv2d_weight(x.float(), wt.shape, gy.float(), padding=p).bfloat16()
    cpu_rel, cpu_rel_w = rel(cpu_gx, ref_gx), rel(cpu_gw, ref_gw)
    for fidelity, bound in (("HiFi4", 1.05), ("HiFi2", 3.0)):
        xt = x.to("tt").requires_grad_()
        wtt = wt.to("tt").requires_grad_()
        before = runs()
        with native.numerics(fidelity=fidelity):
            F.conv2d(xt, wtt, padding=p).backward(gy.to("tt"))
        assert runs() - before == 2, "forward and input gradient did not both run"
        got, got_w = rel(xt.grad.cpu(), ref_gx), rel(wtt.grad.cpu(), ref_gw)
        print(f"{shape_id(shape)} {fidelity}: input grad rel {got:.3e} (cpu bf16 {cpu_rel:.3e}),"
              f" weight grad rel {got_w:.3e} (cpu bf16 {cpu_rel_w:.3e})")
        assert got <= bound * cpu_rel, (fidelity, got, cpu_rel)
        assert got_w <= 4 * cpu_rel_w, (fidelity, got_w, cpu_rel_w)


@pytest.mark.hw
def test_trace_weight_written_in_capture(native):
    """A capture that writes the weight before the conv has no current prepared weight:
    that conv takes conv2d()'s route (counted) and each replay reads the new weight."""
    shape = (1, 64, 15, 17, 64, 3)
    x, wt, _, _, _ = problem(shape, False)
    w2 = sample(wt.shape, 9, 1 / math.sqrt(64 * 9))
    xt, wtt, w2t = as_nhwc(x), wt.to("tt"), w2.to("tt")

    def step_fn(a):
        wtt.copy_(w2t)
        return F.conv2d(a, wtt, padding=1)

    fallbacks = fallback_count()
    step = native.Captured(step_fn, (xt,), 2)
    try:
        got = step(xt).cpu()
    finally:
        step.release()
    assert fallback_count() > fallbacks
    ref = F.conv2d(x.double(), w2.double(), padding=1)
    cpu = F.conv2d(x.float(), w2.float(), padding=1).bfloat16()
    assert rel(got, ref) <= 3 * rel(cpu, ref)


@pytest.mark.hw
def test_output_feeds_next_ops(native):
    """The Nhwc output reads back as NCHW and feeds add, silu and a second conv."""
    shape = (2, 33, 7, 9, 40, 3)
    x, wt, b, _, _ = problem(shape, True)
    y = conv(native, x.to("tt"), wt, b, 3)
    assert layout(y) == "nhwc"
    y_cpu = y.cpu()
    assert y_cpu.shape == (2, 40, 7, 9)
    torch.testing.assert_close((y + y).cpu(), y_cpu + y_cpu)
    torch.testing.assert_close(F.silu(y).cpu(), F.silu(y_cpu), atol=2e-2, rtol=2e-2)
    w2 = sample((16, 40, 3, 3), 7, 1 / math.sqrt(40 * 9))
    before = runs()
    z = F.conv2d(y, w2.to("tt"), padding=1)
    assert runs() - before == 1
    ref = F.conv2d(y_cpu.double(), w2.double(), padding=1)
    cpu = F.conv2d(y_cpu.float(), w2.float(), padding=1).bfloat16()
    assert rel(z.cpu(), ref) <= 3 * rel(cpu, ref)


@pytest.mark.hw
def test_route_without_switch(native, monkeypatch):
    """KTT_NHWC unset: an Nchw input keeps conv2d()'s route (Nchw output); an Nhwc input
    still runs the channels-last conv."""
    monkeypatch.delenv("KTT_NHWC")
    shape = (2, 33, 7, 9, 40, 3)
    x, wt, b, ref, cpu_rel = problem(shape, True)
    before = runs()
    y = conv(native, x.to("tt"), wt, b, 3)
    assert runs() == before and layout(y) == "nchw"
    z = conv(native, as_nhwc(x), wt, b, 3)
    assert runs() == before + 1 and layout(z) == "nhwc"
    assert rel(y.cpu(), ref) <= 3 * cpu_rel and rel(z.cpu(), ref) <= 3 * cpu_rel


@pytest.mark.hw
def test_out_of_scope_keeps_route(native):
    """Strided, dilated, grouped and unpadded convs never take the channels-last conv."""
    x = sample((1, 32, 12, 10), 0)
    w = sample((32, 32, 3, 3), 1, 0.1)
    before = runs()
    for kwargs in (
        dict(stride=2, padding=1),
        dict(dilation=2, padding=2),
        dict(padding=0),
    ):
        got = F.conv2d(x.to("tt"), w.to("tt"), **kwargs).cpu()
        ref = F.conv2d(x.double(), w.double(), **kwargs)
        assert rel(got, ref) < 1e-2, kwargs
    wg = sample((32, 16, 3, 3), 2, 0.1)
    got = F.conv2d(x.to("tt"), wg.to("tt"), padding=1, groups=2).cpu()
    assert rel(got, F.conv2d(x.double(), wg.double(), padding=1, groups=2)) < 1e-2
    assert runs() == before
