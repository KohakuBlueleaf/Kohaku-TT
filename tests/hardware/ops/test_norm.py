"""Kohaku-TT's layer norm / RMS norm kernels (ktt_norm): relative RMS error of y, mean,
rstd, dx, dgamma and dbeta against CPU float64, no worse than CPU at the same dtype,
over ragged widths and rows, offsets, both dtypes, with and without affine
parameters."""

import pytest
import torch
import torch.nn.functional as F

DTYPES = [torch.bfloat16, torch.float32]
SHAPES = [
    (1, 33),
    (5, 70),
    (32, 64),
    (77, 1000),
    (2, 40, 4096),
    (3, 16384),
    (4096, 768),
    (65536, 64),
    (300, 4097),
    (2, 3, 50, 1280),
]
# Relative RMS of tt may exceed CPU's by this factor (same final rounding; the fp32
# accumulation order differs).
SLACK = 1.02


def _rel(a, ref):
    a, ref = a.double().flatten(), ref.double().flatten()
    return ((a - ref).norm() / ref.norm().clamp_min(1e-300)).item()


def _check(name, tt, cpu, ref, dtype):
    assert torch.isfinite(tt).all(), name
    t, c = _rel(tt, ref), _rel(cpu, ref)
    # fp32: CPU's error is a few ulps; allow the same order (1e-6 relative floor).
    floor = 2e-6 if dtype == torch.float32 else 0.0
    assert t <= SLACK * c + floor, (name, t, c)


def _data(shape, dtype, seed, shift=0.0):
    g = torch.Generator().manual_seed(seed)
    w = shape[-1]
    x = (torch.randn(*shape, generator=g) * 2 + shift).to(dtype)
    dy = torch.randn(*shape, generator=g).to(dtype)
    weight = (torch.randn(w, generator=g) * 0.5 + 1).to(dtype)
    bias = (torch.randn(w, generator=g) * 0.1).to(dtype)
    return x, dy, weight, bias


def _layer_norm(dev, dt, x, dy, weight, bias, affine):
    xx = x.to(dev, dt).detach().requires_grad_()
    params = []
    if affine:
        params = [
            weight.to(dev, dt).detach().requires_grad_(),
            bias.to(dev, dt).detach().requires_grad_(),
        ]
    w, b = params if affine else (None, None)
    y = F.layer_norm(xx, (x.shape[-1],), w, b, 1e-5)
    y.backward(dy.to(dev, dt))
    return [y.detach().cpu(), xx.grad.cpu()] + [p.grad.cpu() for p in params]


@pytest.mark.hw
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("affine", [True, False])
def test_layer_norm_forward_backward(native, dtype, shape, affine):
    x, dy, weight, bias = _data(shape, dtype, sum(shape))
    ref = _layer_norm("cpu", torch.float64, x, dy, weight, bias, affine)
    cpu = _layer_norm("cpu", dtype, x, dy, weight, bias, affine)
    tt = _layer_norm("tt", dtype, x, dy, weight, bias, affine)
    for name, a, c, r in zip(["y", "dx", "dweight", "dbias"], tt, cpu, ref):
        _check(name, a, c, r, dtype)


@pytest.mark.hw
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(77, 1000), (300, 4097)])
@pytest.mark.parametrize("shift", [100.0, 3000.0])
def test_layer_norm_offset(native, dtype, shape, shift):
    """Rows with a large common offset (|mean| / std 50 and 1500): y and dx."""
    x, dy, weight, bias = _data(shape, dtype, 7, shift=shift)
    ref = _layer_norm("cpu", torch.float64, x, dy, weight, bias, True)
    cpu = _layer_norm("cpu", dtype, x, dy, weight, bias, True)
    tt = _layer_norm("tt", dtype, x, dy, weight, bias, True)
    for name, a, c, r in zip(["y", "dx", "dweight", "dbias"], tt, cpu, ref):
        _check(name, a, c, r, dtype)


@pytest.mark.hw
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(3, 16384), (300, 4097)])
def test_layer_norm_statistics_fp32(native, dtype, shape):
    x, _, weight, bias = _data(shape, dtype, 7, shift=100.0)
    w = shape[-1]
    _, mean_ref, rstd_ref = torch.native_layer_norm(
        x.double(), (w,), weight.double(), bias.double(), 1e-5
    )
    y, mean, rstd = torch.native_layer_norm(
        x.to("tt"), (w,), weight.to("tt"), bias.to("tt"), 1e-5
    )
    assert mean.dtype == torch.float32 and rstd.dtype == torch.float32
    # fp32 statistics of the rounded input: within a few fp32 ulps of the float64
    # values.
    assert ((mean.cpu().double() - mean_ref).abs() / mean_ref.abs()).max() < 1e-6
    assert ((rstd.cpu().double() - rstd_ref) / rstd_ref).abs().max() < 1e-6
    y_ref = torch.native_layer_norm(
        x.double(), (w,), weight.double(), bias.double(), 1e-5
    )[0]
    y_cpu = torch.native_layer_norm(x, (w,), weight, bias, 1e-5)[0]
    _check("y", y.cpu(), y_cpu, y_ref, dtype)


@pytest.mark.hw
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(5, 70), (300, 4097)])
@pytest.mark.parametrize("affine", [True, False])
def test_rms_norm_forward(native, dtype, shape, affine):
    x, _, weight, _ = _data(shape, dtype, 3)
    w = shape[-1]
    wd = weight.double() if affine else None
    ref = F.rms_norm(x.double(), (w,), wd, 1e-5)
    cpu = F.rms_norm(x, (w,), weight if affine else None, 1e-5)
    with torch.no_grad():
        tt = F.rms_norm(
            x.to("tt"), (w,), weight.to("tt") if affine else None, 1e-5
        ).cpu()
    _check("y", tt, cpu, ref, dtype)


def _rms_norm(dev, dt, x, dy, weight, affine):
    xx = x.to(dev, dt).detach().requires_grad_()
    params = [weight.to(dev, dt).detach().requires_grad_()] if affine else []
    y = F.rms_norm(xx, (x.shape[-1],), params[0] if affine else None, 1e-5)
    y.backward(dy.to(dev, dt))
    return [y.detach().cpu(), xx.grad.cpu()] + [p.grad.cpu() for p in params]


@pytest.mark.hw
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(1, 33), (77, 1000), (2, 40, 4096), (300, 4097)])
@pytest.mark.parametrize("affine", [True, False])
def test_rms_norm_forward_backward(native, dtype, shape, affine):
    x, dy, weight, _ = _data(shape, dtype, sum(shape) + 1)
    ref = _rms_norm("cpu", torch.float64, x, dy, weight, affine)
    cpu = _rms_norm("cpu", dtype, x, dy, weight, affine)
    tt = _rms_norm("tt", dtype, x, dy, weight, affine)
    for name, a, c, r in zip(["y", "dx", "dweight"], tt, cpu, ref):
        _check(name, a, c, r, dtype)


@pytest.mark.hw
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(5, 70), (300, 4097)])
def test_fused_rms_norm_with_bias(native, dtype, shape):
    x, _, weight, bias = _data(shape, dtype, 9)
    w = shape[-1]
    ref = F.rms_norm(x.double(), (w,), weight.double(), 1e-5) + bias.double()
    cpu = torch.ops.ktt.rms_norm(x, weight, bias, 1e-5)
    tt = torch.ops.ktt.rms_norm(x.to("tt"), weight.to("tt"), bias.to("tt"), 1e-5).cpu()
    _check("y", tt, cpu, ref, dtype)


@pytest.mark.hw
@pytest.mark.parametrize("shape", [(64, 256), (4, 77, 256)])
def test_layer_norm_two_dims_moreh(native, shape):
    """Normalisation over the last two dims (TT-NN's moreh layer norm, bf16 statistics):
    mean and rstd within 5 %, dx within 20x CPU bf16's error."""
    x, dy, _, _ = _data(shape, torch.bfloat16, 11, shift=1.0)
    norm = tuple(shape[-2:])
    xd = x.double().requires_grad_()
    _, mean_ref, rstd_ref = torch.native_layer_norm(xd.detach(), norm, None, None, 1e-5)
    F.layer_norm(xd, norm, None, None, 1e-5).backward(dy.double())
    xc = x.clone().requires_grad_()
    F.layer_norm(xc, norm, None, None, 1e-5).backward(dy)
    xt = x.to("tt").requires_grad_()
    F.layer_norm(xt, norm, None, None, 1e-5).backward(dy.to("tt"))
    _, mean, rstd = torch.native_layer_norm(x.to("tt"), norm, None, None, 1e-5)
    assert mean.shape == mean_ref.shape and rstd.shape == rstd_ref.shape
    assert _rel(mean.cpu(), mean_ref) < 5e-2 and _rel(rstd.cpu(), rstd_ref) < 5e-2
    assert _rel(xt.grad.cpu(), xd.grad) <= 20 * _rel(xc.grad, xd.grad)


@pytest.mark.hw
def test_repeat_calls_are_bit_identical(native):
    x, _, weight, bias = _data((300, 4097), torch.bfloat16, 5)
    xt, wt, bt = x.to("tt"), weight.to("tt"), bias.to("tt")
    first = F.layer_norm(xt, (4097,), wt, bt).cpu()
    for _ in range(3):
        assert torch.equal(F.layer_norm(xt, (4097,), wt, bt).cpu(), first)


@pytest.mark.hw
def test_empty_rows(native):
    x = torch.zeros(0, 64)
    with torch.no_grad():
        assert F.rms_norm(x.to("tt"), (64,)).shape == (0, 64)
        assert F.layer_norm(x.to("tt"), (64,)).shape == (0, 64)
