"""kohakutt.sdpa(key_lengths=) route decision: the kernel route only when the static
checks pass and the kernel's own fit check (L1, runtime arguments, with gradients its
backward) says yes; else the explicit-mask route (no deferred stream error)."""

from types import SimpleNamespace

import torch

from kohakutt import sdpa_kernel
from kohakutt.torch_backend.lengths import kernel_takes


def fake(shape, dtype=torch.bfloat16, device="tt", requires_grad=False):
    return SimpleNamespace(
        shape=shape,
        dtype=dtype,
        device=SimpleNamespace(type=device),
        requires_grad=requires_grad,
        dim=lambda: len(shape),
        size=lambda i: shape[i],
    )


def qkv(**kw):
    return fake((2, 8, 64, 64), **kw), fake((2, 2, 64, 64)), fake((2, 2, 64, 64))


def test_fit_check_decides():
    calls = []

    def fits(q, k, v, causal, grads):
        calls.append((causal, grads))
        return fits.answer

    fits.answer = False
    assert not kernel_takes(*qkv(), False, fits=fits)
    fits.answer = True
    assert kernel_takes(*qkv(), True, fits=fits)
    assert calls == [(False, False), (True, False)]
    # Causal with Sq != Sk: the kernel's check decides (its forward is top-left).
    assert kernel_takes(fake((2, 8, 32, 64)), *qkv()[1:], True, fits=fits)
    assert calls[-1] == (True, False)


def test_gradients_ask_for_the_backward():
    seen = []
    q, k, v = qkv(requires_grad=True)
    kernel_takes(q, k, v, False, fits=lambda *a: seen.append(a[-1]) or True)
    assert seen == [True]
    with torch.no_grad():
        kernel_takes(q, k, v, False, fits=lambda *a: seen.append(a[-1]) or True)
    assert seen == [True, False]


def test_static_refusals_skip_the_device_check():
    def fits(*a):
        raise AssertionError("asked the device")

    q, k, v = qkv()
    assert not kernel_takes(fake((2, 8, 64, 64), torch.float32), k, v, False, fits)
    assert not kernel_takes(fake((2, 8, 64, 64), device="cpu"), k, v, False, fits)
    assert not kernel_takes(fake((2, 7, 64, 64)), k, v, False, fits)  # 7 % 2 heads
    with sdpa_kernel("math"):
        assert not kernel_takes(q, k, v, False, fits)
