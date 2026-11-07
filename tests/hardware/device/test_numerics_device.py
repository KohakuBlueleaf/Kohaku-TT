"""Regression: numerics scopes reach the device ops, read when an op is queued."""

import pytest
import torch


def _matmul(dev):
    gen = torch.Generator().manual_seed(0)
    a = torch.randn(256, 512, generator=gen).bfloat16().to(dev)
    b = torch.randn(512, 256, generator=gen).bfloat16().to(dev)
    return a, b


@pytest.mark.hw
def test_native_scope_mirrors_python(native):
    with native.numerics(fidelity="HiFi3", approx=True, attention="centered"):
        assert torch.ops.ktt.numerics_current() == [2, 1, 0, 1]
    assert torch.ops.ktt.numerics_current() == [1, 0, 0, 0]


@pytest.mark.hw
def test_fidelity_reaches_matmul(native):
    a, b = _matmul(native.device(0))
    with native.numerics(fidelity="LoFi"):
        lofi = (a @ b).cpu()
    with native.numerics(fidelity="HiFi3"):
        hifi3 = (a @ b).cpu()
    assert not torch.equal(lofi, hifi3)


@pytest.mark.hw
def test_scope_read_at_queue_time(native):
    a, b = _matmul(native.device(0))
    with native.numerics(fidelity="LoFi"):
        queued = a @ b  # queued under LoFi, read after the scope closed
    native.synchronize()
    with native.numerics(fidelity="LoFi"):
        inside = (a @ b).cpu()
    assert torch.equal(queued.cpu(), inside)
