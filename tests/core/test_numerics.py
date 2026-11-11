"""Numerics scope: defaults, nesting, threads, requirements, module scopes."""

import threading

import pytest
import torch

from kohakutt import Numerics, current_numerics, numerics, numerics_for, sdpa_kernel
from kohakutt.core.numerics import Requirements, resolve


def test_defaults():
    assert current_numerics() == Numerics("HiFi2", False, False, "exact")


def test_scope_nests_and_restores():
    base = current_numerics()
    with numerics(fidelity="HiFi3") as outer:
        assert current_numerics().fidelity == "HiFi3"
        with numerics(approx=True):
            assert current_numerics() == Numerics("HiFi3", True, False)
        assert current_numerics() == outer
    assert current_numerics() == base


def test_decorator():
    @numerics(fidelity="LoFi")
    def inner():
        return current_numerics().fidelity

    assert inner() == "LoFi"
    assert current_numerics().fidelity == "HiFi2"


def test_scope_is_per_thread():
    seen = []
    with numerics(fidelity="HiFi4"):
        worker = threading.Thread(target=lambda: seen.append(current_numerics()))
        worker.start()
        worker.join()
    assert seen == [Numerics()]


def test_rejects_unknown_fidelity():
    with pytest.raises(ValueError):
        Numerics(fidelity="HiFi5")
    with pytest.raises(ValueError):
        numerics(fidelity="HiFi5")
    with pytest.raises(ValueError):
        numerics(attention="int4")


def test_requirements():
    floor, ceiling = Requirements(min_fidelity="HiFi3"), Requirements(
        max_fidelity="LoFi"
    )
    assert resolve(Numerics("HiFi2"), floor).fidelity == "HiFi3"
    assert resolve(Numerics("HiFi4"), ceiling).fidelity == "LoFi"
    assert not resolve(Numerics(approx=True), Requirements(exact=True)).approx
    assert resolve(Numerics("HiFi2", strict=True), floor).fidelity == "HiFi2"


def test_sdpa_kernel_scope():
    assert current_numerics().attention_kernels == ("flash", "own", "ttnn", "math")
    with sdpa_kernel("math", "flash"):
        assert current_numerics().attention_kernels == ("flash", "math")
        with numerics(fidelity="HiFi3"):
            assert current_numerics().attention_kernels == ("flash", "math")
        with sdpa_kernel(["ttnn"]):
            assert current_numerics().attention_kernels == ("ttnn",)
        assert current_numerics().attention_kernels == ("flash", "math")
    assert current_numerics() == Numerics()


def test_sdpa_kernel_rejects_unknown_kernels():
    for kernels in ([], ["flash", "cudnn"]):
        with pytest.raises(ValueError):
            sdpa_kernel(*kernels)


def test_numerics_for_module():
    seen = []

    class Probe(torch.nn.Module):
        def forward(self, x):
            seen.append(current_numerics().fidelity)
            return x

    module = Probe()
    remove = numerics_for(module, fidelity="HiFi3")
    module(torch.zeros(1))
    remove()
    module(torch.zeros(1))
    assert seen == ["HiFi3", "HiFi2"]
