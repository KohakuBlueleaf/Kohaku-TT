"""Library-op overrides (attention, norms, conv) on the numpy model of the binding:
one kernel per op matching tinygrad's CPU decomposition, gradients through the
backward kernels, and the off switch."""

import pytest

pytest.importorskip("tinygrad.engine")

from kohakutt.tinygrad_backend import device as D
from kohakutt.tinygrad_backend import install, overrides, workloads

# The numpy model rounds to bf16 where the card does, so the bound is bf16-sized.
BF16_EPS = (
    2.0**-8
)  # bf16 keeps 8 significant bits: the relative rounding step the model applies
BF16_OPS = (
    "ov sdpa",
    "ov sdpa causal",
    "ov conv2d k3 p1",
    "ov conv2d k3 s2",
    "ov transformer block fwd",
)
BOUND = {
    name: 8 * BF16_EPS for name in BF16_OPS
}  # a few bf16 roundings along a dot product / softmax


@pytest.fixture(autouse=True)
def sim_overrides():
    install(executor="sim", overrides=True)
    yield
    overrides.enable(False)


@pytest.fixture(scope="module", autouse=True)
def close_device():
    yield
    D.get_backend().close()


@pytest.mark.parametrize(
    "name,fn,shapes",
    workloads.OVERRIDE_WORKLOADS,
    ids=[c[0] for c in workloads.OVERRIDE_WORKLOADS],
)
def test_override_workload(name, fn, shapes):
    err, kernels, calls = workloads.run_case(fn, shapes, name=name)
    assert err < BOUND.get(name, 1e-4), (name, err)
    assert any(k.startswith("lib:") for k in calls), calls


def test_overrides_are_fewer_kernels():
    name, fn, shapes = workloads.OVERRIDE_WORKLOADS[2]
    _, with_ov, _ = workloads.run_case(fn, shapes, name=name)
    overrides.enable(False)
    _, without, _ = workloads.run_case(fn, shapes, name=name)
    assert with_ov == 1 and without == 4


@pytest.mark.parametrize(
    "name,fn,shapes",
    workloads.GRAD_WORKLOADS,
    ids=[c[0] for c in workloads.GRAD_WORKLOADS],
)
def test_gradients(name, fn, shapes):
    errs, kernels, calls = workloads.run_grad_case(fn, shapes)
    bound = 16 * BF16_EPS if any(s in name for s in ("sdpa", "conv", "block")) else 1e-4
    assert max(errs) < bound, (name, errs)
    assert any(k.startswith("lib:") for k in calls), calls


@pytest.mark.parametrize("kernel", [2, (2, 4)])
def test_max_pool_gradient_has_no_host_kernel(kernel):
    from kohakutt.tinygrad_backend import executor

    executor.FALLBACKS.clear()
    errs, kernels, calls = workloads.run_grad_case(
        lambda x: x.max_pool2d(kernel).relu(), [(2, 16, 32, 32)]
    )
    assert errs == [0.0], errs
    assert not executor.FALLBACKS, dict(executor.FALLBACKS)


def test_max_pool_override_keeps_other_forms():
    from kohakutt.tinygrad_backend.compat import Tensor

    x = Tensor.arange(50, dtype="float32").reshape(1, 2, 5, 5)
    for kw in ({"stride": 1}, {"padding": 1}, {}):
        ref = x.max_pool2d(**kw).numpy()
        got = x.to("KTT").max_pool2d(**kw).numpy()
        assert (got == ref).all(), kw
