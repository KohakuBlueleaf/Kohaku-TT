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
    xi = Tensor.arange(50, dtype="int32").reshape(1, 2, 5, 5)
    ref = xi.max_pool2d(3, 2, padding=1).numpy()
    assert (xi.to("KTT").max_pool2d(3, 2, padding=1).numpy() == ref).all()


POOLS = {
    "max k3 s2 p1": lambda x: x.max_pool2d(3, 2, padding=1),
    "max k2 s1": lambda x: x.max_pool2d(2, 1),
    "max k2 s2 p1": lambda x: x.max_pool2d(2, 2, padding=1),
    "max k2 size 31": lambda x: x.max_pool2d(2),
    "max k3 s1 p1 d2": lambda x: x.max_pool2d(3, 1, dilation=2, padding=1),
    "max k3 s2 ceil": lambda x: x.max_pool2d(3, 2, ceil_mode=True),
    "max k3 s2 p1 relu ties": lambda x: x.relu().max_pool2d(3, 2, padding=1),
    "avg k3 s2 p1": lambda x: x.avg_pool2d(3, 2, padding=1),
    "avg k3 s2 p1 no pad count": lambda x: x.avg_pool2d(
        3, 2, padding=1, count_include_pad=False
    ),
    "avg k3 s2 ceil": lambda x: x.avg_pool2d(3, 2, ceil_mode=True),
    "avg k2": lambda x: x.avg_pool2d(2),
    "avg k2 size 31": lambda x: x.avg_pool2d(2),
}


@pytest.mark.parametrize("name", list(POOLS))
def test_pooling_gradient_matches_cpu_on_the_device(name):
    from kohakutt.tinygrad_backend import executor

    shape = (2, 16, 31, 31) if "31" in name else (2, 16, 32, 32)
    executor.FALLBACKS.clear()
    fwd, _, _ = workloads.run_case(POOLS[name], [shape])
    errs, kernels, calls = workloads.run_grad_case(POOLS[name], [shape])
    assert fwd < 1e-6 and errs[0] < 1e-6, (fwd, errs)
    assert not executor.FALLBACKS, dict(executor.FALLBACKS)
    if "k2" not in name or "s1" in name or "p1" in name or "31" in name:
        assert any(k.endswith("_pool2d_bw") for k in calls), calls


@pytest.mark.parametrize("name", ["max k3 s2 p1", "avg k3 s2 ceil"])
def test_pooling_on_the_numpy_executor(name):
    install(executor="numpy", overrides=True)
    try:
        errs, _, calls = workloads.run_grad_case(POOLS[name], [(2, 3, 9, 9)])
    finally:
        install(executor="sim", overrides=True)
    assert errs[0] < 1e-6 and any(k.endswith("_pool2d_bw") for k in calls)
