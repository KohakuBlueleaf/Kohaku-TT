"""Max and average pooling on "KTT" (card 0, own process, KTT_HW=1): overlapping,
padded, dilated and non-dividing windows run forward and backward with no host kernel
and match tinygrad's CPU backend; a convnet with 3x3 stride-2 pooling trains."""

import numpy as np
import pytest

pytest.importorskip("tinygrad.engine")

from kohakutt.tinygrad_backend import (
    core,
    executor,
    install,
    models,
    overrides,
    workloads,
)
from kohakutt.tinygrad_backend import device as D

pytestmark = [
    pytest.mark.hw,
    pytest.mark.skipif(not core.available(), reason="_ktt_native.so not built"),
]

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
    "avg k2": lambda x: x.avg_pool2d(2),
}


@pytest.fixture(scope="module", autouse=True)
def card():
    install(executor="bridge", overrides=True)
    yield
    overrides.enable(False)
    D.get_backend().close()


@pytest.mark.parametrize("name", list(POOLS))
def test_pooling_matches_cpu_with_no_host_kernel(name, record_property):
    shape = (2, 16, 31, 31) if "31" in name else (2, 16, 32, 32)
    executor.FALLBACKS.clear()
    fwd, _, _ = workloads.run_case(POOLS[name], [shape])
    errs, _, calls = workloads.run_grad_case(POOLS[name], [shape])
    record_property("forward_and_gradient_error", [fwd, errs[0]])
    assert not executor.FALLBACKS, dict(executor.FALLBACKS)
    assert fwd < 1e-6 and errs[0] < 1e-5, (fwd, errs)


@pytest.mark.parametrize("kind", ["max", "avg"])
def test_bf16_pooling_on_the_card(kind, record_property):
    from tinygrad import Tensor, dtypes

    a = workloads.inputs([(2, 8, 20, 20)])[0]
    pool = POOLS[f"{kind} k3 s2 p1"]
    executor.FALLBACKS.clear()
    out = {}
    for dev in ("CPU", "KTT"):
        x = Tensor(a, device=dev).cast(dtypes.bfloat16)
        x.requires_grad = True
        y = pool(x)
        y.float().square().sum().backward()
        out[dev] = (y.float().numpy(), x.grad.float().numpy())
    assert not executor.FALLBACKS, dict(executor.FALLBACKS)
    fwd = np.abs(out["KTT"][0] - out["CPU"][0]).max() / np.abs(out["CPU"][0]).max()
    grad = np.abs(out["KTT"][1] - out["CPU"][1]).max() / np.abs(out["CPU"][1]).max()
    record_property("forward_and_gradient_error", [fwd, grad])
    assert fwd < 4 * 2**-8 and grad < 8 * 2**-8, (fwd, grad)  # a few bf16 roundings


def test_convnet_with_strided_pooling_trains_on_the_card(record_property):
    m = models.cnn_pool()
    executor.FALLBACKS.clear()
    D.reset_stats()
    lk, _ = m.train("KTT", 3)
    record_property("kernels", dict(D.STATS))
    assert not executor.FALLBACKS, dict(executor.FALLBACKS)
    assert D.STATS["fallback"] == 0 and D.STATS["device"] > 0
    lc, _ = m.train("CPU", 3)
    record_property("losses_ktt_cpu", [lk, lc])
    assert abs(lk[0] - lc[0]) / lc[0] < 16 * 2.0**-8  # bf16-sized: the conv kernels
    assert lk[-1] < lk[0]
    assert np.isfinite(lk).all()
