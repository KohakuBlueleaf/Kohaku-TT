"""The bridge executor on a numpy model of the binding (no card): claimed kernels
match tinygrad's CPU backend; the others fall back and are counted by reason."""

import numpy as np
import pytest

pytest.importorskip("tinygrad.engine")

from kohakutt.tinygrad_backend import device as D
from kohakutt.tinygrad_backend import executor, install, workloads

CASES = workloads.WORKLOADS
# Workloads whose every kernel must run on the bridge (cumsum gathers more than
# GATHER_MAX elements and falls back).
ALL_DEVICE = {c[0] for c in CASES} - {"cumsum"}
# Workloads that must take no gather (memory ops).
NO_GATHER = {
    "conv2d k3 p1",
    "conv2d k3 s2",
    "permute 4d + reshape",
    "pad + slice",
    "cat",
    "sdpa causal",
}


@pytest.fixture(scope="module", autouse=True)
def sim_device():
    install(executor="sim", overrides=False)
    yield
    D.get_backend().close()


@pytest.mark.parametrize("name,fn,shapes", CASES, ids=[c[0] for c in CASES])
def test_workload(name, fn, shapes):
    executor.FALLBACKS.clear()
    executor.ROUTES.clear()
    err, kernels, calls = workloads.run_case(fn, shapes, name=name)
    assert err < 1e-4, (name, err)
    if name in NO_GATHER:
        assert not executor.ROUTES["reads_gather"], dict(executor.ROUTES)
    if name in ALL_DEVICE:
        assert not executor.FALLBACKS, dict(executor.FALLBACKS)
        assert D.STATS["device"] == kernels


def test_linear_is_one_matmul():
    executor.BRIDGE_CALLS.clear()
    workloads.run_case(
        workloads.WORKLOADS[0][1], workloads.WORKLOADS[0][2], name="linear+bias"
    )
    assert executor.BRIDGE_CALLS["mm"] == 1
    assert executor.BRIDGE_CALLS["upload"] == 3


def _tensor(a, device="KTT"):
    from tinygrad import Tensor

    return Tensor(a, device=device)


@pytest.mark.parametrize(
    "shape,axis", [((8, 256), -1), ((4, 6, 40), -1), ((40, 6), 0), ((5, 33), 1)]
)
def test_cumsum_is_the_bridge_cumsum(shape, axis):
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    executor.BRIDGE_CALLS.clear()
    executor.FALLBACKS.clear()
    got = _tensor(x).cumsum(axis).numpy()
    np.testing.assert_allclose(
        got, np.cumsum(x.astype(np.float64), axis), rtol=1e-4, atol=1e-4
    )
    assert executor.BRIDGE_CALLS["cumsum"] == 1, dict(executor.BRIDGE_CALLS)
    assert not executor.FALLBACKS


def test_embedding_is_embedding_rows_with_negative_indices():
    w = np.random.default_rng(1).standard_normal((16, 24)).astype(np.float32)
    ids = np.array([[3, -1, 15], [0, -16, 7]], np.int32)
    executor.BRIDGE_CALLS.clear()
    executor.FALLBACKS.clear()
    got = _tensor(w)[_tensor(ids)].numpy()
    np.testing.assert_array_equal(got, w[ids])
    assert executor.BRIDGE_CALLS["embedding_rows"] == 1, dict(executor.BRIDGE_CALLS)
    assert not executor.FALLBACKS


def test_embedding_index_out_of_range_falls_back_with_reason():
    w = np.random.default_rng(1).standard_normal((16, 24)).astype(np.float32)
    ids = np.array([3, 99], np.int32)
    executor.BRIDGE_CALLS.clear()
    executor.FALLBACKS.clear()
    got = _tensor(w)[_tensor(ids)].numpy()
    np.testing.assert_array_equal(got, _tensor(w, "CPU")[_tensor(ids, "CPU")].numpy())
    assert executor.BRIDGE_CALLS["embedding_rows"] == 0
    assert "embedding index outside the table" in executor.FALLBACKS


def test_buffers_stay_on_the_device_between_kernels():
    from tinygrad import Tensor

    D.reset_stats()
    executor.BRIDGE_CALLS.clear()
    x = Tensor(np.ones((64, 64), np.float32), device="KTT").realize()
    y = ((x + 1) * 2).realize()
    z = (y + y).realize()
    assert executor.BRIDGE_CALLS["upload"] == 1
    assert executor.BRIDGE_CALLS["download"] == 0
    assert np.allclose(z.numpy(), 8.0)
    assert executor.BRIDGE_CALLS["download"] == 1
