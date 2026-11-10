"""TinyJit on the KTT device (numpy model of the binding): jitted steps with fixed
shapes and with a Variable length match the unjitted result, replay without new
to_program calls, and use no arena sub-views."""

import numpy as np
import pytest

pytest.importorskip("tinygrad.engine")

from kohakutt.tinygrad_backend import compat as C
from kohakutt.tinygrad_backend import device as D
from kohakutt.tinygrad_backend import executor, install

Tensor, TinyJit, Variable = C.Tensor, C.TinyJit, C.Variable


@pytest.fixture(scope="module", autouse=True)
def sim_device():
    install(executor="sim", overrides=False)
    yield
    D.get_backend().close()


def test_fixed_shape_step_replays():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((64, 32)).astype(np.float32)
    b = rng.standard_normal(32).astype(np.float32)
    wt, bt = Tensor(w, device="KTT").realize(), Tensor(b, device="KTT").realize()

    @TinyJit
    def step(x):
        return (x @ wt + bt).relu().realize()

    seam_after = []
    calls = []
    for i in range(5):
        x = rng.standard_normal((16, 64)).astype(np.float32)
        executor.BRIDGE_CALLS.clear()
        D.reset_stats()
        out = step(Tensor(x, device="KTT").realize()).numpy()
        ref = np.maximum(x.astype(np.float64) @ w + b, 0)
        assert np.allclose(out, ref, atol=1e-4), i
        seam_after.append(D.SEAM_CALLS["to_program"])
        calls.append((sum(D.KERNELS.values()), sum(executor.BRIDGE_CALLS.values())))
    assert all(
        s == seam_after[-1] for s in seam_after[2:]
    ), seam_after  # no new tinygrad kernels after capture
    assert (
        calls[-1] == calls[-2] and calls[-1][0] >= 1
    ), calls  # every replay: same kernels, same bridge calls


def test_variable_length_step_replays_without_new_kernels():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((32, 32)).astype(np.float32)
    big = rng.standard_normal((128, 32)).astype(np.float32)
    wt, bigt = Tensor(w, device="KTT").realize(), Tensor(big, device="KTT").realize()
    v = Variable("T", 1, 128)

    @TinyJit
    def step(vT):
        return (bigt[:vT] @ wt).relu().sum(0).realize()

    seams = []
    for T in (5, 6, 7, 40, 100, 7):
        out = step(v.bind(T)).numpy()
        ref = np.maximum(big[:T].astype(np.float64) @ w, 0).sum(0)
        assert np.allclose(out, ref, atol=1e-4), T
        seams.append(D.SEAM_CALLS["to_program"])
    assert all(s == seams[-1] for s in seams[2:]), seams


def test_no_arena_sub_buffers():
    """install() sets NO_MEMORY_PLANNER: no device kernel sees a sub-view."""
    assert executor.FALLBACKS.get("sub-buffer view", 0) == 0
    assert executor.FALLBACKS.get("store into a sub-buffer", 0) == 0
