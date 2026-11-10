"""Random numbers on "KTT" drawn by card 0 (own process, KTT_HW=1): no host kernel,
statistics of U(0, 1) / N(0, 1), reproducible under Tensor.manual_seed, fresh values
per JIT replay; KTT_TINYGRAD_RAND=host gives the CPU backend's stream."""

import numpy as np
import pytest

pytest.importorskip("tinygrad.engine")

from kohakutt.tinygrad_backend import core, executor, install, overrides
from kohakutt.tinygrad_backend import device as D

pytestmark = [
    pytest.mark.hw,
    pytest.mark.skipif(not core.available(), reason="_ktt_native.so not built"),
]


@pytest.fixture(scope="module", autouse=True)
def card():
    install(executor="bridge", overrides=True)
    yield
    overrides.enable(False)
    D.get_backend().close()


@pytest.fixture(autouse=True)
def counters(monkeypatch):
    monkeypatch.delenv("KTT_TINYGRAD_RAND", raising=False)
    monkeypatch.delenv("KTT_PHILOX", raising=False)
    executor.FALLBACKS.clear()
    executor.BRIDGE_CALLS.clear()


def _tg():
    from tinygrad import Context, Tensor, TinyJit, dtypes

    return Context, Tensor, TinyJit, dtypes


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_uniform_statistics(dtype, record_property):
    _, Tensor, _, _ = _tg()
    u = Tensor.rand(1024, 1024, device="KTT", dtype=dtype).float().numpy()
    assert not executor.FALLBACKS, dict(executor.FALLBACKS)
    assert executor.BRIDGE_CALLS["uniform"] == 1
    record_property("mean_std_min_max", [u.mean(), u.std(), u.min(), u.max()])
    assert u.min() >= 0 and u.max() < 1
    # bf16 keeps 8 bits: values sit up to 2^-8 below the fp32 draw (CPU: mean 0.4961)
    tol = 2e-3 if dtype == "float32" else 5e-3
    assert abs(u.mean() - 0.5) < tol and abs(u.std() - 12**-0.5) < tol
    hist = np.histogram(u, bins=16, range=(0, 1))[0] / u.size
    assert np.abs(hist - 1 / 16).max() < tol


def test_normal_statistics(record_property):
    _, Tensor, _, _ = _tg()
    z = Tensor.randn(1024, 1024, device="KTT").numpy().astype(np.float64)
    assert not executor.FALLBACKS, dict(executor.FALLBACKS)
    record_property("mean_std_min_max", [z.mean(), z.std(), z.min(), z.max()])
    assert abs(z.mean()) < 5e-3 and abs(z.std() - 1) < 5e-3
    assert 4 < np.abs(z).max() < 7
    assert abs((np.abs(z) < 1).mean() - 0.6827) < 3e-3


def test_dropout_share_and_gradient():
    Context, Tensor, _, _ = _tg()
    x = Tensor.ones(512, 512, device="KTT").contiguous().realize()
    x.requires_grad = True
    with Context(TRAINING=1):
        y = x.dropout(0.3)
        y.sum().backward()
    v, g = y.numpy(), x.grad.numpy()
    assert not executor.FALLBACKS, dict(executor.FALLBACKS)
    assert abs((v == 0).mean() - 0.3) < 5e-3
    np.testing.assert_array_equal(g, v)  # d(sum)/dx = mask / (1 - p) = y for x = 1


def test_manual_seed_reproduces_and_draws_differ():
    _, Tensor, _, _ = _tg()
    Tensor.manual_seed(11)
    a, b = Tensor.rand(256, 256, device="KTT").numpy(), Tensor.rand(256, device="KTT")
    b = b.numpy()
    Tensor.manual_seed(11)
    assert np.array_equal(a, Tensor.rand(256, 256, device="KTT").numpy())
    assert np.array_equal(b, Tensor.rand(256, device="KTT").numpy())
    assert np.corrcoef(a[0], a[1])[0, 1] < 0.2


def test_jit_replays_draw_new_values():
    _, Tensor, TinyJit, _ = _tg()

    @TinyJit
    def noisy(x):
        return (x + Tensor.rand(64, 64, device="KTT")).realize()

    zero = Tensor.zeros(64, 64, device="KTT").contiguous().realize()
    outs = [noisy(zero).numpy() for _ in range(4)]
    assert all(not np.array_equal(p, q) for p, q in zip(outs, outs[1:]))
    assert not executor.FALLBACKS, dict(executor.FALLBACKS)


def test_host_route_is_the_cpu_stream(monkeypatch):
    _, Tensor, _, _ = _tg()
    monkeypatch.setenv("KTT_TINYGRAD_RAND", "host")
    Tensor.manual_seed(3)
    ref = Tensor.randn(64, 33, device="CPU").numpy()
    Tensor.manual_seed(3)
    got = Tensor.randn(64, 33, device="KTT").numpy()
    assert np.abs(got - ref).max() < 1e-5
    assert executor.BRIDGE_CALLS["uniform"] == 0


def test_philox_route_reproduces(monkeypatch):
    _, Tensor, _, _ = _tg()
    monkeypatch.setenv("KTT_PHILOX", "1")
    Tensor.manual_seed(1)
    a = Tensor.rand(64, 64, device="KTT").numpy()
    Tensor.manual_seed(1)
    assert np.array_equal(a, Tensor.rand(64, 64, device="KTT").numpy())
    assert executor.BRIDGE_CALLS["philox_uniform"] == 2
    assert a.min() >= 0 and a.max() < 1 and abs(a.mean() - 0.5) < 0.02
