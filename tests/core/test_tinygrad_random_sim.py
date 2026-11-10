"""Random numbers on "KTT" drawn by the device (numpy model of the binding): one bridge
draw per call, no host kernel, reproducible under Tensor.manual_seed, fresh values per
JIT replay; KTT_TINYGRAD_RAND=host gives the CPU backend's stream."""

import numpy as np
import pytest

pytest.importorskip("tinygrad.engine")

from kohakutt.tinygrad_backend import device as D
from kohakutt.tinygrad_backend import executor, install, overrides


@pytest.fixture(autouse=True)
def sim_overrides(monkeypatch):
    monkeypatch.delenv("KTT_TINYGRAD_RAND", raising=False)
    monkeypatch.delenv("KTT_PHILOX", raising=False)
    install(executor="sim", overrides=True)
    executor.FALLBACKS.clear()
    executor.BRIDGE_CALLS.clear()
    yield
    overrides.enable(False)


@pytest.fixture(scope="module", autouse=True)
def close_device():
    yield
    D.get_backend().close()


def _tg():
    from tinygrad import Context, Tensor, TinyJit, dtypes

    return Context, Tensor, TinyJit, dtypes


@pytest.mark.parametrize("kind", ["rand", "randn", "uniform", "rand bf16"])
def test_draws_run_on_the_device(kind):
    _, Tensor, _, dtypes = _tg()
    make = {
        "rand": lambda: Tensor.rand(64, 96, device="KTT"),
        "randn": lambda: Tensor.randn(64, 96, device="KTT"),
        "uniform": lambda: Tensor.uniform(64, 96, low=-2, high=3, device="KTT"),
        "rand bf16": lambda: Tensor.rand(64, 96, device="KTT", dtype=dtypes.bfloat16),
    }[kind]
    v = make().float().numpy()
    assert not executor.FALLBACKS, dict(executor.FALLBACKS)
    assert executor.BRIDGE_CALLS["uniform"] == 1, dict(executor.BRIDGE_CALLS)
    assert executor.BRIDGE_CALLS["upload"] == 0
    lo, hi = {"uniform": (-2, 3), "randn": (-8, 8)}.get(kind, (0, 1))
    assert v.shape == (64, 96) and lo <= v.min() and v.max() < hi
    assert len(np.unique(v)) > (500 if "bf16" in kind else 5000)


def test_manual_seed_reproduces_the_draws():
    _, Tensor, _, _ = _tg()
    Tensor.manual_seed(5)
    a, b = Tensor.rand(32, 32, device="KTT").numpy(), Tensor.randn(8, device="KTT")
    b = b.numpy()
    Tensor.manual_seed(5)
    a2, b2 = Tensor.rand(32, 32, device="KTT").numpy(), Tensor.randn(8, device="KTT")
    assert np.array_equal(a, a2) and np.array_equal(b, b2.numpy())
    assert not np.array_equal(a, Tensor.rand(32, 32, device="KTT").numpy())
    Tensor.manual_seed(6)
    assert not np.array_equal(a, Tensor.rand(32, 32, device="KTT").numpy())


def test_statistics():
    _, Tensor, _, _ = _tg()
    u = Tensor.rand(256, 256, device="KTT").numpy()
    z = Tensor.randn(256, 256, device="KTT").numpy()
    assert abs(u.mean() - 0.5) < 0.01 and abs(u.std() - 12**-0.5) < 0.01
    assert abs(z.mean()) < 0.02 and abs(z.std() - 1) < 0.02


def test_dropout_keeps_the_mask_on_the_device():
    Context, Tensor, _, _ = _tg()
    x = Tensor.ones(128, 128, device="KTT").realize()
    with Context(TRAINING=1):
        y = x.dropout(0.25).numpy()
    assert not executor.FALLBACKS, dict(executor.FALLBACKS)
    assert set(np.unique(y)) <= {0.0, np.float32(1 / 0.75)}
    assert abs((y == 0).mean() - 0.25) < 0.02
    assert (x.dropout(0.25).numpy() == 1).all()  # outside TRAINING: unchanged


def test_dropout_gradient_uses_the_same_mask():
    Context, Tensor, _, _ = _tg()
    x = Tensor(np.full((64, 64), 3.0, np.float32), device="KTT")
    x.requires_grad = True
    with Context(TRAINING=1):
        y = x.dropout(0.5)
        y.sum().backward()
    kept = y.numpy() != 0
    np.testing.assert_array_equal(x.grad.numpy() != 0, kept)


def test_jit_replays_draw_new_values():
    _, Tensor, TinyJit, _ = _tg()

    @TinyJit
    def noisy(x):
        return (x + Tensor.rand(16, 16, device="KTT")).realize()

    zero = Tensor.zeros(16, 16, device="KTT").contiguous().realize()
    outs = [noisy(zero).numpy() for _ in range(4)]
    assert all(not np.array_equal(a, b) for a, b in zip(outs, outs[1:]))


def test_host_route_is_the_cpu_stream(monkeypatch):
    _, Tensor, _, _ = _tg()
    monkeypatch.setenv("KTT_TINYGRAD_RAND", "host")
    Tensor.manual_seed(3)
    ref = Tensor.randn(33, 7, device="CPU").numpy()
    Tensor.manual_seed(3)
    got = Tensor.randn(33, 7, device="KTT").numpy()
    assert np.abs(got - ref).max() < 1e-6
    assert executor.BRIDGE_CALLS["uniform"] == 0
    assert executor.FALLBACKS["op BITCAST"] == 1  # tinygrad's generator: host kernels


def test_philox_route(monkeypatch):
    _, Tensor, _, _ = _tg()
    monkeypatch.setenv("KTT_PHILOX", "1")
    Tensor.manual_seed(1)
    a = Tensor.rand(40, 40, device="KTT").numpy()
    Tensor.manual_seed(1)
    assert np.array_equal(a, Tensor.rand(40, 40, device="KTT").numpy())
    assert executor.BRIDGE_CALLS["philox_uniform"] == 2
    assert executor.BRIDGE_CALLS["uniform"] == 0


def test_other_dtypes_and_devices_keep_tinygrad_route():
    _, Tensor, _, dtypes = _tg()
    Tensor.manual_seed(2)
    ref = Tensor.rand(9, device="CPU").numpy()
    Tensor.manual_seed(2)
    assert np.array_equal(Tensor.rand(9, device="CPU").numpy(), ref)
    Tensor.rand(4, 4, device="KTT", dtype=dtypes.float16).numpy()
    assert executor.BRIDGE_CALLS["uniform"] == 0


def test_bad_route_is_named(monkeypatch):
    _, Tensor, _, _ = _tg()
    monkeypatch.setenv("KTT_TINYGRAD_RAND", "gpu")
    with pytest.raises(ValueError, match="KTT_TINYGRAD_RAND"):
        Tensor.rand(4, device="KTT")


def test_numpy_executor_draws_too():
    _, Tensor, _, _ = _tg()
    install(executor="numpy", overrides=True)
    try:
        Tensor.manual_seed(4)
        a = Tensor.rand(16, 16, device="KTT").numpy()
        Tensor.manual_seed(4)
        assert np.array_equal(a, Tensor.rand(16, 16, device="KTT").numpy())
        assert a.min() >= 0 and a.max() < 1
    finally:
        install(executor="sim", overrides=True)
