"""Dtypes on "KTT" with card 0 (own process, KTT_HW=1): which common computations stay
on the card, which run on the host (counted, warned), and the error policy."""

import warnings

import numpy as np
import pytest

pytest.importorskip("tinygrad.engine")

from kohakutt.tinygrad_backend import core, executor, install, overrides
from kohakutt.tinygrad_backend import device as D
from kohakutt.tinygrad_backend.errors import DtypeError, DtypeWarning

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
    monkeypatch.delenv("KTT_TINYGRAD_DTYPES", raising=False)
    executor.FALLBACKS.clear()


def _cases():
    from tinygrad import Tensor, dtypes

    f = np.linspace(-1, 1, 640, dtype=np.float32).reshape(64, 10)
    lab = (np.arange(64) % 10).astype(np.int32)
    mask = np.arange(640).reshape(64, 10) % 3 == 0

    def T(a, d):
        return Tensor(a, device=d)

    return {
        "f32 comparison inside where": (
            lambda d: (T(f, d) > 0).where(T(f, d), 0).numpy(),
            "card",
        ),
        "accuracy, int32 labels": (
            lambda d: (T(f, d).argmax(1) == T(lab, d)).mean().numpy(),
            "card",
        ),
        "cross-entropy, int32 labels": (
            lambda d: T(f, d).sparse_categorical_crossentropy(T(lab, d)).numpy(),
            "card",
        ),
        "tril mask as bool, inside where": (
            lambda d: Tensor.ones(10, 10, device=d)
            .tril()
            .cast(dtypes.bool)
            .where(1.0, 0.0)
            .numpy(),
            "card",
        ),
        "rows indexed by int32": (lambda d: T(f, d)[T(lab, d)].numpy(), "card"),
        "f32 comparison stored as bool": (lambda d: (T(f, d) > 0).numpy(), "bool"),
        "numpy bool mask in where": (
            lambda d: T(mask, d).where(T(f, d), 0).numpy(),
            "bool",
        ),
        "cross-entropy, int64 labels": (
            lambda d: T(f, d)
            .sparse_categorical_crossentropy(T(lab.astype(np.int64), d))
            .numpy(),
            "int64",
        ),
        "float16 matmul": (
            lambda d: (T(f, d).half().T @ T(f, d).half()).float().numpy(),
            "float16",
        ),
    }


@pytest.mark.parametrize("case", list(_cases()))
def test_where_common_computations_run(case, record_property):
    fn, where = _cases()[case]
    ref = fn("CPU")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", DtypeWarning)
        got = fn("KTT")
    got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    err = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30))
    record_property("relative_error", err)
    # matmuls run at the card's default fidelity; float16 sums on the host in float16
    assert err < {"float16": 2e-3}.get(where, 1e-2 if "entropy" in case else 1e-6)
    if where == "card":
        assert not executor.FALLBACKS and not seen, dict(executor.FALLBACKS)
    else:
        assert set(executor.FALLBACKS) == {f"dtype {where}"}
        assert any(where in str(w.message) for w in seen)


def test_error_policy_on_the_card(monkeypatch):
    from tinygrad import Tensor

    monkeypatch.setenv("KTT_TINYGRAD_DTYPES", "error")
    a = np.arange(12, dtype=np.int64).reshape(3, 4)
    x = Tensor(a, device="KTT")
    np.testing.assert_array_equal(x.numpy(), a)  # storage and reads need no kernel
    with pytest.raises(DtypeError, match="int64 is not a device dtype"):
        (x + 1).numpy()
