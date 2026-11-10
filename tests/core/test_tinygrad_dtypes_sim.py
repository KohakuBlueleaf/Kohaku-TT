"""Dtypes on "KTT" (numpy model of the binding): float32, bfloat16 and int32 run on the
device; kernels on other dtypes the host evaluator models run there, counted and warned
once per dtype, or raise DtypeError under KTT_TINYGRAD_DTYPES=error; fp8 always raises.
"""

import warnings

import numpy as np
import pytest

pytest.importorskip("tinygrad.engine")

from kohakutt.tinygrad_backend import device as D
from kohakutt.tinygrad_backend import executor, install, numpy_exec, overrides
from kohakutt.tinygrad_backend.errors import DtypeError, DtypeWarning, HostKernelError

HOST_DTYPES = [
    "float16",
    "float64",
    "int8",
    "int16",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bool",
]
DEVICE = "float32, bfloat16, int32"


@pytest.fixture(autouse=True)
def sim(monkeypatch):
    monkeypatch.delenv("KTT_TINYGRAD_DTYPES", raising=False)
    install(executor="sim", overrides=True)
    executor.FALLBACKS.clear()
    yield
    overrides.enable(False)


@pytest.fixture(scope="module", autouse=True)
def close_device():
    yield
    D.get_backend().close()


def _tg():
    from tinygrad import Tensor, dtypes

    return Tensor, dtypes


def _values(name):
    return (np.arange(24).reshape(4, 6) % 5).astype(name)


def _doubled(name, device):
    Tensor, dtypes = _tg()
    return (Tensor(_values(name), device=device).cast(dtypes.float32) * 2).numpy()


@pytest.mark.parametrize("name", HOST_DTYPES)
def test_host_dtype_runs_on_the_host_counted_and_warned(name):
    with pytest.warns(DtypeWarning, match=rf"{name} is not a device dtype.*{DEVICE}"):
        got = _doubled(name, "KTT")
    np.testing.assert_array_equal(got, _doubled(name, "CPU"))
    assert executor.FALLBACKS[f"dtype {name}"] == 1, dict(executor.FALLBACKS)


@pytest.mark.parametrize("name", HOST_DTYPES)
def test_error_policy_raises_a_named_error(name, monkeypatch):
    monkeypatch.setenv("KTT_TINYGRAD_DTYPES", "error")
    with pytest.raises(DtypeError, match=rf"{name} is not a device dtype.*{DEVICE}"):
        _doubled(name, "KTT")


@pytest.mark.parametrize("policy", ["host", "error"])
def test_fp8_is_a_named_error(policy, monkeypatch):
    Tensor, dtypes = _tg()
    monkeypatch.setenv("KTT_TINYGRAD_DTYPES", policy)
    x = Tensor(np.linspace(-1, 1, 16, dtype=np.float32), device="KTT")
    with pytest.raises(DtypeError, match="float8_e4m3 is not supported on KTT"):
        x.cast(dtypes.fp8e4m3).cast(dtypes.float32).numpy()


def test_bad_policy_is_named(monkeypatch):
    monkeypatch.setenv("KTT_TINYGRAD_DTYPES", "strict")
    with pytest.raises(ValueError, match="KTT_TINYGRAD_DTYPES"):
        _doubled("int64", "KTT")


@pytest.mark.parametrize("name", ["float32", "bfloat16", "int32"])
def test_device_dtypes_run_on_the_device(name):
    Tensor, dtypes = _tg()
    a = (np.arange(24).reshape(4, 6) % 5).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DtypeWarning)
        x = Tensor(a, device="KTT").cast(getattr(dtypes, name)).contiguous()
        y = ((x + x) * 3).cast(dtypes.float32).numpy()
    np.testing.assert_array_equal(y, a * 6)
    assert not executor.FALLBACKS, dict(executor.FALLBACKS)


def _common_cases():
    Tensor, dtypes = _tg()
    f = np.linspace(-1, 1, 640, dtype=np.float32).reshape(64, 10)
    lab = (np.arange(64) % 10).astype(np.int32)
    mask = np.arange(640).reshape(64, 10) % 3 == 0

    def T(a, d):
        return Tensor(a, device=d)

    return {
        # (computation, where it runs on KTT)
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
        "rows indexed by int64": (
            lambda d: T(f, d)[T(lab.astype(np.int64), d)].numpy(),
            "int64",
        ),
    }


@pytest.mark.parametrize("case", list(_common_cases()))
def test_common_cases(case):
    fn, where = _common_cases()[case]
    ref = fn("CPU")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DtypeWarning)
        got = fn("KTT")
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-7)
    if where == "card":
        assert not executor.FALLBACKS, dict(executor.FALLBACKS)
    else:
        assert set(executor.FALLBACKS) == {f"dtype {where}"}, dict(executor.FALLBACKS)


def test_bf16_host_kernel_matches_cpu():
    Tensor, dtypes = _tg()
    x = Tensor(np.linspace(-3, 3, 64, dtype=np.float32).reshape(8, 8))
    x = x.cast(dtypes.bfloat16)

    def fn(t):
        return (t.cast(dtypes.int64) * 3).cast(dtypes.bfloat16).float().numpy()

    ref = fn(x)
    with pytest.warns(DtypeWarning, match="int64"):
        got = fn(x.to("KTT"))
    np.testing.assert_array_equal(got, ref)


def test_bf16_draws_on_the_host_route(monkeypatch):
    Tensor, dtypes = _tg()
    monkeypatch.setenv("KTT_TINYGRAD_RAND", "host")
    Tensor.manual_seed(9)
    ref = Tensor.rand(64, 64, dtype=dtypes.bfloat16).float().numpy()
    Tensor.manual_seed(9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DtypeWarning)
        got = Tensor.rand(64, 64, device="KTT", dtype=dtypes.bfloat16)
        got = got.float().numpy()
    np.testing.assert_array_equal(got, ref)


def test_host_evaluator_failure_is_named(monkeypatch):
    def broken(self):
        raise ValueError("operands could not be broadcast")

    monkeypatch.setattr(numpy_exec.Evaluator, "run", broken)
    named = r"did not run on the card \(dtype int64"
    with pytest.raises(HostKernelError, match=named), warnings.catch_warnings():
        warnings.simplefilter("ignore", DtypeWarning)
        _doubled("int64", "KTT")
