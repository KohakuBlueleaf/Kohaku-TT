"""Bridge entry points through the C ABI of _ktt_native.so: scalar max / min,
fused attention dtypes, arguments validated in the calling thread."""

import numpy as np
import pytest

pytest.importorskip("tinygrad.engine")
core = pytest.importorskip("kohakutt.tinygrad_backend.core")

pytestmark = [
    pytest.mark.hw,
    pytest.mark.skipif(not core.available(), reason="_ktt_native.so not built"),
]


@pytest.fixture(scope="module")
def device():
    core.open([0], 1, 1)
    yield
    core.close()


def _values(dtype):
    rng = np.random.default_rng(0)
    if dtype == core.INT32:
        return rng.integers(-20, 20, (64, 96)).astype(np.int32)
    return rng.standard_normal((64, 96)).astype(np.float32)


@pytest.mark.parametrize("dtype", [core.FP32, core.INT32])
@pytest.mark.parametrize("scalar", [0.25, -1.0, 3.0])
def test_scalar_max_min(device, dtype, scalar):
    x = _values(dtype)
    t = core.upload(x, dtype)
    for op, ref in (("maximum", np.maximum), ("minimum", np.minimum)):
        got = core.download(core.binary_scalar(op, t, scalar))
        want = ref(x, np.array(scalar, x.dtype))
        assert got.shape == want.shape
        np.testing.assert_allclose(got, want, atol=1e-6, rtol=0)


def test_scalar_max_min_match_clamp(device):
    x = _values(core.FP32)
    t = core.upload(x)
    np.testing.assert_array_equal(
        core.download(core.binary_scalar("maximum", t, 0.5)),
        core.download(core.clamp(t, lo=0.5)),
    )
    np.testing.assert_array_equal(
        core.download(core.binary_scalar("minimum", t, 0.5)),
        core.download(core.clamp(t, hi=0.5)),
    )


def test_sdpa_float32_raises_at_the_call(device):
    q = core.upload(np.zeros((1, 2, 32, 64), np.float32), core.FP32)
    with pytest.raises(
        core.CoreError, match=r"sdpa.*bfloat16.*float32.*\[1, 2, 32, 64\]"
    ):
        core.sdpa(q, q, q)
    # Nothing was queued: the device is usable and synchronize raises nothing.
    core.synchronize()


def _fp32(*shape):
    return core.upload(np.zeros(shape, np.float32))


# (op name, shapes) -> a call that must raise before anything is queued
BAD_CALLS = {
    "reshape": (
        lambda: core.reshape(_fp32(32, 32), [7, 7]),
        r"reshape.*\[32, 32\].*\[7, 7\]",
    ),
    "permute rank": (
        lambda: core.permute(_fp32(32, 32), [0]),
        r"permute.*\[0\].*\[32, 32\]",
    ),
    "permute repeat": (
        lambda: core.permute(_fp32(32, 32), [0, 0]),
        r"permute.*\[0, 0\]",
    ),
    "slice bounds": (
        lambda: core.slice(_fp32(32, 32), [0, 0], [64, 32]),
        r"slice.*\[32, 32\].*end 64",
    ),
    "slice rank": (lambda: core.slice(_fp32(32, 32), [0], [8]), r"slice.*\[32, 32\]"),
    "binary": (
        lambda: core.binary("add", _fp32(32, 32), _fp32(16, 32)),
        r"binary.*\[32, 32\].*\[16, 32\]",
    ),
    "concat": (
        lambda: core.concat([_fp32(32, 32), _fp32(16, 16)], 0),
        r"concat.*\[32, 32\].*\[16, 16\]",
    ),
    "concat dim": (
        lambda: core.concat([_fp32(32, 32), _fp32(32, 32)], 2),
        r"concat.*dim 2.*\[32, 32\]",
    ),
}


@pytest.mark.parametrize("case", list(BAD_CALLS))
def test_bad_arguments_raise_at_the_call(device, case):
    call, message = BAD_CALLS[case]
    with pytest.raises(core.CoreError, match=message):
        call()
    core.synchronize()


def test_worker_error_names_the_queueing_op(device):
    ints = core.upload(np.ones((32, 32), np.int32), core.INT32)
    core.unary("exp", ints)
    with pytest.raises(core.CoreError, match=r"\[bridge op 'unary'\].*TT_FATAL"):
        core.synchronize()
    core.synchronize()


def test_valid_calls_still_pass(device):
    x = np.arange(32 * 64, dtype=np.float32).reshape(32, 64)
    t = core.upload(x)
    np.testing.assert_array_equal(
        core.download(core.reshape(t, [64, 32])), x.reshape(64, 32)
    )
    np.testing.assert_array_equal(core.download(core.permute(t, [1, 0])), x.T)
    np.testing.assert_array_equal(
        core.download(core.slice(t, [0, 8], [32, 40], [1, 2])), x[:, 8:40:2]
    )
    np.testing.assert_array_equal(
        core.download(core.concat([t, t], 1)), np.concatenate([x, x], 1)
    )
    row = core.upload(x[:1])
    np.testing.assert_array_equal(core.download(core.binary("add", t, row)), x + x[:1])
