"""The C ABI on card 0: upload, matmul, add, download, close."""

import numpy as np
import pytest

pytest.importorskip("tinygrad.engine")

from kohakutt.tinygrad_backend import core

pytestmark = [
    pytest.mark.hw,
    pytest.mark.skipif(not core.available(), reason="_ktt_native.so not built"),
]


@pytest.fixture(scope="module")
def device():
    core.open([0], 1, 1)
    yield
    core.close()


def rel(x, ref):
    return float(np.max(np.abs(x - ref)) / (np.max(np.abs(ref)) + 1e-12))


def test_upload_download_roundtrip(device):
    a = np.random.default_rng(0).standard_normal((64, 96)).astype(np.float32)
    np.testing.assert_array_equal(core.download(core.upload(a)), a)


def test_matmul_add(device):
    rng = np.random.default_rng(1)
    a = rng.standard_normal((64, 256)).astype(np.float32)
    w = rng.standard_normal((512, 256)).astype(np.float32)
    bias = rng.standard_normal((64, 512)).astype(np.float32)
    c = core.mm(core.upload(a), core.upload(w), transpose_b=True)
    out = core.download(core.binary("add", c, core.upload(bias)))
    ref = a.astype(np.float64) @ w.T.astype(np.float64) + bias
    assert out.shape == ref.shape
    # fp32 matmul at the default fidelity
    assert rel(out, ref) < 2e-2


def test_bad_reshape_raises_at_the_call(device):
    with pytest.raises(core.CoreError):
        core.reshape(core.upload(np.zeros((32, 32), np.float32)), [7, 7])
