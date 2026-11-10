"""Both frontends in one process on one card: kohakutt opens the device, a torch op and
a tinygrad op on "KTT" run on it and match the CPU, kohakutt closes it once."""

import numpy as np
import pytest

pytest.importorskip("tinygrad.engine")

from kohakutt.device import session
from kohakutt.tinygrad_backend import core

pytestmark = [
    pytest.mark.hw,
    pytest.mark.skipif(not core.available(), reason="_ktt_native.so not built"),
]


def rel(x, ref):
    x, ref = np.asarray(x, np.float64), np.asarray(ref, np.float64)
    return float(np.max(np.abs(x - ref)) / (np.max(np.abs(ref)) + 1e-12))


def test_torch_and_tinygrad_share_one_open_device():
    import torch

    import kohakutt.tinygrad
    from tinygrad import Tensor

    dev = kohakutt.open(0)
    assert session._LIVE and core.is_open()

    rng = np.random.default_rng(0)
    a = rng.standard_normal((64, 128)).astype(np.float32)
    b = rng.standard_normal((128, 64)).astype(np.float32)
    ref = a.astype(np.float64) @ b.astype(np.float64)

    ta, tb = torch.from_numpy(a).to(dev), torch.from_numpy(b).to(dev)
    assert rel((ta @ tb).cpu().numpy(), ref) < 2e-2

    out = (Tensor(a, device="KTT") @ Tensor(b, device="KTT")).numpy()
    assert rel(out, ref) < 2e-2

    # Back to torch after tinygrad ran: still one live mesh, same device.
    assert rel((ta @ tb).cpu().numpy(), ref) < 2e-2
    assert kohakutt.open(0) == dev

    del ta, tb
    kohakutt.tinygrad.close()
    assert core.is_open(), "tinygrad must not close a device the torch side opened"
    kohakutt.close()
    assert not session._LIVE and not core.is_open()
