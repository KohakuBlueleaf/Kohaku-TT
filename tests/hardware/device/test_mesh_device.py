"""Regression: a 1x2 mesh bound as tt:0 (replicated tensors) behaves like one device."""

import pytest
import torch

from kohakutt.bench.device.device_api import checks


@pytest.mark.hw
@pytest.mark.dual
def test_mesh_1x2_checks(native_module):
    kohakutt = native_module
    kohakutt.mesh([0, 1], shape=(1, 2))
    try:
        assert torch.tt.mesh_size() == 2
        failed = [name for name, fn in checks() if not fn()]
    finally:
        kohakutt.close()
    assert not failed
