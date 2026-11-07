"""Regression: torch.tt device semantics on a 1x1 mesh."""

import pytest

from kohakutt.bench.device.device_api import checks


@pytest.mark.hw
@pytest.mark.parametrize("index", range(5))
def test_device_check(native, index):
    name, fn = checks()[index]
    assert fn(), name
