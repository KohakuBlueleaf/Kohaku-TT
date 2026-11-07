"""Regression: print, save/load, pickle, module.to and unsupported-op errors."""

import pytest

from kohakutt.bench.device.plumbing import checks


@pytest.mark.hw
@pytest.mark.parametrize("index", range(6))
def test_plumbing_check(native, index):
    name, fn = checks()[index]
    assert fn(), name
