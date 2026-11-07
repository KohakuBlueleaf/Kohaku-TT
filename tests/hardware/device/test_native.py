"""Regression: the native tt device keeps ops on device and returns finite results."""

import pytest
import torch

from kohakutt.bench.device.native_mlp import run_native
from kohakutt.bench.device.native_spike import run


@pytest.mark.hw
def test_add_mm_stay_on_device(native):
    record = run(native, 64)
    assert record["device"] == "tt:0"
    assert record["ops"] == {**record["ops"], "uploads": 0, "downloads": 0, "ops": 2}
    assert record["cpu"]["downloads"] == 2
    assert record["add"]["nonfinite"] == 0 and record["mm"]["nonfinite"] == 0


@pytest.mark.hw
def test_mlp_forward_on_device(native):
    record = run_native(native, 32, 256, 512)
    counters = record["forward_counters"]
    assert (counters["uploads"], counters["downloads"], counters["ops"]) == (0, 0, 3)
    assert record["stats"]["nonfinite"] == 0
    assert torch.device(record["device"]).type == "tt"
