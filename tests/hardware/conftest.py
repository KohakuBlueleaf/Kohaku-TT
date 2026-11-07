"""Hardware test fixtures: the native tt backend (opened per test module) and the
exit-without-close run."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

LIBRARY = Path(__file__).parents[2] / "src/kohakutt_native/_ktt_native.so"

# The chip single-device tests open (KTT_TEST_CHIP, default 0).
CHIP = int(os.environ.get("KTT_TEST_CHIP", "0"))

# A process exiting with the device open, tt tensors, cached constants and a trace.
EXIT_SCRIPT = f"""
import torch
import torch.nn.functional as F
import kohakutt
from kohakutt import Captured

kohakutt.open({CHIP})
x = torch.randn(2, 3, 8, 8).to("tt")
y = F.avg_pool2d(x, 3, 2, 1) + torch.arange(4, device="tt").float().sum()
step = Captured(lambda t: t * 2 + 1, (x,), 1)
z = step(x)
print("done", tuple(y.shape), tuple(z.shape))
"""

_EXIT_RUN = {}


def pytest_collection_modifyitems(config, items):
    """Runs EXIT_SCRIPT in a subprocess before any test opens a device (this process
    holds the chip lock from its first open on), when test_exit is selected."""
    selected = any("test_exit" in item.nodeid for item in items)
    if selected and LIBRARY.exists() and os.environ.get("KTT_HW") == "1":
        _EXIT_RUN["run"] = subprocess.run(
            [sys.executable, "-c", EXIT_SCRIPT],
            capture_output=True,
            text=True,
            timeout=600,
        )


@pytest.fixture
def exit_run():
    """EXIT_SCRIPT's CompletedProcess; skips when it did not run."""
    if "run" not in _EXIT_RUN:
        pytest.skip("native build missing")
    return _EXIT_RUN["run"]


@pytest.fixture(scope="module")
def native_module():
    """The kohakutt module, device not opened; skips without a native build."""
    if not LIBRARY.exists():
        pytest.skip("native build missing")
    import kohakutt

    return kohakutt


@pytest.fixture(scope="module")
def native(native_module):
    """kohakutt with chip CHIP opened for one module."""
    native_module.open(CHIP)
    yield native_module
    native_module.close()
