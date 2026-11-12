"""close() ends the device completely: a process may end right after it (os._exit) and
the next one opens the chip (also test_lifecycle_second_process.py and
test_lifecycle_live_state.py, one file each to keep every file short)."""

import os
import subprocess
import sys

import pytest

CHIP = os.environ.get("KTT_TEST_CHIP", "0")

OPEN_CLOSE = f"""
import torch, kohakutt
kohakutt.open({CHIP})
print("sum", float((torch.ones(32, 32).to("tt") + 1).sum().cpu()), flush=True)
kohakutt.close()
print("closed", flush=True)
"""


def _python(script, *args, **kwargs):
    return subprocess.run(
        [sys.executable, "-u", "-c", script, *args],
        capture_output=True,
        text=True,
        timeout=60,
        **kwargs,
    )


@pytest.mark.hw
def test_exit_right_after_close_leaves_the_chip_usable(native_module):
    ended = _python(OPEN_CLOSE + "import os\nos._exit(0)\n")
    assert ended.returncode == 0 and "closed" in ended.stdout, ended.stderr[-2000:]
    after = _python(OPEN_CLOSE)
    assert after.returncode == 0, after.stderr[-2000:]
    assert "sum 2048.0" in after.stdout
