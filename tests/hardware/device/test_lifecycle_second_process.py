"""close() ends the device completely: another process may open the chip while a
closed one lives on."""

import os
import subprocess
import sys
import time

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
def test_another_process_opens_while_a_closed_one_lives(native_module):
    holder = subprocess.Popen(
        [sys.executable, "-u", "-c", OPEN_CLOSE + "import time\ntime.sleep(30)\n"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        for line in holder.stdout:
            if "closed" in line:
                break
        start = time.monotonic()
        other = _python(OPEN_CLOSE)
        assert other.returncode == 0, other.stderr[-2000:]
        assert "sum 2048.0" in other.stdout
        assert holder.poll() is None, "the first process ended before the second opened"
        assert time.monotonic() - start < 25
    finally:
        holder.terminate()
        holder.wait()
