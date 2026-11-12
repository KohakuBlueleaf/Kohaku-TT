"""close() with live tensors, traces and a failed queued op still closes, after which
open() works again and the old tensors report the closed device."""

import os
import subprocess
import sys

import pytest

CHIP = os.environ.get("KTT_TEST_CHIP", "0")

# Live tensors, a trace, a compiled graph's trace and a failed queued op at close().
HARD_CLOSE = f"""
import torch, kohakutt
from kohakutt.torch_backend import compiler as tt_compile
kohakutt.open({CHIP})
tt_compile.register()
x = torch.randn(64, 64).to("tt")
step = kohakutt.capture(lambda t: t * 2 + 1, x)
f = torch.compile(lambda t: (t @ t).relu(), backend="ktt_tt")
for _ in range(4):
    y = f(x)
step(x)
torch.full((24, 2**30), 1.0, dtype=torch.bfloat16, device="tt")  # fails when run
try:
    kohakutt.close()
except RuntimeError as error:
    print("close raised", flush=True)
kohakutt.open({CHIP})
print("reopened", float((torch.ones(32, 32).to("tt") * 3).sum().cpu()), flush=True)
try:
    (x + 1).cpu()
except RuntimeError as error:
    print("stale", "has been closed" in str(error), flush=True)
kohakutt.close()
"""


def _python(script):
    return subprocess.run(
        [sys.executable, "-u", "-c", script], capture_output=True, text=True, timeout=60
    )


@pytest.mark.hw
def test_close_with_live_state_then_open_again(native_module):
    run = _python(HARD_CLOSE)
    assert run.returncode == 0, run.stderr[-2000:]
    for want in ("close raised", "reopened 3072.0", "stale True"):
        assert want in run.stdout, run.stdout
