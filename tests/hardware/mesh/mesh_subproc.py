"""Runs a mesh scenario in a subprocess with a timeout, then a second subprocess that
opens the mesh, does one small op and closes it. A failed reopen means the cards need a
reset: the helper creates the file named by KTT_RESET_MARKER (if set) and stops the
test run."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

RESET = os.environ.get("KTT_RESET_MARKER")
TIMEOUT = 60

REOPEN = """
import torch, kohakutt
dev = kohakutt.mesh([0, 1], shape=(1, 2))
try:
    x = torch.ones(32, 32).to(dev)
    assert float((x + x).sum().cpu()) == 2048.0
    print("REOPEN OK")
finally:
    kohakutt.close()
"""

PRELUDE = """
import torch, kohakutt as k

def ints(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(-3, 4, shape, generator=g).to(torch.bfloat16)

def work(dev):
    a = k.shard(ints((64, 64), 1).to(dev), 0)
    b = k.shard(ints((64, 64), 2).to(dev), 1)
    return (a @ b).cpu().float()

WANT = ints((64, 64), 1).float() @ ints((64, 64), 2).float()
"""


def _run(code, env=None, timeout=TIMEOUT):
    """(returncode or 'timeout', stdout, stderr) of `code` run by python -c; a timed-out
    child is killed by its own pid."""
    full = {**os.environ, **(env or {})}
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=full,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        return "timeout", out or "", err or ""


def scenario(body, env=None):
    """Runs PRELUDE + `body` in a subprocess, then the reopen check. Returns
    (returncode, stdout, stderr). Stops the whole run when the reopen fails."""
    result = _run(PRELUDE + body, env)
    again = _run(REOPEN)
    if again[0] != 0 or "REOPEN OK" not in again[1]:
        if RESET:
            Path(RESET).touch()
        pytest.exit(
            "mesh reopen failed after a scenario: reset.pending created.\n"
            f"scenario stderr:\n{result[2][-3000:]}\n"
            f"reopen stderr:\n{again[2][-3000:]}",
            returncode=3,
        )
    return result
