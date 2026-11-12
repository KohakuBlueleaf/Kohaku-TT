"""Importing kohakutt under a PyTorch older than 2.11 fails with a clear message."""

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parents[2] / "src"


def test_old_torch_is_refused():
    code = (
        f"import sys; sys.path.insert(0, {str(SRC)!r}); import torch; "
        "torch.__version__ = '2.10.0'; import kohakutt_native"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert run.returncode != 0
    assert "PyTorch 2.11 or newer" in run.stderr
