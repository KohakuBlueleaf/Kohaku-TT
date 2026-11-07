"""Regression: a process exiting with the device open and tt tensors, cached
constants and a captured trace alive (no close()) exits cleanly."""

import pytest


@pytest.mark.hw
def test_exit_without_close(exit_run):
    assert exit_run.returncode == 0, exit_run.stderr[-2000:]
    assert "done (2, 3, 4, 4) (2, 3, 8, 8)" in exit_run.stdout
