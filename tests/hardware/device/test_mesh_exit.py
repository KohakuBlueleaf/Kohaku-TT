"""Regression: a process ending with the 1x2 mesh open and tt tensors and a captured
trace alive (no close()) leaves both cards openable: the next process opens the mesh,
computes and closes. A failure of this test means the cards need a board reset."""

import pytest


@pytest.mark.hw
@pytest.mark.dual
@pytest.mark.parametrize("how", ["exit", "sys.exit", "raise"])
def test_mesh_exit_without_close(mesh_exit_runs, how):
    if how not in mesh_exit_runs:
        dead = any(run.returncode != 0 for _, run in mesh_exit_runs.values())
        pytest.skip(
            "cards need a reset after a failed reopen"
            if dead
            else "not selected at collection"
        )
    ended, reopened = mesh_exit_runs[how]
    assert f"done {how} (64, 64) (64, 64)" in ended.stdout, ended.stderr[-2000:]
    assert ended.returncode == (1 if how == "raise" else 0), ended.stderr[-2000:]
    assert reopened.returncode == 0, reopened.stderr[-2000:]
    assert "reopened 2048.0" in reopened.stdout
