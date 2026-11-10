"""Regression: a process ending with the 1x2 mesh open and tt tensors and a captured
trace alive (no close()) leaves both cards openable: the next process opens the mesh,
computes and closes. A failure of this test means the cards need a board reset.
The same for a process ended by SIGTERM or SIGINT, on one card and on the mesh."""

import signal

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


@pytest.mark.hw
@pytest.mark.parametrize(
    "cards,name",
    [(c, n) for c in ("one_card", "mesh") for n in ("SIGTERM", "SIGINT")],
)
def test_signal_without_close(signal_runs, cards, name):
    """A process signalled with the device open ends by that signal (SIGINT: or exit
    code 130 or 1) and leaves the cards openable."""
    if (cards, name) not in signal_runs:
        pytest.skip("not run: not selected, or an earlier reopen failed")
    ended, reopened = signal_runs[cards, name]
    assert "ready" in ended.stdout, ended.stderr[-2000:]
    expected = {-getattr(signal, name)} | ({130, 1} if name == "SIGINT" else set())
    assert ended.returncode in expected, (ended.returncode, ended.stderr[-2000:])
    assert reopened.returncode == 0, reopened.stdout + reopened.stderr[-2000:]
    assert "reopened 2048.0" in reopened.stdout
