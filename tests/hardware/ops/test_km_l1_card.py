"""Metal-level programs on a crowded L1 (tests/km_l1.py in a process whose activations
live in L1): a repeat launch and a torch-op replay are refused by the launch-time check
and a new launch by the first-launch check, a trace replay by the trace's L1 guard;
each runs again once the L1 tensors are freed."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "km_l1.py"
REFUSED = re.compile(
    r"RuntimeError: .*user program l1_chain: this launch's circular buffers and "
    r"scratch take (\d+) bytes of L1 per core, (\d+) are free below the lowest live "
    r"L1 buffer; nothing was launched$"
)


def _outcomes(mode):
    """{label: outcome} of the script's run in `mode`."""
    env = {**os.environ, "KTT_L1_BUDGET": "0.7"}
    run = subprocess.run(
        [sys.executable, str(SCRIPT), mode],
        capture_output=True,
        text=True,
        timeout=100,
        env=env,
    )
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    lines = [line.split(": ", 1) for line in run.stdout.splitlines()]
    return {line[0]: line[1] for line in lines if len(line) == 2}


@pytest.mark.hw
def test_launches_on_a_crowded_l1_are_refused_and_run_once_it_is_freed():
    out = _outcomes("eager")
    assert out["first launch"] == "ran, equal True"
    assert out["first op"] == "ran, equal True"
    for label in ("repeat launch", "op replay"):
        refused = REFUSED.search(out[label])
        assert refused, out[label]
        need, free = map(int, refused.groups())
        assert free < need
    assert re.fullmatch(
        r"ValueError: kohakutt.dsl.metal: l1_chain: this launch's circular buffers "
        r"and scratch take \d+ bytes of L1 per core, \d+ are free",
        out["new launch"],
    )
    assert out["after repeat launch"] == "ran, equal True"
    assert out["after op replay"] == "ran, equal True"


@pytest.mark.hw
def test_a_traced_program_is_not_replayed_over_a_live_l1_tensor():
    out = _outcomes("trace")
    assert out["captured"] == "ran, equal True"
    assert "lies inside the circular-buffer region of trace" in out["replay"]
    assert out["after replay"] == "ran, equal True"
