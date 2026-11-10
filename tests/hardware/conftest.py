"""Hardware test fixtures: the native tt backend (opened per test module) and the
exit-without-close run."""

import os
import signal
import subprocess
import sys
import threading
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

# A process ending with the 1x2 mesh open, tt tensors and a trace alive: `how` is
# "exit" (falls off the end), "sys.exit" or "raise" (an uncaught exception).
MESH_EXIT_SCRIPT = """
import sys
import torch
import kohakutt
from kohakutt import Captured

how = sys.argv[1]
dev = kohakutt.mesh([0, 1], shape=(1, 2))
x = torch.randn(64, 64).to(dev)
y = kohakutt.shard(torch.randn(64, 64).to(dev), 0) @ x
step = Captured(lambda t: t * 2 + 1, (x,), 1)
z = step(x)
print("done", how, tuple(y.shape), tuple(z.shape), flush=True)
if how == "sys.exit":
    sys.exit(0)
if how == "raise":
    raise RuntimeError("uncaught")
"""

# Opens the 1x2 mesh, computes on it and closes it.
MESH_REOPEN_SCRIPT = """
import torch
import kohakutt

dev = kohakutt.mesh([0, 1], shape=(1, 2))
x = torch.ones(32, 32).to(dev)
print("reopened", float((x + x).sum().cpu()), flush=True)
kohakutt.close()
"""

# Opens the chips of argv[1] (e.g. "0" or "0,1"), keeps a trace replaying and waits
# for a signal; argv[2] "kill" ends the process with SIGKILL after "ready" instead.
SIGNAL_SCRIPT = """
import os
import signal
import sys
import time
import torch
import kohakutt
from kohakutt import Captured

chips = [int(c) for c in sys.argv[1].split(",")]
dev = kohakutt.mesh(chips, shape=(1, len(chips)))
x = torch.randn(64, 64).to(dev)
step = Captured(lambda t: t * 2 + 1, (x,), 1)
step(x)
print("ready", flush=True)
if sys.argv[2:] == ["kill"]:
    os.kill(os.getpid(), signal.SIGKILL)
while True:
    step(x)
    time.sleep(0.01)
"""

# Opens the chips of argv[1], computes on them and closes; prints the error type and
# text instead when the open fails.
REOPEN_SCRIPT = """
import sys
import torch
import kohakutt

chips = [int(c) for c in sys.argv[1].split(",")]
try:
    dev = kohakutt.mesh(chips, shape=(1, len(chips)))
except Exception as error:
    print("open failed", type(error).__name__, error, flush=True)
    sys.exit(3)
x = torch.ones(32, 32).to(dev)
print("reopened", float((x + x).sum().cpu()), flush=True)
kohakutt.close()
"""

_EXIT_RUN = {}


def _run_signalled(signum, chips, timeout=100):
    """Starts SIGNAL_SCRIPT on `chips`, sends `signum` to that process once it prints
    "ready" and returns its CompletedProcess (code 124 when it outlived `timeout`)."""
    command = [sys.executable, "-c", SIGNAL_SCRIPT, chips]
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    late = threading.Timer(timeout, proc.kill)
    late.start()
    head, sent = "", False
    try:
        for line in proc.stdout:
            head += line
            if line.startswith("ready"):
                proc.send_signal(signum)
                sent = True
                break
        out, err = proc.communicate()
    finally:
        late.cancel()
    code = proc.returncode if sent or proc.returncode != -signal.SIGKILL else 124
    return subprocess.CompletedProcess(command, code, head + out, err)


def _run(script, *args, timeout=600):
    """The CompletedProcess of `script` run with `args`; return code 124 when it took
    longer than `timeout` seconds."""
    command = [sys.executable, "-c", script, *args]
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as late:
        out = late.stdout.decode() if isinstance(late.stdout, bytes) else late.stdout
        return subprocess.CompletedProcess(command, 124, out or "", "timed out")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """Runs the exit scripts in subprocesses before any test opens a device (this
    process holds the chip lock from its first open on): EXIT_SCRIPT when test_exit is
    selected, each MESH_EXIT_SCRIPT ending and a reopen when test_mesh_exit is, up to
    the first reopen that fails (the cards need a board reset from then on)."""
    if not LIBRARY.exists() or os.environ.get("KTT_HW") != "1":
        return
    if any("test_exit" in item.nodeid for item in items):
        _EXIT_RUN["run"] = _run(EXIT_SCRIPT)
    both_cards = len(os.environ.get("TT_VISIBLE_DEVICES", "0,1").split(",")) >= 2
    if both_cards and any("test_mesh_exit" in item.nodeid for item in items):
        _EXIT_RUN["mesh"] = {}
        for how in ("exit", "sys.exit", "raise"):
            name = f"test_mesh_exit_without_close[{how}]"
            if not any(item.nodeid.endswith(name) for item in items):
                continue
            ended = _run(MESH_EXIT_SCRIPT, how, timeout=150)
            reopened = _run(MESH_REOPEN_SCRIPT, timeout=150)
            _EXIT_RUN["mesh"][how] = (ended, reopened)
            if reopened.returncode != 0:
                break
    _run_signal_endings(items, both_cards)
    _run_bad_state(items, both_cards)


def _selected(items, name):
    return any(item.nodeid.endswith(name) for item in items)


def _run_signal_endings(items, both_cards):
    """Fills _EXIT_RUN["signal"] with {(cards, signal name): (ended, reopened)}: a
    process signalled with a device open, then a reopen. Stops at the first failed
    reopen; the mesh runs only after the one-card ones passed."""
    runs = _EXIT_RUN["signal"] = {}
    for cards, chips in (("one_card", str(CHIP)), ("mesh", "0,1")):
        if cards == "mesh" and not both_cards:
            continue
        for name in ("SIGTERM", "SIGINT"):
            if not _selected(items, f"test_signal_without_close[{cards}-{name}]"):
                continue
            if any(run[1].returncode != 0 for run in runs.values()):
                return
            ended = _run_signalled(getattr(signal, name), chips)
            runs[cards, name] = (ended, _run(REOPEN_SCRIPT, chips, timeout=150))


def _run_bad_state(items, both_cards):
    """Fills _EXIT_RUN["bad_state"] with (the process killed by SIGKILL with the mesh
    open, the open after it) when KTT_BAD_STATE=1; the cards need a reset after."""
    wanted = os.environ.get("KTT_BAD_STATE") == "1"
    if both_cards and wanted and _selected(items, "test_mesh_bad_state_error"):
        killed = _run(SIGNAL_SCRIPT, "0,1", "kill", timeout=100)
        _EXIT_RUN["bad_state"] = (killed, _run(REOPEN_SCRIPT, "0,1", timeout=150))


@pytest.fixture
def exit_run():
    """EXIT_SCRIPT's CompletedProcess; skips when it did not run."""
    if "run" not in _EXIT_RUN:
        pytest.skip("native build missing")
    return _EXIT_RUN["run"]


@pytest.fixture
def mesh_exit_runs():
    """{how: (the ended process, the reopening process)} of MESH_EXIT_SCRIPT; skips
    when they did not run."""
    if "mesh" not in _EXIT_RUN:
        pytest.skip("native build missing, or one card visible")
    return _EXIT_RUN["mesh"]


@pytest.fixture
def signal_runs():
    """{(cards, signal name): (ended, reopened)} of SIGNAL_SCRIPT; skips if none ran."""
    if not _EXIT_RUN.get("signal"):
        pytest.skip("native build missing, or not selected at collection")
    return _EXIT_RUN["signal"]


@pytest.fixture
def bad_state_runs():
    """(killed, reopened) of the bad-state run; skips unless KTT_BAD_STATE=1."""
    if "bad_state" not in _EXIT_RUN:
        pytest.skip("leaves the cards needing a board reset: set KTT_BAD_STATE=1")
    return _EXIT_RUN["bad_state"]


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
