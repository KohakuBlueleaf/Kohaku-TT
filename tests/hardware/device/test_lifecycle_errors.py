"""Clear errors for two misuses: a tt tensor used after its device was closed and opened
again, and a module-level shard mark on tensors that lack the dim."""

import os
import subprocess
import sys

import pytest
import torch

CHIP = os.environ.get("KTT_TEST_CHIP", "0")

STALE_SCRIPT = """
import sys
import torch
import kohakutt

chips = [int(c) for c in sys.argv[1].split(",")]
dev = kohakutt.mesh(chips, shape=(1, len(chips)))
old = torch.ones(64, 64).to(dev)
kohakutt.close()
dev = kohakutt.mesh(chips, shape=(1, len(chips)))
try:
    (old + 1).cpu()
except RuntimeError as error:
    print("raised", str(error).splitlines()[0])
fresh = torch.ones(32, 32).to(dev)
print("fresh", float((fresh + fresh).sum().cpu()))
kohakutt.close()
"""


def _stale_run(chips):
    return subprocess.run(
        [sys.executable, "-c", STALE_SCRIPT, chips],
        capture_output=True,
        text=True,
        timeout=100,
    )


def _check_stale(run):
    assert run.returncode == 0, run.stderr[-2000:]
    assert "raised kohakutt: this tt tensor belongs to a device" in run.stdout
    assert "TT_FATAL" not in run.stdout
    assert "fresh 2048.0" in run.stdout


@pytest.mark.hw
def test_stale_tensor_one_card(native_module):
    _check_stale(_stale_run(CHIP))


@pytest.mark.hw
@pytest.mark.dual
def test_stale_tensor_mesh(native_module):
    _check_stale(_stale_run("0,1"))


@pytest.mark.hw
@pytest.mark.dual
def test_shard_module_skips_missing_dim(native_module):
    """shard(module, 1) leaves the 1-d bias replicated; a mark of its own on it raises
    an error naming it and its shape."""
    kohakutt = native_module
    dev = kohakutt.mesh([0, 1], shape=(1, 2))
    try:
        net = torch.nn.Sequential(torch.nn.Linear(64, 96)).bfloat16()
        kohakutt.shard(net, 1).to(dev)
        assert kohakutt.placement(net[0].weight) == "Sharded(dim=1)"
        assert kohakutt.placement(net[0].bias) == "Replicated"
        net = torch.nn.Sequential(torch.nn.Linear(64, 96)).bfloat16()
        kohakutt.place(net[0], bias="S(1)")
        with pytest.raises(ValueError, match=r"0\.bias has shape \(96,\) and no dim 1"):
            net.to(dev)
    finally:
        kohakutt.close()
