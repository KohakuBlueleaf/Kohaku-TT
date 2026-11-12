"""Nearest upsampling with activations in L1 (KTT_L1_BUDGET=0.3, read once per process:
its own process) below the row copy's circular buffers (sized within cb_budget):
bit-exact against PyTorch."""

import os
import subprocess
import sys

import pytest

L1_SCRIPT = """
import torch, torch.nn.functional as F, kohakutt
kohakutt.open(int(__import__("os").environ.get("KTT_TEST_CHIP", "0")))
g = torch.Generator().manual_seed(0)
live = [torch.randn(2, 64, 64, 64, generator=g).bfloat16().to("tt") * 1.0
        for _ in range(6)]
for shape, dt in (((2, 64, 32, 32), torch.bfloat16), ((1, 40, 17, 16), torch.float32)):
    for layout in ("nchw", "nhwc"):
        x = torch.randn(*shape, generator=g).to(dt)
        t = x.to("tt") * 1.0  # an activation: in L1 when it fits
        if layout == "nhwc":
            torch.ops.ktt._debug_to_nhwc(t)
        got = F.interpolate(t, scale_factor=2, mode="nearest").cpu()
        ok = torch.equal(got, F.interpolate(x, scale_factor=2, mode="nearest"))
        print("OK" if ok else "FAIL", shape, dt, layout, flush=True)
kohakutt.close()
"""


@pytest.mark.hw
def test_upsample_with_activations_in_l1(native_module):
    """KTT_L1_BUDGET=0.3 (read once per process: its own process): live activations in
    L1 below the row copy's circular buffers, results bit-exact."""
    run = subprocess.run(
        [sys.executable, "-u", "-c", L1_SCRIPT],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "KTT_L1_BUDGET": "0.3"},
    )
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.count("OK") == 4, run.stdout
