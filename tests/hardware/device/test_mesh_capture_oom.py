"""Regression: an op that runs out of device memory inside a trace capture fails the
capture cleanly (the trace is ended and released, the error raised) and the device stays
usable: eager ops and a new capture run after it, on one chip and on the 1x2 mesh."""

import os

import pytest
import torch

CHIP = int(os.environ.get("KTT_TEST_CHIP", "0"))

# 64 GiB of bf16: more DRAM than a P150A has, also when sharded over two chips.
HUGE = (1 << 18, 1 << 17)


def _check_capture_oom(kohakutt):
    gen = torch.Generator().manual_seed(0)
    host = torch.randn(256, 512, generator=gen, dtype=torch.bfloat16)
    x = host.to("tt")
    calls = []

    def step(t):
        calls.append(1)
        out = t * 2 + 1
        if len(calls) == 2:
            # The second call is the capture (the first is the prepare pass).
            torch.empty(HUGE, dtype=torch.bfloat16, device=t.device)
        return out

    with torch.no_grad():
        with pytest.raises(RuntimeError, match="(?i)memory"):
            kohakutt.Captured(step, (x,), 1)
        # Eager ops after the failure.
        eager = (x * 2 + 1).cpu()
        # A new capture after the failure replays like eager.
        good = kohakutt.capture(lambda t: t * 2 + 1, x, warmup=1)
        x2 = (host * -0.5).to("tt")
        replay = good(x2).cpu()
        want = (x2 * 2 + 1).cpu()
        good.release()
    assert len(calls) == 2
    assert torch.equal(eager, (x * 2 + 1).cpu())
    assert torch.equal(replay, want)


@pytest.mark.hw
def test_capture_oom_one_chip(native_module):
    kohakutt = native_module
    kohakutt.open(CHIP)
    try:
        _check_capture_oom(kohakutt)
    finally:
        kohakutt.close()


@pytest.mark.hw
@pytest.mark.dual
def test_capture_oom_mesh(native_module):
    kohakutt = native_module
    kohakutt.mesh([0, 1], shape=(1, 2))
    try:
        _check_capture_oom(kohakutt)
    finally:
        kohakutt.close()
