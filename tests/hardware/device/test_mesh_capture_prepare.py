"""Regression: a capture's prepare pass runs the programs its capture runs, so nothing
compiles inside the capture. Each step is captured with no eager call before the
prepare pass, then replayed on a new input and compared with eager: a heads merge
(x.transpose(1, 2).contiguous() of a whole [B, H, S, D] tensor) used inside the step,
a zero-element host tensor and a host scalar moved to the device inside the step, and
(1x2 mesh) a tensor allocated inside the step at a size that is placed sharded outside
captures."""

import os

import pytest
import torch

CHIP = int(os.environ.get("KTT_TEST_CHIP", "0"))


def _merge_heads(t):
    merged = (t * 2).transpose(1, 2).contiguous()
    return merged.reshape(t.shape[0], t.shape[2], -1) * 3


def _empty_host(t):
    return t * 2, torch.tensor([], dtype=t.dtype, device=t.device)


def _host_scalar(t):
    return t * torch.tensor(1.5, dtype=t.dtype).to(t.device)


def _allocated(t):
    # 2 MiB of bf16: sharded when allocated outside a capture.
    buf = torch.empty(1024, 1024, dtype=t.dtype, device=t.device)
    buf.copy_(t.expand(1024, 1024))
    return buf * 3


def _replay_equals_eager(kohakutt, step, shape):
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(*shape, generator=gen, dtype=torch.bfloat16).to("tt")
    x2 = torch.randn(*shape, generator=gen, dtype=torch.bfloat16).to("tt")
    with torch.no_grad():
        captured = kohakutt.capture(step, x, warmup=1)
        got = captured(x2)
        got = [t.cpu() for t in (got if isinstance(got, tuple) else (got,))]
        captured.release()
        want = step(x2)
        want = [t.cpu() for t in (want if isinstance(want, tuple) else (want,))]
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert g.shape == w.shape
        assert torch.equal(g, w)


CASES = {
    "merge_heads": (_merge_heads, (1, 12, 64, 64)),
    "empty_host": (_empty_host, (256, 256)),
    "host_scalar": (_host_scalar, (256, 256)),
}


@pytest.mark.hw
@pytest.mark.parametrize("case", sorted(CASES))
def test_prepare_matches_capture_one_chip(native_module, case):
    kohakutt = native_module
    kohakutt.open(CHIP)
    try:
        _replay_equals_eager(kohakutt, *CASES[case])
    finally:
        kohakutt.close()


@pytest.mark.hw
@pytest.mark.dual
@pytest.mark.parametrize("case", sorted([*CASES, "allocated"]))
def test_prepare_matches_capture_mesh(native_module, case):
    kohakutt = native_module
    kohakutt.mesh([0, 1], shape=(1, 2))
    try:
        step, shape = CASES.get(case, (_allocated, (1, 1024)))
        _replay_equals_eager(kohakutt, step, shape)
    finally:
        kohakutt.close()
