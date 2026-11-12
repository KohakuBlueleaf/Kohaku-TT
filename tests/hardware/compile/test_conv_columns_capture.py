"""A captured columns-route conv reads the tap-major weight prepared before the
capture: the trace allocates no weight matrix (so a replay lays out no weight), a
replay equals the eager conv, and a weight written after the capture is seen by the
next replay and by eager calls."""

import pytest
import torch
import torch.nn.functional as F

from kohakutt.torch_backend.trace import footprint

# x sizes, weight sizes, conv arguments: a small activation with a large weight, so
# a weight matrix allocated in the capture dominates its device bytes.
CASES = {
    "3x3": ((1, 1280, 4, 4), (1280, 1280, 3, 3), dict(padding=1)),
    "1x1": ((1, 1280, 4, 4), (1280, 1280, 1, 1), dict()),
}


def sample(shape, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=g) * scale).to(torch.bfloat16)


def weight(ws, seed):
    return sample(ws, seed, (ws[1] * ws[2] * ws[3]) ** -0.5)


def rel(got, ref):
    return ((got.double() - ref).norm() / ref.norm()).item()


def check(got, x, w, args):
    """got within 3x CPU bf16's error of the float64 conv of x with w."""
    ref = F.conv2d(x.double(), w.double(), **args)
    cpu = F.conv2d(x.float(), w.float(), **args).bfloat16()
    got_rel, cpu_rel = rel(got.cpu(), ref), rel(cpu, ref)
    assert got_rel <= 3 * cpu_rel, (got_rel, cpu_rel)


@pytest.mark.hw
@pytest.mark.parametrize("case", list(CASES))
def test_capture_reads_prepared_weight(native, monkeypatch, case):
    """The device memory freed during the capture (the trace's temporaries, fenced
    for every replay) stays far below the tap-major weight matrix's size: the matrix
    prepared before the capture is read, not laid out again in the trace. (The bytes
    left allocated are the conv's output, kept by any capture.) A replay equals the
    eager conv."""
    monkeypatch.setenv("KTT_CONV_COLUMNS", "all")
    xs, ws, args = CASES[case]
    x, w = sample(xs, 1), weight(ws, 2)
    xt, wt = x.to("tt"), w.to("tt")

    def conv(a):
        return F.conv2d(a, wt, **args)

    with torch.no_grad():
        want = conv(xt).cpu()
        check(want, x, w, args)
        step = native.capture(conv, xt, warmup=2)
        try:
            f = footprint(step)
            matrix = ws[0] * ws[1] * ws[2] * ws[3] * 2
            assert f["fenced_bytes"] < matrix // 2, (f, matrix)
            for _ in range(2):
                assert torch.equal(step(xt).cpu(), want)
        finally:
            step.release()


@pytest.mark.hw
@pytest.mark.parametrize("case", list(CASES))
def test_capture_follows_weight_writes(native, monkeypatch, case):
    """Weight writes after the capture (in place, then a copy) reach the next replay,
    which equals a fresh eager conv of the new weight; eager convs before and after
    the capture agree with the CPU."""
    monkeypatch.setenv("KTT_CONV_COLUMNS", "all")
    xs, ws, args = CASES[case]
    x, w1, w3 = sample(xs, 3), weight(ws, 4), weight(ws, 5)
    xt, wt = x.to("tt"), w1.to("tt")

    def conv(a):
        return F.conv2d(a, wt, **args)

    with torch.no_grad():
        check(conv(xt), x, w1, args)
        step = native.capture(conv, xt, warmup=2)
        try:
            assert torch.equal(step(xt).cpu(), conv(xt).cpu())
            wt.mul_(2)
            got = step(xt).cpu()
            assert torch.equal(got, conv(xt).cpu())
            check(got, x, w1 * 2, args)
            # Two replays without a write between them read the same weight.
            assert torch.equal(step(xt).cpu(), got)
            wt.copy_(w3.to("tt"))
            eager = conv(xt).cpu()
            check(eager, x, w3, args)
            assert torch.equal(step(xt).cpu(), eager)
        finally:
            step.release()
        check(conv(xt), x, w3, args)
