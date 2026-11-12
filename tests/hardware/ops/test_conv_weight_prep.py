"""Convs on TT-NN's conv2d with a Flat weight: once TT-NN's own prepared weight was
found equal to the device build for a conv's geometry, later weight versions are
prepared on device with no host transfer, and a weight write between two forwards
changes the output accordingly, eager and inside a captured trace."""

import pytest
import torch
import torch.nn.functional as F

import kohakutt

# x sizes, weight sizes, conv arguments. TT-NN folds the first (stride == kernel).
FOLDED = ((1, 96, 28, 28), (192, 96, 2, 2), dict(stride=2))
STRIDED = ((1, 64, 28, 28), (64, 64, 3, 3), dict(stride=2, padding=1))


@pytest.fixture(autouse=True)
def _ttnn_route(monkeypatch):
    """These convs on TT-NN's conv2d, not the columns route."""
    monkeypatch.setenv("KTT_CONV_COLUMNS", "0")


def sample(shape, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=g) * scale).to(torch.bfloat16)


def weight(ws, seed):
    return sample(ws, seed, (ws[1] * ws[2] * ws[3]) ** -0.5)


def preps():
    """(host weight preparations, device weight preparations) so far."""
    kohakutt.synchronize()
    host, device, _, _ = torch.ops.ktt._debug_conv_weight_preps()
    return host, device


def keys():
    """(geometries verified for the device build, geometries rejected) so far."""
    kohakutt.synchronize()
    _, _, verified, rejected = torch.ops.ktt._debug_conv_weight_preps()
    return verified, rejected


def rel(got, ref):
    return ((got.double() - ref).norm() / ref.norm()).item()


def check(got, x, w, args):
    """got within 3x CPU bf16's error of the float64 conv of x with w."""
    ref = F.conv2d(x.double(), w.double(), **args)
    cpu = F.conv2d(x.float(), w.float(), **args).bfloat16()
    got_rel, cpu_rel = rel(got.cpu(), ref), rel(cpu, ref)
    assert got_rel <= 3 * cpu_rel, (got_rel, cpu_rel)


@pytest.mark.hw
def test_folded_conv_weight_is_built_on_device(native, monkeypatch):
    xs, ws, args = FOLDED
    x, w1, w2, w3 = sample(xs, 0), weight(ws, 1), weight(ws, 2), weight(ws, 3)
    xt = x.to("tt")
    # The first conv of the geometry in this process prepares through the host and
    # compares; from then on the geometry is prepared on device.
    check(F.conv2d(xt, w1.to("tt"), **args), x, w1, args)
    host, device = preps()
    w2t = w2.to("tt")
    assert torch.ops.ktt._debug_layout(w2t) == "flat"
    y2 = F.conv2d(xt, w2t, **args)
    assert preps() == (host, device + 1)
    check(y2, x, w2, args)
    # The same forward again reads the cached weight.
    assert torch.equal(F.conv2d(xt, w2t, **args).cpu(), y2.cpu())
    assert preps() == (host, device + 1)
    # A weight written in place is prepared again, on device.
    w2t.copy_(w3.to("tt"))
    y3 = F.conv2d(xt, w2t, **args)
    assert preps() == (host, device + 2)
    check(y3, x, w3, args)
    # A tile-form weight takes the host preparation and gives the same output.
    monkeypatch.setenv("KTT_FLAT", "0")
    tile = w3.to("tt")
    assert torch.ops.ktt._debug_layout(tile) == "nchw"
    assert torch.equal(F.conv2d(xt, tile, **args).cpu(), y3.cpu())
    assert preps() == (host + 1, device + 2)


# TT-NN's weight for the stem is not the device build; for the folded conv it is.
STEM = ((1, 3, 224, 224), (64, 3, 7, 7), dict(stride=2, padding=3))
FOLDED_SMALL = ((1, 64, 20, 20), (96, 64, 2, 2), dict(stride=2))


@pytest.mark.hw
@pytest.mark.parametrize(
    "case, on_device", [(STEM, False), (FOLDED_SMALL, True)], ids=["stem", "folded"]
)
def test_first_weight_all_zero(native, case, on_device):
    """A geometry's verdict does not come from the first weight's values: after a first
    conv with an all-zero weight, the weight written with real values gives the float64
    conv's output, prepared through the host for the stem and on device for the folded
    conv."""
    xs, ws, args = case
    x, w = sample(xs, 11), weight(ws, 12)
    xt = x.to("tt")
    wt = torch.zeros(ws, dtype=torch.bfloat16).to("tt")
    assert torch.ops.ktt._debug_layout(wt) == "flat"
    verified, rejected = keys()
    assert not F.conv2d(xt, wt, **args).cpu().any()
    decided = keys()
    # Decided by this conv, unless an earlier conv of the geometry already did.
    first = (verified + 1, rejected) if on_device else (verified, rejected + 1)
    assert decided in ((verified, rejected), first)
    wt.copy_(w.to("tt"))
    host, device = preps()
    check(F.conv2d(xt, wt, **args), x, w, args)
    assert preps() == ((host, device + 1) if on_device else (host + 1, device))
    assert keys() == decided


@pytest.mark.hw
@pytest.mark.parametrize("case", [FOLDED, STRIDED], ids=["folded", "strided"])
def test_weight_write_between_forwards(native, case):
    """Eager: the forward after a weight write reads the new weight, whichever way the
    geometry's weight is prepared."""
    xs, ws, args = case
    x, w1, w2 = sample(xs, 4), weight(ws, 5), weight(ws, 6)
    xt, wt = x.to("tt"), w1.to("tt")
    check(F.conv2d(xt, wt, **args), x, w1, args)
    wt.copy_(w2.to("tt"))
    check(F.conv2d(xt, wt, **args), x, w2, args)
    with torch.no_grad():
        wt.mul_(0.5)
    check(F.conv2d(xt, wt, **args), x, w2 * 0.5, args)


@pytest.mark.hw
@pytest.mark.parametrize("case", [FOLDED, STRIDED], ids=["folded", "strided"])
def test_weight_write_between_replays(native, case):
    """A captured conv replays with the weight prepared before the capture, with the new
    weight after it is written in place, and an eager forward in between does not detach
    the trace from its weight. On the folded geometry the refreshes use no host
    preparation."""
    xs, ws, args = case
    x, w1, w2, w3 = sample(xs, 7), weight(ws, 8), weight(ws, 9), weight(ws, 10)
    xt, wt = x.to("tt"), w1.to("tt")
    check(F.conv2d(xt, wt, **args), x, w1, args)
    step = native.Captured(lambda a: F.conv2d(a, wt, **args), (xt,), 2)
    try:
        check(step(xt), x, w1, args)
        host, _ = preps()
        wt.copy_(w2.to("tt"))
        check(step(xt), x, w2, args)
        check(step(xt), x, w2, args)
        wt.copy_(w3.to("tt"))
        check(F.conv2d(xt, wt, **args), x, w3, args)
        check(step(xt), x, w3, args)
        if case is FOLDED:
            assert preps()[0] == host
    finally:
        step.release()


@pytest.mark.hw
@pytest.mark.parametrize(
    "case", [FOLDED, STRIDED, STEM], ids=["folded", "strided", "stem"]
)
def test_replays_do_not_prepare_unchanged_weights(native, case):
    """Replays of a captured conv prepare its weight again only after the weight was
    written: none over three replays, one after a write, none on the replay after it."""
    xs, ws, args = case
    x, w1, w2 = sample(xs, 20), weight(ws, 21), weight(ws, 22)
    xt, wt = x.to("tt"), w1.to("tt")
    check(F.conv2d(xt, wt, **args), x, w1, args)
    step = native.Captured(lambda a: F.conv2d(a, wt, **args), (xt,), 2)
    try:
        check(step(xt), x, w1, args)
        before = preps()
        for _ in range(3):
            check(step(xt), x, w1, args)
        assert preps() == before
        wt.copy_(w2.to("tt"))
        check(step(xt), x, w2, args)
        after = preps()
        assert sum(after) == sum(before) + 1
        check(step(xt), x, w2, args)
        assert preps() == after
    finally:
        step.release()
