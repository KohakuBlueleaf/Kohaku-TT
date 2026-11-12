"""Convs on the columns route (im2col rows from ktt_conv::to_columns, then the
matmul): strides, dilations, 1x1, partial last row tiles and odd sizes agree with
PyTorch on CPU, a warm call is three own programs, and a captured conv follows weight
writes between replays."""

import pytest
import torch
import torch.nn.functional as F

# x sizes, weight sizes, conv arguments.
CASES = {
    "unet3x3": ((2, 1280, 16, 16), (1280, 1280, 3, 3), dict(padding=1)),
    "stride2": ((2, 320, 30, 30), (320, 320, 3, 3), dict(stride=2, padding=1)),
    "dilated": ((1, 64, 21, 19), (96, 64, 3, 3), dict(padding=2, dilation=2)),
    "odd": ((3, 96, 13, 7), (64, 96, 3, 3), dict(padding=1)),
    "valid5x5": ((1, 32, 20, 24), (40, 32, 5, 5), dict()),
    "rect": ((1, 64, 12, 16), (64, 64, (1, 3)), dict(padding=(0, 1))),
    "1x1": ((2, 256, 9, 11), (300, 256, 1, 1), dict()),
    "folded": ((1, 96, 28, 28), (192, 96, 2, 2), dict(stride=2)),
    # 94 MB of im2col rows: two chunks of at most 64 MB.
    "chunked": ((2, 640, 64, 64), (64, 640, 3, 3), dict(padding=1)),
}


def sample(shape, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=g) * scale).to(torch.bfloat16)


def weight(ws, seed):
    if len(ws) == 3:  # (O, C, (KH, KW))
        ws = (ws[0], ws[1], *ws[2])
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
def test_columns_route_matches_cpu(native, monkeypatch, case):
    monkeypatch.setenv("KTT_CONV_COLUMNS", "all")
    xs, ws, args = CASES[case]
    x, w = sample(xs, 1), weight(ws, 2)
    check(F.conv2d(x.to("tt"), w.to("tt"), **args), x, w, args)


@pytest.mark.hw
def test_columns_route_programs(native, monkeypatch):
    """The measured rule takes the UNet conv: own programs to_rows, to_columns and
    to_nchw around the matmul; KTT_CONV_COLUMNS=0 sends it back to TT-NN's conv2d
    (no own program)."""
    xs, ws, args = CASES["unet3x3"]
    xt, wt = sample(xs, 3).to("tt"), weight(ws, 4).to("tt")

    def programs():
        F.conv2d(xt, wt, **args)
        torch.tt.synchronize()
        before = native.counters()["programs"]
        F.conv2d(xt, wt, **args)
        torch.tt.synchronize()
        return native.counters()["programs"] - before

    monkeypatch.delenv("KTT_CONV_COLUMNS", raising=False)
    assert programs() == 3
    monkeypatch.setenv("KTT_CONV_COLUMNS", "0")
    assert programs() == 0


@pytest.mark.hw
@pytest.mark.parametrize("case", ["unet3x3", "stride2"])
def test_columns_route_replays_follow_weight_writes(native, monkeypatch, case):
    monkeypatch.setenv("KTT_CONV_COLUMNS", "all")
    xs, ws, args = CASES[case]
    x, w1, w2, w3 = sample(xs, 5), weight(ws, 6), weight(ws, 7), weight(ws, 8)
    xt, wt = x.to("tt"), w1.to("tt")
    check(F.conv2d(xt, wt, **args), x, w1, args)
    step = native.Captured(lambda a: F.conv2d(a, wt, **args), (xt,), 2)
    try:
        check(step(xt), x, w1, args)
        wt.copy_(w2.to("tt"))
        check(step(xt), x, w2, args)
        wt.copy_(w3.to("tt"))
        check(F.conv2d(xt, wt, **args), x, w3, args)
        check(step(xt), x, w3, args)
    finally:
        step.release()
