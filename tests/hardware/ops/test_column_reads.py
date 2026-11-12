"""Regression: last-dim strided reads (x[..., o::s], step 2 .. 16) and members of a
small last dim (x[..., j], unbind(-1), complex real / imag parts) run as one Kohaku-TT
data-movement program each; values equal CPU exactly, also under
KTT_COLUMN_PROGRAMS=0 and in a replayed capture."""

import pytest
import torch

import kohakutt

DTYPES = [torch.bfloat16, torch.float32, torch.int32]


def sample(shape, seed, dtype):
    g = torch.Generator().manual_seed(seed)
    if dtype == torch.int32:
        return torch.randint(-1000, 1000, shape, generator=g, dtype=dtype)
    return torch.randn(shape, generator=g).to(dtype)


def ops_of(fn):
    kohakutt.synchronize()
    before = kohakutt.counters()["ops"]
    out = fn()
    kohakutt.synchronize()
    return kohakutt.counters()["ops"] - before, out


@pytest.mark.hw
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize(
    "shape, index, most",
    [
        ((3, 40, 64), (Ellipsis, slice(0, None, 2)), 1),
        ((3, 40, 64), (Ellipsis, slice(1, None, 2)), 1),
        ((2, 33, 100), (Ellipsis, slice(5, 97, 3)), 1),
        ((70, 520), (Ellipsis, slice(7, None, 16)), 1),
        ((2, 3, 50, 96), (slice(1, 2), Ellipsis, slice(2, None, 4)), 2),
        ((4, 64, 48), (slice(None), slice(3, 40), slice(1, None, 5)), 2),
    ],
    ids=["even", "odd", "step3", "step16", "rows_cut", "mid_rows"],
)
def test_strided_columns(native, dtype, shape, index, most):
    x = sample(shape, 1, dtype)
    t = x.to("tt")
    n, got = ops_of(lambda: t[index].contiguous())
    assert torch.equal(got.cpu(), x[index].contiguous())
    assert n <= most, n


@pytest.mark.hw
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize(
    "shape, j",
    [
        ((20000, 4), 0),
        ((20000, 4), 3),
        ((9, 16, 72, 32, 2), 1),
        ((3, 70, 45, 7), 6),
        ((2, 33, 31), 30),
        ((5, 1, 2), 1),
    ],
    ids=["boxes0", "boxes3", "pairs", "odd_sizes", "d31", "tiny"],
)
def test_member_of_small_last_dim(native, dtype, shape, j):
    x = sample(shape, 2, dtype)
    t = x.to("tt")
    n, got = ops_of(lambda: t[..., j].contiguous())
    assert torch.equal(got.cpu(), x[..., j])
    assert n <= 1, n


@pytest.mark.hw
def test_unbind_and_partial_members(native):
    x = sample((6, 50, 4), 3, torch.float32)
    t = x.to("tt")
    for got, want in zip(t.unbind(-1), x.unbind(-1), strict=True):
        assert torch.equal(got.contiguous().cpu(), want)
    # Not the whole base: the general route, same values.
    assert torch.equal(t[1:4, :, 2].contiguous().cpu(), x[1:4, :, 2])
    assert torch.equal(t[..., 1:3].contiguous().cpu(), x[..., 1:3])


@pytest.mark.hw
def test_pairs_of_a_wider_storage(native):
    x = sample((9, 16, 64, 64), 4, torch.float32)
    t = x.to("tt")
    pairs, tp = x.view(9, 16, 64, 32, 2), t.view(9, 16, 64, 32, 2)
    for j in range(2):
        n, got = ops_of(lambda j=j: tp[..., j].contiguous())
        assert torch.equal(got.cpu(), pairs[..., j]) and n <= 1
    a, b = tp.unbind(-1)
    assert torch.equal((-b).cpu(), -pairs[..., 1])
    c = torch.randn(2, 8, 100, 32, dtype=torch.complex64)
    tc = c.to("tt")
    assert torch.equal(tc.real.contiguous().cpu(), c.real)
    assert torch.equal(tc.imag.contiguous().cpu(), c.imag)
    assert torch.equal(torch.view_as_real(tc)[..., 1].contiguous().cpu(), c.imag)


@pytest.mark.hw
def test_knob_off_same_values(native, monkeypatch):
    x = sample((3, 40, 64), 5, torch.bfloat16)
    b = sample((300, 4), 6, torch.float32)
    t, tb = x.to("tt"), b.to("tt")
    on = (t[..., 1::2].contiguous().cpu(), tb[:, 2].contiguous().cpu())
    monkeypatch.setenv("KTT_COLUMN_PROGRAMS", "0")
    off = (t[..., 1::2].contiguous().cpu(), tb[:, 2].contiguous().cpu())
    assert torch.equal(on[0], off[0]) and torch.equal(on[1], off[1])
    assert torch.equal(on[0], x[..., 1::2]) and torch.equal(on[1], b[:, 2])


@pytest.mark.hw
def test_capture_replay(native):
    xs = [sample((4, 64, 64), 10 + k, torch.bfloat16).to("tt") for k in range(3)]
    boxes = [sample((500, 4), 20 + k, torch.float32).to("tt") for k in range(3)]

    def step(x, b):
        return x[..., 0::2] * x[..., 1::2], b[:, 2] - b[:, 0]

    captured = kohakutt.capture(step, xs[0], boxes[0], warmup=2)
    for x, b in zip(xs, boxes, strict=True):
        got = captured(x, b)
        want = step(x.cpu(), b.cpu())
        assert torch.equal(got[0].cpu(), step(x, b)[0].cpu())
        torch.testing.assert_close(got[0].cpu(), want[0], rtol=1e-2, atol=1e-2)
        assert torch.equal(got[1].cpu(), step(x, b)[1].cpu())
    captured.release()


@pytest.mark.hw
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=str)
def test_upsample_and_shuffle_share_the_program_host(native, dtype):
    x = sample((2, 8, 20, 24), 12, dtype)
    t = x.to("tt")
    up = torch.nn.functional.interpolate(t, scale_factor=2, mode="nearest")
    assert torch.equal(up.cpu(), torch.nn.functional.interpolate(x, scale_factor=2))
    shuffled = torch.nn.functional.pixel_shuffle(t, 2)
    assert torch.equal(shuffled.cpu(), torch.nn.functional.pixel_shuffle(x, 2))
