"""Regression: torch.stack(xs, -1) into a small last dim runs as one column-interleave
program stored Fold ([.., N, d] as its [.., N d] rows, [N d / w, w] for [N, d]), not
padded to 32 columns; whole [.., N, d] views of [.., N d] rows are Fold for free.
Elementwise ops, casts, fills and all-dims reductions keep Fold; other reads convert.
Values equal CPU (exactly for data movement), eager, written in place and replayed."""

import os

import pytest
import torch

import kohakutt

DTYPES = [torch.bfloat16, torch.float32]


def layout(t):
    return torch.ops.ktt._debug_layout(t)


def sample(shape, seed, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g).to(dtype)


def ops_of(fn):
    kohakutt.synchronize()
    before = kohakutt.counters()["ops"]
    out = fn()
    kohakutt.synchronize()
    return kohakutt.counters()["ops"] - before, out


def rotate_pairwise(x):
    v = x.view(*x.shape[:-1], -1, 2)
    a, b = v.unbind(-1)
    return torch.stack((-b, a), dim=-1).flatten(start_dim=-2)


@pytest.mark.hw
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize(
    "shape, k, fold",
    [
        ((9, 16, 64, 32), 2, True),
        ((2, 40, 50), 2, True),
        ((3, 64, 20), 3, True),
        ((2, 32, 33), 8, True),
        ((4, 6, 32), 2, False),  # few rows: the tile form is not 4x the Fold form
        ((100, 64), 2, True),
        ((2560,), 2, False),  # [N, 2]: TT-NN's concat, the tile form
    ],
    ids=["sam3", "odd_n", "k3", "k8", "few_rows", "rank2", "rank1"],
)
def test_stack_last_dim(native, dtype, shape, k, fold):
    xs = [sample(shape, 10 + i, dtype) for i in range(k)]
    ts = [x.to("tt") for x in xs]
    n, got = ops_of(lambda: torch.stack(ts, -1))
    assert torch.equal(got.cpu(), torch.stack(xs, -1))
    assert layout(got) == ("fold" if fold else "nchw")
    assert n <= 2 or len(shape) == 1, n
    cat = torch.cat([t.unsqueeze(-1) for t in ts], -1)
    assert torch.equal(cat.cpu(), torch.stack(xs, -1))


@pytest.mark.hw
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_rotate_pairwise(native, dtype):
    x = sample((9, 16, 64, 64), 2, dtype)
    t = x.to("tt")
    n, got = ops_of(lambda: rotate_pairwise(t))
    assert torch.equal(got.cpu(), rotate_pairwise(x))
    assert n <= 4, n


@pytest.mark.hw
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_fold_ops_and_views(native, dtype):
    a, b = sample((4, 64, 40), 3, dtype), sample((4, 64, 40), 4, dtype)
    s = torch.stack([a, b], -1)
    t = torch.stack([a.to("tt"), b.to("tt")], -1)
    assert layout(t) == "fold"
    e = torch.ops.ktt._debug_layout_expansions()
    # Elementwise ops and casts keep Fold, values as CPU's.
    assert layout(-t) == "fold" and torch.equal((-t).cpu(), -s)
    assert torch.equal((t * 2).cpu(), s * 2) and layout(t * 2) == "fold"
    assert torch.equal((t + t).cpu(), s + s) and layout(t + t) == "fold"
    other = sample(s.shape, 5, dtype)
    assert torch.equal((t * other.to("tt")).cpu(), s * other)
    to = torch.float32 if dtype == torch.bfloat16 else torch.bfloat16
    assert layout(t.to(to)) == "fold"
    assert torch.equal(t.to(to).cpu(), plain_cast(s, to, dtype))
    assert torch.equal(t.clone().cpu(), s)
    assert torch.ops.ktt._debug_layout_expansions() == e
    # Views read the stored rows: the flattened view is the stored tensor itself.
    n, flat = ops_of(lambda: t.flatten(-2))
    assert n == 0 and torch.equal(flat.cpu(), s.flatten(-2))
    assert torch.equal(t[..., 0].contiguous().cpu(), a)
    assert torch.equal(t[..., 1].contiguous().cpu(), b)
    assert torch.equal(t[1:3, 5:20].contiguous().cpu(), s[1:3, 5:20])
    assert torch.equal(t.transpose(-1, -2).contiguous().cpu(), s.transpose(-1, -2))
    assert torch.equal(t.reshape(4, 64, 80).cpu(), s.reshape(4, 64, 80))
    assert torch.equal(torch.cat([t, t], 0).cpu(), torch.cat([s, s], 0))
    # Reductions: over all dims as stored, others through the tile form.
    tol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    whole = s.double().sum()
    assert abs(t.sum().item() - whole.item()) <= tol * s.double().abs().sum().item()
    assert torch.equal(t.amax().cpu(), s.amax())
    torch.testing.assert_close(t.sum(-1).cpu(), s.sum(-1), rtol=tol, atol=tol)


def plain_cast(s, to, dtype):
    """The device cast of s in the tile form (the device rounding, not CPU's)."""
    return s.to("tt").to(to).cpu() if dtype != to else s


@pytest.mark.hw
def test_fold_in_place(native):
    a, b = sample((4, 64, 40), 6), sample((4, 64, 40), 7)
    t = torch.stack([a.to("tt"), b.to("tt")], -1)
    s = torch.stack([a, b], -1)
    t.mul_(2)
    s.mul_(2)
    assert layout(t) == "fold" and torch.equal(t.cpu(), s)
    t[..., 0] = 0.5
    s[..., 0] = 0.5
    assert torch.equal(t.cpu(), s)
    t[1] = 3.0
    s[1] = 3.0
    assert torch.equal(t.cpu(), s)
    t.copy_(torch.stack([b.to("tt"), a.to("tt")], -1))
    assert torch.equal(t.cpu(), torch.stack([b, a], -1))
    t.zero_()
    assert torch.equal(t.cpu(), torch.zeros_like(s)) and layout(t) == "fold"
    t.fill_(2.5)
    assert torch.equal(t.cpu(), torch.full_like(s, 2.5)) and layout(t) == "fold"
    m = torch.stack([a > 0, b > 0], -1)
    t.masked_fill_(m.to("tt"), -1.0)
    want = torch.full_like(s, 2.5).masked_fill_(m, -1.0)
    assert torch.equal(t.cpu(), want) and layout(t) == "fold"


@pytest.mark.hw
def test_fold_capture_replay(native):
    w = sample((9, 16, 64, 64), 8).to("tt")
    xs = [sample((9, 16, 64, 64), 20 + k).to("tt") for k in range(3)]

    def step(x):
        return rotate_pairwise(x) * w + x

    captured = kohakutt.capture(step, xs[0], warmup=2)
    for x in xs:
        want = rotate_pairwise(x.cpu()) * w.cpu() + x.cpu()
        assert torch.equal(captured(x).cpu(), step(x).cpu())
        torch.testing.assert_close(captured(x).cpu(), want, rtol=1e-2, atol=1e-2)
    captured.release()


@pytest.mark.hw
def test_knobs_keep_the_tile_form(native, monkeypatch):
    a, b = sample((4, 64, 40), 9), sample((4, 64, 40), 10)
    monkeypatch.setenv("KTT_FOLD", "0")
    t = torch.stack([a.to("tt"), b.to("tt")], -1)
    assert layout(t) == "nchw" and torch.equal(t.cpu(), torch.stack([a, b], -1))
    x = sample((3, 40, 64), 11).to("tt")
    assert torch.equal(rotate_pairwise(x).cpu(), rotate_pairwise(x.cpu()))
    assert layout(x.view(3, 40, 32, 2) * 2) == "nchw"


@pytest.mark.hw
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_boxes(native, dtype):
    """A [N, 4] boxes tensor (rank 2: kept in the tile form, where Fold measured
    slower): elementwise ops, members and reductions over the last dim as CPU's."""
    cols = [sample((5000,), 30 + i, dtype) for i in range(4)]
    s = torch.stack(cols, -1)
    t = torch.stack([c.to("tt") for c in cols], -1)
    assert layout(t) == "nchw"
    assert torch.equal(t.cpu(), s)
    scaled = t * 0.5 + 1.0
    torch.testing.assert_close(scaled.cpu(), s * 0.5 + 1.0, rtol=1e-2, atol=1e-2)
    area = (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])
    want = (s[:, 2] - s[:, 0]) * (s[:, 3] - s[:, 1])
    torch.testing.assert_close(area.cpu(), want, rtol=2e-2, atol=2e-2)
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(t.sum(-1).cpu(), s.sum(-1), rtol=tol, atol=tol)
    assert torch.equal(t.amax(-1).cpu(), s.amax(-1))
    assert torch.equal(t.amin(0).cpu(), s.amin(0))
    assert torch.equal(t.argmax(-1).cpu(), s.argmax(-1))


@pytest.mark.hw
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_views_of_wider_rows(native, dtype):
    """[.., N, 2] views of [.., 2N] tensors (view_as_real style) are Fold for free."""
    x = sample((4, 64, 80), 13, dtype)
    t = x.to("tt")
    v = t.view(4, 64, 40, 2)
    n, doubled = ops_of(lambda: v * 2)
    assert layout(doubled) == "fold" and n == 1, n
    assert torch.equal(doubled.cpu(), x.view(4, 64, 40, 2) * 2)
    assert torch.equal(doubled.flatten(-2).cpu(), x * 2)
    w = sample((4, 64, 40, 2), 14, dtype)
    tol = 1e-2 if dtype == torch.bfloat16 else 0.0
    torch.testing.assert_close(
        (v + w.to("tt")).cpu(), x.view(4, 64, 40, 2) + w, rtol=tol, atol=tol
    )
    c = torch.randn(2, 64, 40, dtype=torch.complex64)
    r = torch.view_as_real(c.to("tt"))
    torch.testing.assert_close((r * 3).cpu(), torch.view_as_real(c) * 3)
    back = torch.view_as_complex((r * 3).contiguous())
    torch.testing.assert_close(back.cpu(), c * 3)


@pytest.mark.hw
def test_writes_between_views_and_rows(native):
    x = sample((4, 64, 80), 15)
    base = x.to("tt")
    # [.., N, d] view written whole into a [.., N d] base.
    n, _ = ops_of(lambda: base.view(4, 64, 40, 2).mul_(2))
    assert torch.equal(base.cpu(), x * 2) and layout(base) == "nchw"
    assert n <= 1, n
    # [.., N d] rows written whole into a Fold base.
    a, b = sample((4, 64, 40), 16), sample((4, 64, 40), 17)
    f = torch.stack([a.to("tt"), b.to("tt")], -1)
    s = torch.stack([a, b], -1)
    f.view(4, 64, 80).add_(1.0)
    assert layout(f) == "fold"
    torch.testing.assert_close(f.cpu(), s + 1.0, rtol=1e-2, atol=1e-2)
    f.view(4, 64, 80).copy_(x.to("tt"))
    assert torch.equal(f.cpu(), x.view(4, 64, 40, 2)) and layout(f) == "fold"
    # Part of a Fold base through a rows view.
    f.view(4, 64, 80)[:, :, :10] = 7.0
    want = x.clone()
    want[:, :, :10] = 7.0
    assert torch.equal(f.cpu(), want.view(4, 64, 40, 2))


@pytest.mark.hw
def test_mixed_with_tile_operands(native):
    a, b = sample((4, 64, 40), 18), sample((4, 64, 40), 19)
    s = torch.stack([a, b], -1)
    f = torch.stack([a.to("tt"), b.to("tt")], -1)
    other = sample((4, 64, 40, 2), 20)
    o = other.to("tt")
    assert layout(o) == "nchw"
    mask = s > 0
    got = torch.where(mask.to("tt"), f, o)
    assert torch.equal(got.cpu(), torch.where(mask, s, other))
    assert torch.equal(torch.cat([f, o], 0).cpu(), torch.cat([s, other], 0))
    assert torch.equal(torch.cat([o, f], 1).cpu(), torch.cat([other, s], 1))
    idx = torch.tensor([3, 0])
    dst = o.clone()
    dst.index_copy_(0, idx.to("tt"), f[:2])
    want = other.clone()
    want.index_copy_(0, idx, s[:2])
    assert torch.equal(dst.cpu(), want)
    fd = f.clone()
    fd.index_copy_(0, idx.to("tt"), o[:2])
    want = s.clone()
    want.index_copy_(0, idx, other[:2])
    assert torch.equal(fd.cpu(), want)


@pytest.mark.hw
def test_state_takeover_with_fold_state(native):
    from kohakutt.torch_backend.trace import Captured

    def fold(seed):
        a, b = sample((4, 64, 40), seed), sample((4, 64, 40), seed + 1)
        return torch.stack([a.to("tt"), b.to("tt")], -1)

    def step(state, x):
        state.add_(x)
        return state * 2

    a, x = fold(40), fold(42)
    ref = a.clone()
    step(ref, x)  # the eager result of one call, on the device
    trace = Captured(step, [a, x], 1, static=[True, False])
    assert torch.equal(a.cpu(), ref.cpu())  # the prepare pass ran once
    b = fold(44)
    b_ref, a_now = b.clone(), a.cpu()
    out = trace(b, x)
    want = step(b_ref, x)
    assert torch.equal(b.cpu(), b_ref.cpu()) and torch.equal(a.cpu(), a_now)
    assert torch.equal(out.cpu(), want.cpu())
    trace(b, x)
    step(b_ref, x)
    assert torch.equal(b.cpu(), b_ref.cpu())
    assert layout(b) == "fold"
    trace.release()


@pytest.mark.hw
def test_parts_in_different_memory(native, monkeypatch):
    """Parts in DRAM and in L1 (KTT_L1_BUDGET) interleave correctly."""
    monkeypatch.setenv("KTT_L1_BUDGET", "0.3")
    a = sample((4, 64, 40), 50)
    ta = a.to("tt")  # an upload: DRAM
    tb = ta * 2  # an activation: L1 when it fits
    got = torch.stack([ta, tb, ta], -1)
    assert torch.equal(got.cpu(), torch.stack([a, (a * 2), a], -1))
    got = torch.stack([tb, ta], -1)
    assert torch.equal(got.cpu(), torch.stack([a * 2, a], -1))


@pytest.mark.hw
def test_where_with_a_fold_mask_and_int_operands(native):
    a, b = sample((4, 64, 40), 51), sample((4, 64, 40), 52)
    t = torch.stack([a.to("tt"), b.to("tt")], -1)
    m = t > 0
    idx = torch.arange(t.numel(), dtype=torch.int32).view(t.shape)
    got = torch.where(m, idx.to("tt"), idx.to("tt") + 1)
    want = torch.where(torch.stack([a, b], -1) > 0, idx, idx + 1)
    assert torch.equal(got.cpu(), want)


@pytest.mark.hw
def test_rank6_all_dims_reductions(native):
    xs = [sample((2, 3, 4, 64, 40), 53 + i, torch.float32) for i in range(2)]
    t = torch.stack([x.to("tt") for x in xs], -1)
    s = torch.stack(xs, -1)
    assert layout(t) == "fold" and t.dim() == 6
    torch.testing.assert_close(t.sum().cpu(), s.sum(), rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(t.mean().cpu(), s.mean(), rtol=1e-4, atol=1e-5)
    assert torch.equal(t.amax().cpu(), s.amax())


@pytest.mark.hw
def test_zero_size_parts(native):
    a = torch.empty(0, 5, 3, dtype=torch.bfloat16)
    got = torch.stack([a.to("tt"), a.to("tt")], -1)
    assert got.shape == (0, 5, 3, 2)
    b = torch.empty(2, 0, 3, dtype=torch.bfloat16)
    assert torch.stack([b.to("tt"), b.to("tt")], -1).shape == (2, 0, 3, 2)


@pytest.mark.hw
@pytest.mark.skipif(
    os.environ.get("KTT_CUSTOM_KERNELS") != "0",
    reason="needs KTT_CUSTOM_KERNELS=0 (read once per process)",
)
def test_custom_kernels_off(native):
    """KTT_CUSTOM_KERNELS=0: stack and strided reads take TT-NN routes, same values."""
    a, b = sample((4, 64, 40), 56), sample((4, 64, 40), 57)
    t = torch.stack([a.to("tt"), b.to("tt")], -1)
    assert layout(t) == "nchw" and torch.equal(t.cpu(), torch.stack([a, b], -1))
    x = sample((3, 40, 64), 58)
    assert torch.equal(x.to("tt")[..., 1::2].contiguous().cpu(), x[..., 1::2])
    p = sample((300, 4), 59)
    assert torch.equal(p.to("tt")[:, 2].contiguous().cpu(), p[:, 2])
