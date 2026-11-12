"""Regression: index_copy_ along dim -2 or a leading dim is one in-place program (the
static KV-cache write), equal to PyTorch's for indices in range (others are skipped);
other dims and views keep the selection route."""

import pytest
import torch

import kohakutt

DTYPES = [torch.bfloat16, torch.float32, torch.int32]


def sample(shape, seed, dtype):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g).mul(8)
    return x.round().to(dtype) if dtype == torch.int32 else x.to(dtype)


def counted(fn):
    """(ops, own programs) queued by fn(); a storage's first in-place write also
    queues one private copy of its base (an op)."""
    kohakutt.synchronize()
    before = kohakutt.counters()
    fn()
    kohakutt.synchronize()
    after = kohakutt.counters()
    return after["ops"] - before["ops"], after["programs"] - before["programs"]


@pytest.mark.hw
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape, steps",
    [((1, 2, 125, 64), [[77], [78], list(range(77))]), ((2, 3, 50, 40), [[49, 3]])],
    ids=["kv", "ragged"],
)
def test_rows_in_place(native, dtype, shape, steps):
    ref = sample(shape, 0, dtype)
    cache = ref.to("tt")
    alias = cache[:, :, :40]
    for i, step in enumerate(steps):
        idx = torch.tensor(step)
        src = sample((*shape[:2], len(step), shape[3]), i + 1, dtype)
        ref.index_copy_(2, idx, src)
        s, j = src.to("tt"), idx.to("tt")
        assert counted(lambda s=s, j=j: cache.index_copy_(2, j, s)) == (1 + (i == 0), 1)
        assert torch.equal(cache.cpu(), ref)
    assert torch.equal(alias.cpu(), ref[:, :, :40])


@pytest.mark.hw
def test_rows_write_leaves_other_tensors(native):
    a = torch.zeros(1, 2, 64, 64, dtype=torch.bfloat16, device="tt")
    b = torch.zeros(1, 2, 64, 64, dtype=torch.bfloat16, device="tt")
    c = a.view(2, 64, 64)
    src = sample((1, 2, 3, 64), 1, torch.bfloat16)
    idx = torch.tensor([5, 33, 63])
    a.index_copy_(2, idx.to("tt"), src.to("tt"))
    want = torch.zeros(1, 2, 64, 64, dtype=torch.bfloat16).index_copy_(2, idx, src)
    kept = a.clone()
    a.index_copy_(2, idx[:1].to("tt"), torch.ones(1, 2, 1, 64).bfloat16().to("tt"))
    assert torch.equal(kept.cpu(), want)
    want[:, :, 5] = 1
    assert torch.equal(a.cpu(), want)
    assert torch.equal(c.cpu(), want.view(2, 64, 64))
    assert not b.cpu().any()


@pytest.mark.hw
def test_rows_rank3_negative_dim_int32_index(native):
    ref = sample((6, 40, 33), 13, torch.float32)
    cache = ref.to("tt")
    idx = torch.tensor([39, 0, 17], dtype=torch.int32)
    src = sample((6, 3, 33), 14, torch.float32)[:, :, :]
    s, j = src.to("tt"), idx.to("tt")
    cache.index_copy_(-2, j, s)
    assert counted(lambda: cache.index_copy_(-2, j, s)) == (1, 1)
    assert torch.equal(cache.cpu(), ref.index_copy_(-2, idx.long(), src))


@pytest.mark.hw
def test_functional_out_view_and_strided_source(native):
    ref = sample((1, 2, 60, 64), 15, torch.bfloat16)
    idx = torch.tensor([3, 59])
    big = sample((1, 2, 2, 128), 16, torch.bfloat16)
    src = big[:, :, :, ::2]
    t, j, st = ref.to("tt"), idx.to("tt"), big.to("tt")[:, :, :, ::2]
    want = ref.index_copy(2, idx, src)
    assert torch.equal(torch.index_copy(t, 2, j, st).cpu(), want)
    out = torch.empty_like(t)
    torch.index_copy(t, 2, j, st, out=out)
    assert torch.equal(out.cpu(), want)
    assert torch.equal(t.cpu(), ref)
    part = t[:, :, :, :32]
    part.index_copy_(2, j, st[:, :, :, :32])
    want_part = ref.clone()
    want_part[:, :, :, :32].index_copy_(2, idx, src[:, :, :, :32])
    assert torch.equal(t.cpu(), want_part)


@pytest.mark.hw
def test_rows_index_out_of_range_skipped(native):
    ref = sample((1, 1, 40, 32), 17, torch.float32)
    cache = ref.to("tt")
    src = sample((1, 1, 2, 32), 18, torch.float32)
    cache.index_copy_(2, torch.tensor([200, 3]).to("tt"), src.to("tt"))
    ref[0, 0, 3] = src[0, 0, 1]
    assert torch.equal(cache.cpu(), ref)


@pytest.mark.hw
def test_rows_repeated_index_keeps_last(native):
    cache = torch.zeros(1, 1, 40, 32)
    src = sample((1, 1, 3, 32), 2, torch.float32)
    idx = torch.tensor([7, 7, 39])
    out = cache.to("tt").index_copy_(2, idx.to("tt"), src.to("tt")).cpu()
    want = cache.clone()
    want[0, 0, 7] = src[0, 0, 1]
    want[0, 0, 39] = src[0, 0, 2]
    assert torch.equal(out, want)


@pytest.mark.hw
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("seq, fused", [(1, False), (5, False), (5, True)])
def test_rows_from_head_view(native, dtype, seq, fused):
    heads, d = 2, 64
    width = heads * d * (3 if fused else 1)
    x = sample((1, seq, width), 3, dtype)
    ref = sample((1, heads, 90, d), 4, dtype)
    cache = ref.to("tt")
    xt = x.to("tt")
    at = heads * d if fused else 0

    def heads_of(t):
        return t[:, :, at : at + heads * d].view(1, seq, heads, d).transpose(1, 2)

    view = heads_of(xt)
    idx = torch.arange(seq) + 31
    ref.index_copy_(2, idx, heads_of(x))
    j = idx.to("tt")
    assert counted(lambda: cache.index_copy_(2, j, view)) == (2, 1)
    assert torch.equal(cache.cpu(), ref)


@pytest.mark.hw
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape, dim", [((6, 3, 40), 0), ((2, 5, 3, 33), 1)], ids=["dim0", "dim1"]
)
def test_planes_in_place(native, dtype, shape, dim):
    ref = sample(shape, 5, dtype)
    cache = ref.to("tt")
    idx = torch.tensor([4, 1])
    src_shape = list(shape)
    src_shape[dim] = 2
    src = sample(src_shape, 6, dtype)
    ref.index_copy_(dim, idx, src)
    s, j = src.to("tt"), idx.to("tt")
    assert counted(lambda: cache.index_copy_(dim, j, s)) == (2, 1)
    assert torch.equal(cache.cpu(), ref)


@pytest.mark.hw
def test_last_dim_keeps_selection_route(native):
    ref = sample((3, 20, 40), 7, torch.float32)
    idx = torch.tensor([0, 39, 5])
    src = sample((3, 20, 3), 8, torch.float32)
    out = ref.to("tt").index_copy_(-1, idx.to("tt"), src.to("tt")).cpu()
    assert torch.equal(out, ref.index_copy_(-1, idx, src))


@pytest.mark.hw
def test_rows_in_trace_read_position_at_replay(native):
    shape = (1, 2, 48, 64)
    ref = torch.zeros(shape, dtype=torch.bfloat16)
    cache = ref.to("tt")

    def step(pos, k):
        cache.index_copy_(2, pos, k)
        return cache * 1

    first = sample((1, 2, 1, 64), 0, torch.bfloat16)
    ref.index_copy_(2, torch.tensor([0]), first)
    captured = kohakutt.Captured(step, (torch.tensor([0]).to("tt"), first.to("tt")), 2)
    for p in [5, 31, 32, 47, 6]:
        k = sample((1, 2, 1, 64), p, torch.bfloat16)
        ref.index_copy_(2, torch.tensor([p]), k)
        out = captured(torch.tensor([p]).to("tt"), k.to("tt")).cpu()
        assert torch.equal(out, ref)
        assert torch.equal(cache.cpu(), ref)
    captured.release()


@pytest.mark.hw
def test_rows_rank2(native):
    ref = sample((40, 33), 19, torch.bfloat16)
    cache = ref.to("tt")
    idx = torch.tensor([39, 2])
    src = sample((2, 33), 20, torch.bfloat16)
    s, j = src.to("tt"), idx.to("tt")
    cache.index_copy_(0, j, s)
    assert counted(lambda: cache.index_copy_(0, j, s)) == (1, 1)
    assert torch.equal(cache.cpu(), ref.index_copy_(0, idx, src))


@pytest.mark.hw
def test_written_base_not_shared_by_op_results(native):
    ref = sample((1, 2, 64, 64), 21, torch.bfloat16)
    x = ref.to("tt")
    idx = torch.tensor([3]).to("tt")
    x.index_copy_(2, idx, torch.ones(1, 2, 1, 64).bfloat16().to("tt"))
    first = x.cpu()
    # New storages: each must keep its value when x is written in place again.
    copies = {
        "clone": x.clone(),
        "mul": x * 1,
        "add": x + 0,
        "cast twice": x.float().bfloat16(),
        "to copy": x.to(torch.bfloat16, copy=True),
        "cat": torch.cat([x]),
        "stack": torch.stack([x])[0],
        "transposed": x.transpose(2, 3).contiguous(),
        "copy_": torch.empty_like(x).copy_(x),
        "detach clone": x.detach().clone(),
        "round trip": x.cpu().to("tt"),
        "reshape clone": x.reshape(2, 64, 64).clone(),
        "slice": x[..., :8].contiguous(),
        "where": torch.where(torch.tensor(True).to("tt"), x, x),
        "repeat": x.repeat(1, 1, 1, 1),
        "pad": torch.nn.functional.pad(x, (0, 0)),
        "index_select": x.index_select(0, torch.tensor([0]).to("tt")),
        "masked_fill": x.masked_fill(torch.tensor(False).to("tt"), 0.0),
        "hstack": torch.hstack([x]),
    }
    # Same storage: each must see the write.
    views = {
        "detach": x.detach(),
        "view": x.view(2, 64, 64),
        "to same": x.to(torch.bfloat16),
        "contiguous": x.contiguous(),
    }
    x.index_copy_(2, idx, torch.full((1, 2, 1, 64), 2.0).bfloat16().to("tt"))
    second = first.clone()
    second[:, :, 3] = 2
    assert torch.equal(x.cpu(), second)
    for name, y in copies.items():
        want = first[..., :8] if name == "slice" else first
        got = y.cpu()
        got = got.transpose(2, 3) if name == "transposed" else got
        assert torch.equal(got.reshape(want.shape), want), name
    for name, y in views.items():
        assert torch.equal(y.cpu().reshape(second.shape), second), name
