"""Regression: head views of a linear output (q / k / v, rotate_half halves, heads
merged back) take the same programs in a trace capture as in eager, and replays equal
eager, also when the base is written in place between two views in the captured call."""

import pytest
import torch
import torch.nn.functional as F

import kohakutt

B, T, H, D = 1, 128, 4, 64


def sample(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=g) / 4).bfloat16()


def rotate_half(x):
    return torch.cat((-x[..., D // 2 :], x[..., : D // 2]), -1)


def heads(t):
    return t.view(B, T, H, D).transpose(1, 2)


def block(x, wq, wk, wv, wo, cos, sin, write=False):
    pq = F.linear(x, wq)
    q = heads(pq)
    q = q * cos + rotate_half(q) * sin
    if write:
        pq.mul_(2.0)  # the base changes between two head views of it
        q = q + heads(pq)
    k = heads(F.linear(x, wk))
    k = k * cos + rotate_half(k) * sin
    v = heads(F.linear(x, wv))
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return F.linear(o.transpose(1, 2).reshape(B, T, H * D), wo)


def ops_of(fn, prepare):
    kohakutt.synchronize()
    before = kohakutt.counters()["ops"]
    if prepare:
        torch.ops.ktt.prepare_begin()
    try:
        fn()
    finally:
        if prepare:
            torch.ops.ktt.prepare_end()
    kohakutt.synchronize()
    return kohakutt.counters()["ops"] - before


@pytest.mark.hw
@pytest.mark.parametrize("write", [False, True])
def test_head_views_in_a_trace(native, write):
    ws = [sample((H * D, H * D), k).to("tt") for k in range(4)]
    angle = torch.rand(1, 1, T, D, generator=torch.Generator().manual_seed(9)) * 6
    cos, sin = angle.cos().bfloat16().to("tt"), angle.sin().bfloat16().to("tt")
    xs = [sample((B, T, H * D), 10 + k).to("tt") for k in range(3)]

    def step(x):
        with torch.no_grad():
            return block(x, *ws, cos, sin, write=write)

    want = [step(x).cpu() for x in xs]
    eager_ops = ops_of(lambda: step(xs[0]), prepare=False)
    traced_ops = ops_of(lambda: step(xs[0]), prepare=True)
    # A write in place under traces is copied into the existing buffer (extra ops).
    if not write:
        assert traced_ops <= eager_ops + 1, (eager_ops, traced_ops)
    captured = kohakutt.capture(step, xs[0], warmup=2)
    got = [captured(x).cpu() for x in xs]
    after = step(xs[1]).cpu()
    captured.release()
    for g, w in zip(got, want, strict=True):
        assert torch.equal(g, w)
    assert torch.equal(after, want[1])


def head_of(t, h=H):
    return t.view(t.shape[0], t.shape[1], h, -1).transpose(1, 2)


@pytest.mark.hw
def test_one_token_head_view_is_a_reshape(native):
    w = sample((H * D, H * D), 1).to("tt")
    x = sample((B, 1, H * D), 2).to("tt")
    with torch.no_grad():
        pq, pq2 = F.linear(x, w), F.linear(x, w)
        heads_ops = ops_of(lambda: head_of(pq) * 2, prepare=False)
        plain_ops = ops_of(lambda: pq2.view(B, H, 1, D) * 2, prepare=False)
        assert heads_ops == plain_ops, (heads_ops, plain_ops)
        assert torch.equal((head_of(pq) * 2).cpu(), head_of(pq.cpu()) * 2)
        # The halves of the same head view share its one copy (eager rotate_half).
        n = ops_of(
            lambda: torch.cat((-head_of(pq)[..., 32:], head_of(pq)[..., :32]), -1),
            prepare=False,
        )
        assert n <= 5, n


@pytest.mark.hw
def test_merged_heads_into_persistent_state(native):
    w = sample((H * D, H * D), 3).to("tt")
    xs = [sample((B, T, H * D), 20 + k).to("tt") for k in range(3)]
    state = sample((B, T, H, D), 25).to("tt")

    def step(x, s):
        with torch.no_grad():
            seen = s * 1
            q = head_of(F.linear(x, w))
            o = F.scaled_dot_product_attention(q, q, q)  # a whole [B, H, T, D]
            s.copy_(o.transpose(1, 2))
            return seen + s

    where = torch.ops.ktt._debug_buffer_range(state)
    # warmup=1: no eager call before the prepare pass, which rebinds nothing.
    captured = kohakutt.capture(lambda x: step(x, state), xs[0], warmup=1)
    assert torch.ops.ktt._debug_buffer_range(state) == where
    for x in xs:
        ref = state.clone()
        want = step(x, ref).cpu()
        assert torch.equal(captured(x).cpu(), want)
        assert torch.equal(state.cpu(), ref.cpu())
    assert torch.ops.ktt._debug_buffer_range(state) == where
    captured.release()


@pytest.mark.hw
def test_head_view_of_output_across_replays(native):
    w = sample((H * D, H * D), 4).to("tt")
    xs = [sample((B, T, H * D), 30 + k).to("tt") for k in range(3)]

    def step(x):
        with torch.no_grad():
            return F.linear(x, w)

    captured = kohakutt.capture(step, xs[0], warmup=2, copy_outputs=False)
    for x in xs:
        y = captured(x)
        got = (head_of(y) * 1).cpu()
        assert torch.equal(got, head_of(step(x).cpu()))
    captured.release()


@pytest.mark.hw
def test_head_view_of_persistent_buffer(native):
    kv = sample((B, T, H * D), 5).to("tt")
    xs = [sample((B, T, H * D), 40 + k).to("tt") for k in range(3)]

    def step(x, buf):
        with torch.no_grad():
            before = head_of(buf) * 1
            buf.add_(x)
            return before + head_of(buf)

    captured = kohakutt.capture(lambda x: step(x, kv), xs[0], warmup=2)
    for x in xs:
        ref = kv.clone()
        want = step(x, ref).cpu()
        assert torch.equal(captured(x).cpu(), want)
        assert torch.equal(kv.cpu(), ref.cpu())
    captured.release()


@pytest.mark.hw
def test_fused_qkv_head_split_in_a_trace(native):
    w3 = sample((3 * H * D, H * D), 6).to("tt")
    xs = [sample((B, T, H * D), 50 + k).to("tt") for k in range(3)]

    def step(x):
        with torch.no_grad():
            qkv = F.linear(x, w3).view(B, T, 3 * H, D)
            q, k, v = (qkv[:, :, i * H : (i + 1) * H].transpose(1, 2) for i in range(3))
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    want = [step(x).cpu() for x in xs]
    eager_ops = ops_of(lambda: step(xs[0]), prepare=False)
    traced_ops = ops_of(lambda: step(xs[0]), prepare=True)
    assert traced_ops <= eager_ops + 1, (eager_ops, traced_ops)
    captured = kohakutt.capture(step, xs[0], warmup=2)
    for x, w in zip(xs, want, strict=True):
        assert torch.equal(captured(x).cpu(), w)
    captured.release()


@pytest.mark.hw
def test_pass_head_caches_freed_after_release(native):
    kv = sample((B, T, H * D), 7).to("tt")
    x0 = sample((B, T, H * D), 8).to("tt")

    def step(x):
        with torch.no_grad():
            return head_of(kv) * head_of(x)

    kohakutt.synchronize()
    live = torch.ops.ktt.memory_stats()[0]
    captured = kohakutt.capture(step, x0, warmup=2)
    captured(x0)
    captured.release()
    del captured
    kohakutt.synchronize()
    # No head-major copy of kv made in the passes stays cached.
    assert torch.ops.ktt.memory_stats()[0] <= live


@pytest.mark.hw
def test_nested_capture_keeps_the_outer_one(native):
    w = sample((H * D, H * D), 9).to("tt")
    xs = [sample((B, T, H * D), 60 + k).to("tt") for k in range(2)]

    def inner(x):
        return x * 2

    def step(x):
        with torch.no_grad():
            with pytest.raises(RuntimeError, match="already active"):
                kohakutt.capture(inner, x, warmup=0)
            return head_of(F.linear(x, w)) * 3

    want = [(head_of(F.linear(x, w)) * 3).cpu() for x in xs]
    captured = kohakutt.capture(step, xs[0], warmup=0)
    for x, v in zip(xs, want, strict=True):
        assert torch.equal(captured(x).cpu(), v)
    captured.release()
