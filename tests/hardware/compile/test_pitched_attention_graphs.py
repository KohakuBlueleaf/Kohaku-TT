"""An SDXL-style attention block (q / k / v projections viewed as heads, SDPA, heads
merged back, output projection) under torch.compile (ktt_tt, ktt_tt_aot) and
kohakutt.capture: results equal eager; the capture takes no more device ops than eager
(4 linears + 1 SDPA, no head copies), also when it returns the strided SDPA output."""

import pytest
import torch
import torch.nn.functional as F

import kohakutt

B, T, TK, H, D = 2, 128, 77, 20, 64
C = H * D


def sample(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=g) / 8).bfloat16()


def block(x, ctx, wq, wk, wv, wo):
    q = F.linear(x, wq).view(B, -1, H, D).transpose(1, 2)
    k = F.linear(ctx, wk).view(B, -1, H, D).transpose(1, 2)
    v = F.linear(ctx, wv).view(B, -1, H, D).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v)
    return F.linear(o.transpose(1, 2).reshape(B, -1, C), wo)


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


def _rel(y, ref):
    return ((y.double() - ref.double()).norm() / ref.double().norm()).item()


@pytest.fixture
def weights(native):
    return [sample((C, C), k).to("tt") for k in range(4)]


@pytest.mark.hw
@pytest.mark.parametrize(
    "backend, cross", [("ktt_tt", False), ("ktt_tt", True), ("ktt_tt_aot", True)]
)
def test_compiled_block_matches_eager(native, weights, backend, cross):
    x = sample((B, T, C), 10).to("tt")
    ctx = sample((B, TK, C), 11).to("tt") if cross else x
    with torch.no_grad():
        want = block(x, ctx, *weights).cpu()
        compiled = torch.compile(block, backend=backend)
        got = compiled(x, ctx, *weights).cpu()
    assert _rel(got, want) <= 1e-2


@pytest.mark.hw
def test_captured_block_matches_eager(native, weights):
    xs = [sample((B, T, C), 20 + k).to("tt") for k in range(3)]

    def step(x):
        with torch.no_grad():
            return block(x, x, *weights)

    want = [step(x).cpu() for x in xs]
    eager_ops = ops_of(lambda: step(xs[0]), prepare=False)
    traced_ops = ops_of(lambda: step(xs[0]), prepare=True)
    assert eager_ops == 5, eager_ops
    assert traced_ops <= eager_ops, (eager_ops, traced_ops)
    captured = kohakutt.capture(step, xs[0], warmup=2)
    got = [captured(x).cpu() for x in xs]
    captured.release()
    for g, w in zip(got, want, strict=True):
        assert torch.equal(g, w)


@pytest.mark.hw
def test_capture_returning_the_strided_attention_output(native, weights):
    xs = [sample((B, T, C), 30 + k).to("tt") for k in range(3)]
    wq, wk, wv = weights[:3]

    def step(x):
        with torch.no_grad():
            q, k, v = (
                F.linear(x, w).view(B, -1, H, D).transpose(1, 2) for w in (wq, wk, wv)
            )
            return F.scaled_dot_product_attention(q, k, v)

    eager = [step(x) for x in xs]
    assert not eager[0].is_contiguous()  # the in-place route's strided output
    want = [y.cpu() for y in eager]
    captured = kohakutt.capture(step, xs[0], warmup=2)
    got = [captured(x) for x in xs]
    replays = [(y.stride(), y.cpu()) for y in got]
    captured.release()
    for (stride, g), w in zip(replays, want, strict=True):
        assert stride == eager[0].stride()  # the replay keeps the strided form
        assert torch.equal(g, w)
