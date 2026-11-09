"""Kohaku-TT's flash attention (ktt_flash, the default bf16 route): error against CPU float64
over batch, MHA / GQA / MQA heads, ragged lengths, head dims 32-256, causal, and additive / bool
masks broadcast over batch and heads.

Bound: rel RMS <= 8e-3 against float64. At HiFi2 the matmul keeps 7 of Q's 8 significand bits
and P is bf16, which alone gives about 5.5e-3 on these inputs; CPU bf16 (fp32 math, bf16 output)
is about 1.7e-3.
"""

import pytest
import torch
import torch.nn.functional as F

BOUND = 8e-3

# (b, h, hk, s, d, causal)
CASES = [
    (1, 2, 2, 32, 32, False),
    (1, 2, 2, 256, 64, False),
    (2, 4, 4, 512, 64, True),
    (1, 4, 2, 96, 96, False),
    (3, 4, 1, 300, 128, False),
    (1, 4, 4, 1000, 64, False),
    (1, 4, 4, 1000, 64, True),
    (1, 8, 8, 1024, 128, True),
    (4, 8, 4, 640, 128, False),
    (1, 16, 4, 1024, 128, True),
    (8, 16, 16, 256, 64, False),
    (1, 2, 1, 1100, 256, False),
    (2, 2, 2, 96, 256, True),
]

# (b, h, hk, s, d, mask kind): row = [b, 1, s, s] additive, head = [b, h, s, s] additive,
# bcast = [b, 1, 1, s] additive, bool = [b, 1, s, s] keep mask.
MASK_CASES = [
    (2, 4, 4, 512, 64, "row"),
    (2, 4, 4, 256, 64, "head"),
    (2, 8, 2, 1024, 128, "bcast"),
    (2, 4, 4, 300, 64, "bool"),
]


def _rel(y, ref):
    return ((y.double() - ref).norm() / ref.norm()).item()


def _qkv(b, h, hk, s, d, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, h, s, d, generator=g).to(torch.bfloat16)
    k = torch.randn(b, hk, s, d, generator=g).to(torch.bfloat16)
    v = torch.randn(b, hk, s, d, generator=g).to(torch.bfloat16)
    return q, k, v, g


@pytest.mark.parametrize("b,h,hk,s,d,causal", CASES)
def test_matches_float64(native, b, h, hk, s, d, causal):
    q, k, v, _ = _qkv(b, h, hk, s, d, b + h + s + d)
    rep = h // hk
    kr, vr = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    ref = F.scaled_dot_product_attention(q.double(), kr.double(), vr.double(), is_causal=causal)
    y = F.scaled_dot_product_attention(
        q.to("tt"), k.to("tt"), v.to("tt"), is_causal=causal, enable_gqa=rep > 1
    ).cpu()
    assert y.shape == (b, h, s, d)
    assert torch.isfinite(y).all()
    assert _rel(y, ref) <= BOUND


@pytest.mark.parametrize("log2_logit", [10, 14, 17])
@pytest.mark.parametrize("s", [192, 201])
def test_large_logits(native, s, log2_logit):
    """Logits of about 2^log2_logit (dinov3 reaches about 1e5): the row reference jumps by far
    more than 127 between key chunks. The output is finite at every fidelity."""
    q, k, v, _ = _qkv(1, 4, 4, s, 64, s + log2_logit)
    m = 2.0 ** (log2_logit / 2)
    q, k = (q.float() * m).bfloat16(), (k.float() * m).bfloat16()
    for fidelity in ("HiFi4", "HiFi2", "LoFi"):
        with native.numerics(fidelity=fidelity):
            y = F.scaled_dot_product_attention(q.to("tt"), k.to("tt"), v.to("tt")).cpu()
        assert torch.isfinite(y).all(), fidelity


@pytest.mark.parametrize("b,h,hk,s,d,kind", MASK_CASES)
def test_masks(native, b, h, hk, s, d, kind):
    q, k, v, g = _qkv(b, h, hk, s, d, b + h + s + d + 1)
    keys = torch.arange(s)
    keep = torch.ones(b, 1, 1, s, dtype=torch.bool)
    for bi in range(b):
        # Even batches padded on the right, odd batches on the left (the first key block masked).
        keep[bi, ..., (keys >= int(0.7 * s)) if bi % 2 == 0 else (keys < int(0.4 * s))] = False
    if kind == "bool":
        mask = keep.expand(b, 1, s, s) & (torch.rand(b, 1, s, s, generator=g) >= 0.1)
        bias = torch.zeros(b, 1, s, s, dtype=torch.float64).masked_fill(~mask, float("-inf"))
    else:
        rows = 1 if kind == "bcast" else s
        heads = h if kind == "head" else 1
        mask = (0.5 * torch.randn(b, heads, rows, s, generator=g)).to(torch.bfloat16)
        mask = mask.masked_fill(~keep, float("-inf"))
        bias = mask.double()
    rep = h // hk
    kr, vr = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    ref = F.scaled_dot_product_attention(q.double(), kr.double(), vr.double(), attn_mask=bias)
    y = F.scaled_dot_product_attention(
        q.to("tt"), k.to("tt"), v.to("tt"), attn_mask=mask.to("tt"), enable_gqa=rep > 1
    ).cpu()
    assert torch.isfinite(y).all()
    assert _rel(y, ref) <= BOUND


APPROX_BOUND = 3e-2  # approximate exponential (TT-NN SDPA's approximate path: about 2.3e-2)

# (b, h, hk, s, d, causal, mask kind or None)
APPROX_CASES = [
    (1, 4, 4, 512, 64, False, None),
    (2, 4, 4, 512, 64, True, None),
    (2, 4, 4, 512, 64, False, "row"),
]


@pytest.mark.parametrize("b,h,hk,s,d,causal,kind", APPROX_CASES)
def test_approx_exp(native, b, h, hk, s, d, causal, kind):
    q, k, v, g = _qkv(b, h, hk, s, d, b + h + s + d + 2)
    mask = None
    bias = None
    if kind == "row":
        mask = (0.5 * torch.randn(b, 1, s, s, generator=g)).to(torch.bfloat16)
        mask[0, ..., int(0.7 * s):] = float("-inf")
        mask[1, ..., : int(0.4 * s)] = float("-inf")
        bias = mask.double()
    ref = F.scaled_dot_product_attention(q.double(), k.double(), v.double(), attn_mask=bias, is_causal=causal)
    with native.numerics(approx=True):
        y = F.scaled_dot_product_attention(
            q.to("tt"), k.to("tt"), v.to("tt"), attn_mask=None if mask is None else mask.to("tt"),
            is_causal=causal,
        ).cpu()
    assert torch.isfinite(y).all()
    assert _rel(y, ref) <= APPROX_BOUND


# (b, h): the most (batch, head) pairs of one query tile whose jobs fit a core's runtime args on
# the 110-core grid, then a count past it (today's route: TT-NN SDPA, bf16 Dst, approximate exp).
@pytest.mark.parametrize("b,h,bound", [(41, 110, BOUND), (44, 110, 5e-2)])
def test_many_pairs_short_queries(native, b, h, bound):
    q, k, v, _ = _qkv(b, h, h, 32, 32, b + h)
    ref = F.scaled_dot_product_attention(q.double(), k.double(), v.double())
    y = F.scaled_dot_product_attention(q.to("tt"), k.to("tt"), v.to("tt")).cpu()
    assert torch.isfinite(y).all()
    assert _rel(y, ref) <= bound


def _train(q, k, v, dout, device):
    """Output and gradients (dq, dk, dv) of attention trained on `device` ("cpu": float64)."""
    if device == "cpu":
        qq, kk, vv = (x.double().requires_grad_() for x in (q, k, v))
        y = F.scaled_dot_product_attention(qq, kk, vv)
        y.backward(dout.double())
    else:
        qq, kk, vv = (x.to(device).detach().requires_grad_() for x in (q, k, v))
        y = F.scaled_dot_product_attention(qq, kk, vv)
        y.backward(dout.to(device))
    return [t.detach().cpu() for t in (y, qq.grad, kk.grad, vv.grad)]


def test_later_kernels_see_default_constants(native, monkeypatch):
    """ktt_flash programs end with the vector unit's shared constant (LReg 11 = -1) restored: the
    attention training kernels of the KTT_FLASH=0 route, which read it, run right after them on
    the same cores and stay finite and within the bound of float64."""
    b, h, s, d = 2, 4, 256, 64
    q, k, v, g = _qkv(b, h, h, s, d, 19)
    dout = torch.randn(b, h, s, d, generator=g).to(torch.bfloat16)
    F.scaled_dot_product_attention(q.to("tt"), k.to("tt"), v.to("tt")).cpu()
    monkeypatch.setenv("KTT_FLASH", "0")
    ref = _train(q, k, v, dout, "cpu")
    got = _train(q, k, v, dout, "tt")
    for name, x, r in zip(("out", "dq", "dk", "dv"), got, ref, strict=True):
        assert torch.isfinite(x).all(), name
        assert _rel(x, r) <= BOUND, name


def test_no_mixing_across_pairs(native):
    """Every (batch, head) pair gets its own value scale: an output row taking another pair's keys
    or values lands far outside the bound."""
    b, h, s, d = 8, 16, 512, 64
    q, k, v, _ = _qkv(b, h, h, s, d, 7)
    scale = (1.0 + torch.arange(b * h, dtype=torch.float32)).reshape(b, h, 1, 1)
    v = (v.float() * scale).to(torch.bfloat16)
    ref = F.scaled_dot_product_attention(q.double(), k.double(), v.double())
    y = F.scaled_dot_product_attention(q.to("tt"), k.to("tt"), v.to("tt")).cpu()
    per_pair = ((y.double() - ref).flatten(2).norm(dim=2) / ref.flatten(2).norm(dim=2))
    assert per_pair.max().item() <= BOUND
