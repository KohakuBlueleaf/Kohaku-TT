"""fp32 q / k / v without gradients run on ktt_flash, cast to bf16 at entry and the
output back to fp32 (with gradients: the math route); causal attention with Sq != Sk
(PyTorch's top-left form) runs as one flash call. Errors against float64 within
test_flash's bounds."""

import pytest
import torch
import torch.nn.functional as F

import kohakutt

BOUND = 8e-3  # test_flash.BOUND

# (b, h, hk, s, d, causal, masked)
CASES = [
    (1, 4, 4, 256, 64, False, False),
    (2, 8, 2, 300, 128, True, False),
    (2, 4, 4, 197, 80, False, True),
]


def _rel(y, ref):
    return ((y.double() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("b,h,hk,s,d,causal,masked", CASES)
def test_fp32_on_flash(native, b, h, hk, s, d, causal, masked):
    g = torch.Generator().manual_seed(s + d)
    q, k, v = (torch.randn(b, n, s, d, generator=g) for n in (h, hk, hk))
    rep = h // hk
    mask = None
    if masked:
        mask = torch.randn(b, 1, s, s, generator=g)
    ref = F.scaled_dot_product_attention(
        q.double(),
        k.repeat_interleave(rep, 1).double(),
        v.repeat_interleave(rep, 1).double(),
        attn_mask=None if mask is None else mask.double(),
        is_causal=causal,
    )
    args = [t.to("tt") for t in (q, k, v)]
    mt = None if mask is None else mask.to("tt")
    before = kohakutt.counters()
    y = F.scaled_dot_product_attention(
        *args, attn_mask=mt, is_causal=causal, enable_gqa=rep > 1
    )
    torch.tt.synchronize()
    after = kohakutt.counters()
    assert after["attention_flash"] - before["attention_flash"] == 1
    assert after["attention_math"] == before["attention_math"]
    assert y.dtype == torch.float32
    assert _rel(y.cpu(), ref) <= BOUND


def test_fp32_with_gradients_keeps_math(native):
    g = torch.Generator().manual_seed(3)
    q, k, v = (torch.randn(1, 2, 64, 64, generator=g) for _ in range(3))
    ins = [t.to("tt").requires_grad_() for t in (q, k, v)]
    before = kohakutt.counters()
    y = F.scaled_dot_product_attention(*ins)
    y.sum().backward()
    torch.tt.synchronize()
    after = kohakutt.counters()
    assert after["attention_flash"] == before["attention_flash"]
    assert after["attention_math"] - before["attention_math"] == 1
    ref = [t.double().requires_grad_() for t in (q, k, v)]
    F.scaled_dot_product_attention(*ref).sum().backward()
    for x, r in zip(ins, ref):
        assert _rel(x.grad.cpu(), r.grad) <= 2e-2


# (b, h, hk, sq, sk, d): queries fewer than keys and more (rows past Sk see every key;
# 300 / 290: the diagonal in the ragged last key tile). One flash call each.
CAUSAL = [
    (1, 4, 4, 1, 77, 64),
    (2, 8, 2, 64, 300, 64),
    (1, 4, 4, 300, 128, 80),
    (1, 8, 8, 1024, 512, 64),
    (1, 2, 2, 300, 290, 64),
    (1, 2, 2, 290, 300, 64),
]


@pytest.mark.parametrize("b,h,hk,sq,sk,d", CAUSAL)
def test_causal_queries_and_keys_differ(native, b, h, hk, sq, sk, d):
    g = torch.Generator().manual_seed(sq + sk)
    q = torch.randn(b, h, sq, d, generator=g).bfloat16()
    k, v = (torch.randn(b, hk, sk, d, generator=g).bfloat16() for _ in range(2))
    rep = h // hk
    ref = F.scaled_dot_product_attention(
        q.double(),
        k.repeat_interleave(rep, 1).double(),
        v.repeat_interleave(rep, 1).double(),
        is_causal=True,
    )
    before = kohakutt.counters()
    y = F.scaled_dot_product_attention(
        q.to("tt"), k.to("tt"), v.to("tt"), is_causal=True, enable_gqa=rep > 1
    )
    torch.tt.synchronize()
    after = kohakutt.counters()
    assert after["attention_flash"] - before["attention_flash"] == 1
    assert after["attention_math"] == before["attention_math"]
    assert _rel(y.cpu(), ref) <= BOUND


def test_causal_queries_and_keys_differ_gradients(native):
    """Forward on flash, backward on the math route (top-left causal mask)."""
    g = torch.Generator().manual_seed(9)
    q = torch.randn(1, 2, 64, 64, generator=g).bfloat16()
    k, v = (torch.randn(1, 2, 160, 64, generator=g).bfloat16() for _ in range(2))
    dout = torch.randn(q.shape, generator=g).bfloat16()
    ref = [t.double().requires_grad_() for t in (q, k, v)]
    F.scaled_dot_product_attention(*ref, is_causal=True).backward(dout.double())
    ins = [t.to("tt").requires_grad_() for t in (q, k, v)]
    F.scaled_dot_product_attention(*ins, is_causal=True).backward(dout.to("tt"))
    for name, x, r in zip(["dq", "dk", "dv"], ins, ref):
        assert _rel(x.grad.cpu(), r.grad) <= 2e-2, name
