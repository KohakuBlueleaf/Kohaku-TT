"""Head dims above 256 run on ktt_flash while its circular buffers fit L1 (Q K^T summed
over every head-dim tile in fp32 Dst; P V in column groups): plain, causal, GQA,
one-row and full masks against float64 within test_flash's bound; a head dim whose
buffers do not fit is refused with the reason; training keeps its route."""

import pytest
import torch
import torch.nn.functional as F

import kohakutt

BOUND = 8e-3  # test_flash.BOUND


def _rel(y, ref):
    return ((y.double() - ref).norm() / ref.norm()).item()


# (b, h, hk, s, d, causal, mask): VAE mid-block (1 head, 512), Gemma (512, 77 tokens),
# 384, and 1024 (above 512).
CASES = [
    (1, 1, 1, 1024, 512, False, None),
    (1, 8, 8, 77, 512, True, None),
    (1, 8, 2, 77, 512, False, "row"),
    (2, 4, 4, 160, 384, False, "key"),
    (1, 2, 2, 128, 1024, False, None),
]


@pytest.mark.parametrize("b,h,hk,s,d,causal,mask", CASES)
def test_wide_head_dim_on_flash(native, b, h, hk, s, d, causal, mask):
    g = torch.Generator().manual_seed(s + d)
    q = torch.randn(b, h, s, d, generator=g).bfloat16()
    k, v = (torch.randn(b, hk, s, d, generator=g).bfloat16() for _ in range(2))
    m = None
    if mask == "row":
        m = torch.ones(s, s, dtype=torch.bool).tril().view(1, 1, s, s)
    elif mask == "key":
        m = (torch.arange(s) < s - 29).view(1, 1, 1, s).expand(b, 1, 1, s)
    rep = h // hk
    ref = F.scaled_dot_product_attention(
        q.double(),
        k.repeat_interleave(rep, 1).double(),
        v.repeat_interleave(rep, 1).double(),
        attn_mask=m,
        is_causal=causal,
    )
    before = kohakutt.counters()
    y = F.scaled_dot_product_attention(
        q.to("tt"),
        k.to("tt"),
        v.to("tt"),
        attn_mask=None if m is None else m.to("tt"),
        is_causal=causal,
        enable_gqa=rep > 1,
    )
    torch.tt.synchronize()
    after = kohakutt.counters()
    assert after["attention_flash"] - before["attention_flash"] == 1
    assert _rel(y.cpu(), ref) <= BOUND


def test_head_dim_beyond_l1_refused(native):
    q, k, v = (torch.randn(1, 1, 64, 1536).bfloat16().to("tt") for _ in range(3))
    refused = pytest.raises(RuntimeError, match="do not fit L1")
    with refused, kohakutt.sdpa_kernel("flash"):
        F.scaled_dot_product_attention(q, k, v)


def test_wide_head_dim_gradients(native):
    """With gradients the call keeps its training route (flash's fused backward takes
    head dims up to 256): gradients against float64."""
    g = torch.Generator().manual_seed(4)
    q, k, v = (torch.randn(1, 2, 64, 512, generator=g).bfloat16() for _ in range(3))
    dout = torch.randn(q.shape, generator=g).bfloat16()
    ref = [t.double().requires_grad_() for t in (q, k, v)]
    F.scaled_dot_product_attention(*ref, is_causal=True).backward(dout.double())
    ins = [t.to("tt").requires_grad_() for t in (q, k, v)]
    F.scaled_dot_product_attention(*ins, is_causal=True).backward(dout.to("tt"))
    for name, x, r in zip(["dq", "dk", "dv"], ins, ref):
        assert _rel(x.grad.cpu(), r.grad) <= 2e-2, name
