"""Head dims that are not whole tiles (40, 72, 80, 100, 120, 200) run on ktt_flash, the
columns past the head dim zeroed in its L1: one program, error against float64 within
test_flash's bound; with gradients the forward stays on flash and the backward on the
math route."""

import pytest
import torch
import torch.nn.functional as F

import kohakutt

BOUND = 8e-3  # test_flash.BOUND
BW_BOUND = 2e-2  # test_flash.BW_BOUND

# (b, h, hk, s, d, causal, masked)
CASES = [
    (1, 4, 4, 256, 80, False, False),
    (2, 8, 8, 512, 72, False, False),
    (1, 4, 2, 300, 40, False, False),
    (1, 2, 2, 128, 100, False, False),
    (2, 4, 4, 256, 120, True, False),
    (1, 2, 2, 77, 200, False, False),
    (2, 4, 4, 300, 80, False, True),
    (1, 14, 14, 1024, 80, True, False),
]


def _rel(y, ref):
    return ((y.double() - ref).norm() / ref.norm()).item()


def _qkv(b, h, hk, s, d, seed):
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randn(b, n, s, d, generator=g).to(torch.bfloat16) for n in (h, hk, hk)
    ]


@pytest.mark.parametrize("b,h,hk,s,d,causal,masked", CASES)
def test_ragged_head_dim_on_flash(native, b, h, hk, s, d, causal, masked):
    q, k, v = _qkv(b, h, hk, s, d, s + d)
    rep = h // hk
    mask = None
    if masked:
        mask = (torch.arange(s) < s - 37)[None, None, None, :].expand(b, 1, 1, s)
    ref = F.scaled_dot_product_attention(
        q.double(),
        k.repeat_interleave(rep, 1).double(),
        v.repeat_interleave(rep, 1).double(),
        attn_mask=mask,
        is_causal=causal,
    )
    args = [t.to("tt") for t in (q, k, v)]
    mt = None if mask is None else mask.to("tt")
    torch.tt.synchronize()
    before = kohakutt.counters()
    y = F.scaled_dot_product_attention(
        *args, attn_mask=mt, is_causal=causal, enable_gqa=rep > 1
    )
    torch.tt.synchronize()
    after = kohakutt.counters()
    assert after["attention_flash"] - before["attention_flash"] == 1
    assert after["attention_math"] == before["attention_math"]
    y = y.cpu()
    assert y.shape == (b, h, s, d) and torch.isfinite(y).all()
    assert _rel(y, ref) <= BOUND


@pytest.mark.parametrize("causal", [False, True])
def test_ragged_head_dim_gradients(native, causal):
    """Forward on flash, backward on the math route: gradients against float64."""
    q, k, v = _qkv(1, 4, 4, 128, 80, 5)
    g = torch.Generator().manual_seed(6)
    dout = torch.randn(q.shape, generator=g).to(torch.bfloat16)
    ref = [x.double().requires_grad_() for x in (q, k, v)]
    F.scaled_dot_product_attention(*ref, is_causal=causal).backward(dout.double())
    got = [x.to("tt").detach().requires_grad_() for x in (q, k, v)]
    before = kohakutt.counters()
    y = F.scaled_dot_product_attention(*got, is_causal=causal)
    y.backward(dout.to("tt"))
    torch.tt.synchronize()
    after = kohakutt.counters()
    assert after["attention_flash"] - before["attention_flash"] == 1
    for name, x, r in zip(["dq", "dk", "dv"], got, ref):
        assert _rel(x.grad.cpu(), r.grad) <= BW_BOUND, name
