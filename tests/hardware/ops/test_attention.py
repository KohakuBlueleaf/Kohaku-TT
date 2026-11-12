"""Regression: F.scaled_dot_product_attention on tt takes TT-NN flash attention for
supported shapes (one forward program, no host transfers, backward on device) and the
math path otherwise, or when sdpa_kernel disables the overrideable backend."""

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import kohakutt

OVERRIDEABLE, MATH = 4, 0


def transfers():
    """Host uploads + downloads so far."""
    c = kohakutt.counters()
    return c["uploads"] + c["downloads"]


@pytest.mark.hw
@pytest.mark.parametrize(
    "shape, causal, mask",
    [
        ((2, 4, 256, 256, 64), True, None),
        ((1, 2, 77, 77, 64), True, None),
        ((2, 2, 64, 64, 32), False, "float"),
    ],
)
def test_sdpa_fused(native, shape, causal, mask):
    b, h, sq, sk, d = shape
    q, k, v = (
        torch.randn(b, h, s, d).bfloat16().to("tt").requires_grad_()
        for s in (sq, sk, sk)
    )
    attn = None if mask is None else torch.randn(b, 1, sq, sk).bfloat16().to("tt")
    assert torch.ops.aten._fused_sdp_choice(q, k, v, attn, 0.0, causal) == OVERRIDEABLE

    # Warm-up builds the cached causal mask (one upload); later calls transfer nothing.
    warm = F.scaled_dot_product_attention(q, k, v, attn_mask=attn, is_causal=causal)
    warm.backward(torch.ones_like(warm))
    for t in (q, k, v):
        t.grad = None
    before_ops, before_xfer = kohakutt.counters()["ops"], transfers()
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn, is_causal=causal)
    assert kohakutt.counters()["ops"] - before_ops == 1
    out.backward(torch.ones_like(out))
    assert transfers() == before_xfer
    assert out.shape == (b, h, sq, d)
    assert all(t.grad.shape == t.shape for t in (q, k, v))


@pytest.mark.hw
def test_sdpa_math_path(native):
    """Dropout is not fused: the math route (head dim 48 is fused since the flash kernel
    pads head dims)."""
    q, k, v = (torch.randn(1, 2, 64, 48).bfloat16().to("tt") for _ in range(3))
    assert torch.ops.aten._fused_sdp_choice(q, k, v, None, 0.0, False) != MATH
    assert torch.ops.aten._fused_sdp_choice(q, k, v, None, 0.1, False) == MATH
    assert F.scaled_dot_product_attention(q, k, v, dropout_p=0.1).shape == q.shape


@pytest.mark.hw
def test_sdpa_kernel_selects_math(native):
    q, k, v = (torch.randn(1, 2, 64, 64).bfloat16().to("tt") for _ in range(3))
    with sdpa_kernel(SDPBackend.MATH):
        assert torch.ops.aten._fused_sdp_choice(q, k, v, None, 0.0, False) == MATH


@pytest.mark.hw
def test_masked_head_dim_512_ragged(native):
    """A bool causal mask at head dim 512 and a ragged length (flash's row-mask path;
    TT-NN's before flash took head dims above 256): padded key columns take no weight,
    it matches the is_causal=True form and float64 (compiled Gemma prefill)."""
    s, d = 77, 512
    g = torch.Generator().manual_seed(7)
    q, k, v = (torch.randn(1, 8, s, d, generator=g).bfloat16() for _ in range(3))
    keep = torch.ones(s, s, dtype=torch.bool).tril().view(1, 1, s, s)
    ref = F.scaled_dot_product_attention(
        q.double(), k.double(), v.double(), is_causal=True
    )
    qt, kt, vt = (x.to("tt") for x in (q, k, v))
    masked = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=keep.to("tt")).cpu()
    causal = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True).cpu()

    def rel(y, r):
        return ((y.double() - r).norm() / r.norm()).item()

    assert rel(causal, ref) <= 1.5e-2
    assert rel(masked, ref) <= 1.5e-2
    assert rel(masked, causal.double()) <= 1.5e-2
