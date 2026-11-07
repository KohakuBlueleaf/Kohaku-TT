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
    q, k, v = (torch.randn(1, 2, 64, 48).bfloat16().to("tt") for _ in range(3))
    assert torch.ops.aten._fused_sdp_choice(q, k, v, None, 0.0, False) == MATH
    assert F.scaled_dot_product_attention(q, k, v).shape == q.shape


@pytest.mark.hw
def test_sdpa_kernel_selects_math(native):
    q, k, v = (torch.randn(1, 2, 64, 64).bfloat16().to("tt") for _ in range(3))
    with sdpa_kernel(SDPBackend.MATH):
        assert torch.ops.aten._fused_sdp_choice(q, k, v, None, 0.0, False) == MATH
