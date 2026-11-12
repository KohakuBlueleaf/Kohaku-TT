"""A mask of one query row ([B or 1, H or 1, 1, Sk], bool or additive) stays one row on
ktt_flash: its reader clamps, scales and tail-masks the row in L1 (no conversion
programs, no host read). Any pattern and finite additive values, against float64 within
test_flash's bound; under trace capture too."""

import pytest
import torch
import torch.nn.functional as F

import kohakutt

BOUND = 8e-3  # test_flash.BOUND


def _rel(y, ref):
    return ((y.double() - ref).norm() / ref.norm()).item()


def _mask(kind, b, h, s, g):
    keep = torch.rand(b, h, 1, s, generator=g) > 0.3
    keep[..., 0] = True  # every row keeps a key
    if kind == "bool":
        return keep
    if kind == "inf":
        return torch.zeros(b, h, 1, s).masked_fill(~keep, float("-inf"))
    if kind == "finfo":
        return torch.zeros(b, h, 1, s).masked_fill(~keep, torch.finfo(torch.float).min)
    return torch.randn(b, h, 1, s, generator=g) * 3  # finite additive values


# (b, h, hk, sq, sk, d, mask batch, mask heads, kind)
CASES = [
    (2, 4, 4, 256, 256, 64, 2, 1, "bool"),
    (2, 8, 2, 197, 197, 64, 2, 1, "bool"),
    (2, 4, 4, 300, 300, 80, 2, 1, "inf"),
    (1, 4, 4, 33, 1000, 128, 1, 1, "finfo"),
    (2, 4, 4, 128, 160, 64, 1, 4, "values"),
    (3, 2, 2, 64, 77, 32, 3, 2, "values"),
]


@pytest.mark.parametrize("b,h,hk,sq,sk,d,mb,mh,kind", CASES)
def test_one_row_mask_on_flash(native, b, h, hk, sq, sk, d, mb, mh, kind):
    g = torch.Generator().manual_seed(sq + sk + d)
    q = torch.randn(b, h, sq, d, generator=g).bfloat16()
    k, v = (torch.randn(b, hk, sk, d, generator=g).bfloat16() for _ in range(2))
    m = _mask(kind, mb, mh, sk, g)
    if m.dtype != torch.bool:
        m = m.bfloat16()
    rep = h // hk
    ref = F.scaled_dot_product_attention(
        q.double(),
        k.repeat_interleave(rep, 1).double(),
        v.repeat_interleave(rep, 1).double(),
        attn_mask=m if m.dtype == torch.bool else m.double(),
    )
    before = kohakutt.counters()
    y = F.scaled_dot_product_attention(
        q.to("tt"), k.to("tt"), v.to("tt"), attn_mask=m.to("tt"), enable_gqa=rep > 1
    )
    torch.tt.synchronize()
    after = kohakutt.counters()
    assert after["attention_flash"] - before["attention_flash"] == 1
    assert after["attention_math"] == before["attention_math"]
    assert _rel(y.cpu(), ref) <= BOUND


def test_one_row_mask_under_capture(native):
    g = torch.Generator().manual_seed(1)
    q, k, v = (torch.randn(2, 4, 128, 64, generator=g).bfloat16() for _ in range(3))
    m = _mask("bool", 2, 1, 128, g)
    ins = [t.to("tt") for t in (q, k, v, m)]

    def fn(q, k, v, m):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=m)

    step = kohakutt.capture(fn, *ins)
    eager = fn(*ins).cpu()
    assert torch.equal(step(*ins).cpu(), eager)
    ref = F.scaled_dot_product_attention(
        q.double(), k.double(), v.double(), attn_mask=m
    )
    assert _rel(eager, ref) <= BOUND
