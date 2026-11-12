"""Head views of [B, T, C] projections (separate q / k / v, or one fused qkv) reach
ktt_flash read in place: one device op per attention call, no head copies, the output a
[B, H, T, D] view of [B, T, H D] (q's dimension order). Masked calls, gradients, head
dims off the tile and views head_view refuses keep their routes (a dense output)."""

import pytest
import torch
import torch.nn.functional as F

import kohakutt

BOUND = 8e-3  # test_flash.BOUND


def _rel(y, ref):
    return ((y.double() - ref).norm() / ref.norm()).item()


def _separate(g, b, t, tk, h, d, hk=None):
    hk = hk or h
    q = torch.randn(b, t, h * d, generator=g).bfloat16()
    k, v = (torch.randn(b, tk, hk * d, generator=g).bfloat16() for _ in range(2))

    def views(q, k, v):
        return [
            x.view(b, -1, n, d).transpose(1, 2)
            for x, n in zip((q, k, v), (h, hk, hk), strict=True)
        ]

    return (q, k, v), views


def _columns(g, b, t, h, d):
    """q at column 32 of a wider base (a tile, not a head, off its start); k the second
    half of a [B, T, 2 H D] base; v its own [B, T, H D] base (k and v of different C).
    """
    qb = torch.randn(b, t, h * d + 64, generator=g).bfloat16()
    kb = torch.randn(b, t, 2 * h * d, generator=g).bfloat16()
    vb = torch.randn(b, t, h * d, generator=g).bfloat16()

    def views(qb, kb, vb):
        cut = ((qb, 32), (kb, h * d), (vb, 0))
        return [x[..., c : c + h * d].view(b, t, h, d).transpose(1, 2) for x, c in cut]

    return (qb, kb, vb), views


def _fused(g, b, t, h, d):
    qkv = torch.randn(b, t, 3 * h * d, generator=g).bfloat16()

    def views(qkv):
        return list(qkv.reshape(b, t, 3, h, d).permute(2, 0, 3, 1, 4).unbind(0))

    return (qkv,), views


# name: (inputs and their views, SDPA keywords)
CASES = {
    "separate-20h": (lambda g: _separate(g, 2, 128, 128, 20, 64), {}),
    "cross-77": (lambda g: _separate(g, 1, 256, 77, 10, 64), {}),
    "fused-300": (lambda g: _fused(g, 1, 300, 4, 128), {}),
    "ragged-b3-cross": (lambda g: _separate(g, 3, 77, 100, 4, 64), {}),
    "gqa-8-2": (lambda g: _separate(g, 2, 96, 96, 8, 64, hk=2), {"enable_gqa": True}),
    "causal-fused-300": (lambda g: _fused(g, 1, 300, 4, 128), {"is_causal": True}),
    "causal-96-200": (lambda g: _separate(g, 2, 96, 200, 4, 64), {"is_causal": True}),
    "columns": (lambda g: _columns(g, 2, 64, 4, 64), {}),
}


@pytest.mark.parametrize("case", list(CASES))
def test_head_views_read_in_place(native, case):
    g = torch.Generator().manual_seed(11)
    build, kw = CASES[case]
    host, views = build(g)
    ref = F.scaled_dot_product_attention(*[x.double() for x in views(*host)], **kw)
    dev = [x.to("tt") for x in host]
    torch.tt.synchronize()
    before = kohakutt.counters()
    y = F.scaled_dot_product_attention(*views(*dev), **kw)
    torch.tt.synchronize()
    after = kohakutt.counters()
    assert after["ops"] - before["ops"] == 1
    assert after["attention_flash"] - before["attention_flash"] == 1
    b, h, t, d = y.shape
    assert y.stride() == (t * h * d, d, h * d, 1)
    # The strides PyTorch's meta gives the overrideable op for the same views.
    meta = torch.ops.aten._scaled_dot_product_fused_attention_overrideable(
        *views(*[x.to("meta") for x in host]), is_causal=kw.get("is_causal", False)
    )[0]
    assert y.stride() == meta.stride()
    merged = y.transpose(1, 2).reshape(b, t, h * d)
    before = kohakutt.counters()
    host_merged = merged.cpu()
    assert kohakutt.counters()["ops"] - before["ops"] == 0
    assert _rel(host_merged, ref.transpose(1, 2).reshape(b, t, h * d)) <= BOUND
    assert _rel(y.cpu(), ref) <= BOUND


def test_masked_views_keep_their_route(native):
    g = torch.Generator().manual_seed(12)
    host, views = _separate(g, 2, 96, 96, 4, 64)
    m = torch.arange(96).view(1, 1, 1, 96) < torch.tensor([60, 96]).view(2, 1, 1, 1)
    ref = F.scaled_dot_product_attention(
        *[x.double() for x in views(*host)], attn_mask=m
    )
    y = F.scaled_dot_product_attention(
        *views(*[x.to("tt") for x in host]), attn_mask=m.to("tt")
    )
    assert y.is_contiguous()  # the dense route, not the in-place one
    assert _rel(y.cpu(), ref) <= BOUND


@pytest.mark.parametrize("d", [64, 80])
def test_gradients_and_off_tile_head_dims_keep_their_route(native, d):
    g = torch.Generator().manual_seed(d)
    host, views = _fused(g, 1, 128, 2, d)
    dout = torch.randn(1, 2, 128, d, generator=g).bfloat16()
    ref_in = host[0].double().requires_grad_()
    F.scaled_dot_product_attention(*views(ref_in)).backward(dout.double())
    x = host[0].to("tt").requires_grad_()
    y = F.scaled_dot_product_attention(*views(x))
    assert y.is_contiguous()  # the dense route, not the in-place one
    y.backward(dout.to("tt"))
    assert (
        _rel(y.detach().cpu(), F.scaled_dot_product_attention(*views(host[0].double())))
        <= BOUND
    )
    assert _rel(x.grad.cpu(), ref_in.grad) <= 2.5e-2


def _refused(kind, g):
    """Views head_view refuses, as (inputs, views, SDPA keywords)."""
    b, t, h, d = 2, 96, 4, 64
    if kind == "fp32":
        host, views = _separate(g, b, t, t, h, d)
        return tuple(x.float() for x in host), views, {}
    host, views = _separate(g, b, t, t, h, d)
    if kind == "slice-s":
        return host, lambda q, k, v: [x[:, :, : t // 2] for x in views(q, k, v)], {}
    if kind == "slice-b":
        return host, lambda q, k, v: [x[1:] for x in views(q, k, v)], {}
    # expanded heads: k / v of one head expanded over the four
    kb = torch.randn(b, t, d, generator=g).bfloat16()
    vb = torch.randn(b, t, d, generator=g).bfloat16()

    def expanded(q, k, v):
        one = [x.view(b, t, 1, d).transpose(1, 2).expand(b, h, t, d) for x in (k, v)]
        return [q.view(b, t, h, d).transpose(1, 2), *one]

    return (host[0], kb, vb), expanded, {}


@pytest.mark.parametrize("kind", ["slice-s", "slice-b", "expanded", "fp32"])
def test_refused_views_keep_their_route(native, kind):
    g = torch.Generator().manual_seed(13)
    host, views, kw = _refused(kind, g)
    ref = F.scaled_dot_product_attention(*[x.double() for x in views(*host)], **kw)
    y = F.scaled_dot_product_attention(*views(*[x.to("tt") for x in host]), **kw)
    assert y.is_contiguous()  # the dense route, not the in-place one
    assert _rel(y.cpu(), ref) <= BOUND


def test_writes_into_the_output_and_a_second_attention(native):
    g = torch.Generator().manual_seed(14)
    host, views = _separate(g, 2, 128, 128, 4, 64)
    ref = F.scaled_dot_product_attention(*[x.double() for x in views(*host)])
    dev = [x.to("tt") for x in host]
    y = F.scaled_dot_product_attention(*views(*dev))
    assert not y.is_contiguous()  # the in-place route's strided output
    y.mul_(2)
    y[:, 1].add_(1)
    ref = ref * 2
    ref[:, 1] += 1
    assert _rel(y.cpu(), ref) <= BOUND
    # The written output read in place as the next attention's q.
    k, v = views(*dev)[1:]
    torch.tt.synchronize()
    before = kohakutt.counters()["ops"]
    z = F.scaled_dot_product_attention(y, k, v)
    torch.tt.synchronize()
    assert kohakutt.counters()["ops"] - before == 1
    assert not z.is_contiguous()
    hk, hv = [x.double() for x in views(*host)[1:]]
    want = F.scaled_dot_product_attention(y.cpu().double(), hk, hv)
    assert _rel(z.cpu(), want) <= BOUND
    assert _rel(y.contiguous().cpu(), ref) <= BOUND
