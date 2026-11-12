"""Unmasked attention with up to 32 queries (decode) runs on ktt_flash's key-lengths
variant, whose key count is a runtime value: a KV length that grows by one token per
call reuses one cached program (one program per call, no new cache entry, no compile),
results against float64 within test_flash's bound. Masked calls (a static cache's mask)
keep their route."""

import time

import pytest
import torch
import torch.nn.functional as F

import kohakutt

BOUND = 8e-3  # test_flash.BOUND


def _rel(y, ref):
    return ((y.double() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("hk,d,sq", [(8, 64, 1), (2, 128, 1), (8, 64, 16)])
def test_growing_kv_length(native, hk, d, sq):
    g = torch.Generator().manual_seed(d + hk)
    b, h = 2, 8
    k_all, v_all = (
        torch.randn(b, hk, 512, d, generator=g).bfloat16() for _ in range(2)
    )
    q = torch.randn(b, h, sq, d, generator=g).bfloat16()
    rep = h // hk
    qt = q.to("tt")
    times, deltas = [], []
    # Within a tile, across tile boundaries and across 32 x KB = 384 keys (a second
    # K / V block), all on one cached program.
    for length in (100, 101, 127, 128, 129, 383, 384, 385, 417, 500):
        k, v = k_all[:, :, :length], v_all[:, :, :length]
        kt, vt = k.to("tt"), v.to("tt")
        torch.tt.synchronize()
        before = kohakutt.counters()
        t = time.perf_counter()
        y = F.scaled_dot_product_attention(qt, kt, vt, enable_gqa=rep > 1)
        torch.tt.synchronize()
        times.append(time.perf_counter() - t)
        after = kohakutt.counters()
        deltas.append(
            {
                n: after[n] - before[n]
                for n in ("programs", "desc_hits", "kernel_compiles", "program_entries")
            }
        )
        ref = F.scaled_dot_product_attention(
            q.double(),
            k.repeat_interleave(rep, 1).double(),
            v.repeat_interleave(rep, 1).double(),
        )
        assert _rel(y.cpu(), ref) <= BOUND, length
    # The first length may compile; every later one reuses that build (a compile of
    # the flash kernels takes about a second).
    assert max(times[1:]) < 0.25, times
    # Every later length: one program, a cache hit, no new entry and no compile.
    want = {"programs": 1, "desc_hits": 1, "kernel_compiles": 0, "program_entries": 0}
    assert all(d == want for d in deltas[1:]), deltas


def _reference(q, k, v, causal=False):
    rep = q.shape[1] // k.shape[1]
    return F.scaled_dot_product_attention(
        q.double(),
        k.repeat_interleave(rep, 1).double(),
        v.repeat_interleave(rep, 1).double(),
        is_causal=causal,
    )


# (b, h, hk, sq, sk, causal): decode lanes longer than one core with more than one K / V
# block (the lane plan's rows-free cores), causal Sq != Sk on the key-lengths route, and
# Sq > Sk with Sk inside one key tile.
LANES = {
    "b1-h14-sk500": (1, 14, 14, 1, 500, False),
    "b2-h32-sk700": (2, 32, 8, 1, 700, False),
    "causal-sq16-sk200": (1, 8, 8, 16, 200, True),
    "causal-sq8-sk5": (1, 4, 4, 8, 5, True),
    "causal-sq64-sk20": (1, 4, 4, 64, 20, True),
}


@pytest.mark.parametrize("case", list(LANES))
def test_long_keys_and_rectangular_causal(native, case):
    b, h, hk, sq, sk, causal = LANES[case]
    g = torch.Generator().manual_seed(sk)
    q = torch.randn(b, h, sq, 64, generator=g).bfloat16()
    k, v = (torch.randn(b, hk, sk, 64, generator=g).bfloat16() for _ in range(2))
    y = F.scaled_dot_product_attention(
        q.to("tt"), k.to("tt"), v.to("tt"), is_causal=causal, enable_gqa=h != hk
    )
    assert _rel(y.cpu(), _reference(q, k, v, causal)) <= BOUND
    if sq <= 32:
        assert torch.ops.ktt._debug_sdpa_all_keys_built(b)


def test_explicit_ragged_lengths_long_keys(native):
    """kohakutt.sdpa with different lengths per batch past 32 x KB keys, B x H small
    (lanes of several cores)."""
    g = torch.Generator().manual_seed(3)
    b, h, sk, lengths = 2, 4, 700, [700, 333]
    q = torch.randn(b, h, 1, 64, generator=g).bfloat16()
    k, v = (torch.randn(b, h, sk, 64, generator=g).bfloat16() for _ in range(2))
    keep = (torch.arange(sk)[None, :] < torch.tensor(lengths)[:, None])[:, None, None]
    ref = F.scaled_dot_product_attention(
        q.double(), k.double(), v.double(), attn_mask=keep
    )
    y = kohakutt.sdpa(q.to("tt"), k.to("tt"), v.to("tt"), key_lengths=lengths)
    assert _rel(y.cpu(), ref) <= BOUND


def test_masked_decode_keeps_its_route(native):
    """A [B, 1, 1, Sk] cache mask goes to flash's broadcast-mask route, not the
    key-lengths variant (which builds its all-keys lengths for the batch size first):
    one flash call, results against float64. Batch 3 is used by no other test here."""
    g = torch.Generator().manual_seed(5)
    q = torch.randn(3, 8, 1, 64, generator=g).bfloat16()
    k, v = (torch.randn(3, 8, 256, 64, generator=g).bfloat16() for _ in range(2))
    m = (torch.arange(256) < 200).view(1, 1, 1, 256).expand(3, 1, 1, 256)
    ref = F.scaled_dot_product_attention(
        q.double(), k.double(), v.double(), attn_mask=m
    )
    before = kohakutt.counters()
    y = F.scaled_dot_product_attention(
        q.to("tt"), k.to("tt"), v.to("tt"), attn_mask=m.to("tt")
    )
    torch.tt.synchronize()
    after = kohakutt.counters()
    assert after["attention_flash"] - before["attention_flash"] == 1
    assert _rel(y.cpu(), ref) <= BOUND
    assert not torch.ops.ktt._debug_sdpa_all_keys_built(3)
    # The same call without the mask takes the key-lengths route.
    F.scaled_dot_product_attention(q.to("tt"), k.to("tt"), v.to("tt"))
    torch.tt.synchronize()
    assert torch.ops.ktt._debug_sdpa_all_keys_built(3)
