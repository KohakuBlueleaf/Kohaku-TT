"""kohakutt.sdpa with key_lengths against float64 attention with the bool key mask
(forward, gradients, causal, GQA; rel RMS <= 8e-3 / 2.5e-2 as test_flash.py), with
garbage past every length and in the tile padding, and one program for every set of
lengths."""

import pytest
import torch
import torch.nn.functional as F

BOUND = 8e-3
GRAD_BOUND = 2.5e-2
# RMS of dq, dk of a one-key sequence (exactly 0 in float64), q / k / v / dO ~ N(0, 1).
ONE_KEY_RMS = 1.5e-2

# (B, H, Hk, S, D, lengths, causal)
CASES = {
    "33": (4, 2, 2, 33, 32, [1, 32, 33, 17], False),
    "33-causal": (4, 2, 2, 33, 32, [1, 32, 33, 17], True),
    "197": (2, 4, 2, 197, 64, [197, 64], False),
    "70-causal-gqa": (4, 4, 2, 70, 64, [1, 33, 70, 64], True),
    "128": (3, 2, 2, 128, 64, [96, 1, 128], False),
}


def _values(shape, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=gen).to(torch.bfloat16)


def _inputs(case, seed=0):
    b, h, hk, s, d, _, _ = CASES[case]
    return [
        _values((b, n, s, d), seed + i) for i, n in enumerate((h, hk, hk, h))
    ]  # q, k, v, output gradient


def _reference(q, k, v, g, lengths, causal):
    """Float64 attention with the bool key mask, and its dq, dk, dv."""
    s = k.size(2)
    keep = torch.arange(s)[None, :] < torch.tensor(lengths)[:, None]
    mask = keep[:, None, None, :].expand(q.size(0), 1, q.size(2), s)
    if causal:
        mask = mask & torch.ones(q.size(2), s, dtype=torch.bool).tril()
    rep = q.size(1) // k.size(1)
    ins = [q.double()] + [t.double().repeat_interleave(rep, 1) for t in (k, v)]
    ins = [t.requires_grad_(True) for t in ins]
    out = F.scaled_dot_product_attention(*ins, attn_mask=mask)
    out.backward(g.double())
    fold = [ins[0].grad] + [
        t.grad.unflatten(1, (k.size(1), rep)).sum(2) for t in ins[1:]
    ]
    return [out.detach()] + fold


def _device(native, q, k, v, g, lengths, causal, grads=True):
    ins = [t.detach().requires_grad_(grads) for t in (q, k, v)]
    out = native.sdpa(*ins, key_lengths=lengths, is_causal=causal)
    if not grads:
        return [out.cpu()]
    got = torch.autograd.grad(out, ins, g)
    return [out.detach().cpu()] + [t.cpu() for t in got]


def _rel(y, ref):
    return ((y.double() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("case", list(CASES))
def test_matches_masked_attention(native, case):
    """Output and dq, dk, dv against float64 attention with the bool key mask; dk and
    dv are exactly 0 past each length."""
    lengths, causal = CASES[case][5], CASES[case][6]
    q, k, v, g = _inputs(case)
    want = _reference(q, k, v, g, lengths, causal)
    got = _device(native, *(t.to("tt") for t in (q, k, v, g)), lengths, causal)
    for y in got:
        assert torch.isfinite(y).all()
    assert _rel(got[0], want[0]) <= BOUND
    for b, n in enumerate(lengths):
        for name, y, ref in zip(["dq", "dk", "dv"], got[1:], want[1:]):
            if n == 1 and name != "dv":
                # One key: dq, dk are exactly 0; the kernel's D = rowsum(dO * O)
                # from the bf16 output leaves bf16-level values (RMS ~7e-3).
                rms = y[b].double().pow(2).mean().sqrt().item()
                assert rms <= ONE_KEY_RMS, (name, b, rms)
            else:
                assert _rel(y[b], ref[b]) <= GRAD_BOUND, (name, b, _rel(y[b], ref[b]))
        assert not got[2][b, :, n:].any() and not got[3][b, :, n:].any(), b


def _dirty(x, lengths=None):
    """x on tt with inf in its tile padding and (with `lengths`) at every key position
    at or past the length of its batch."""
    *lead, r, c = x.shape
    big = torch.full(
        (*lead, (r + 31) // 32 * 32 + 32, (c + 31) // 32 * 32 + 32),
        float("inf"),
        dtype=x.dtype,
    )
    big[..., :r, :c] = x
    if lengths is not None:
        for b, n in enumerate(lengths):
            big[b, :, n:r, :c] = float("inf")
    return big.to("tt")[..., :r, :c]


@pytest.mark.parametrize("case", list(CASES))
def test_garbage_past_lengths(native, case):
    """inf in k and v at and past every length and in the tile padding of q, k, v and
    the output gradient: output and gradients equal the clean run bit for bit."""
    lengths, causal = CASES[case][5], CASES[case][6]
    q, k, v, g = _inputs(case)
    clean = _device(native, *(t.to("tt") for t in (q, k, v, g)), lengths, causal)
    dirty = _device(
        native,
        _dirty(q),
        _dirty(k, lengths),
        _dirty(v, lengths),
        _dirty(g),
        lengths,
        causal,
    )
    for name, c, d in zip(["out", "dq", "dk", "dv"], clean, dirty):
        assert torch.equal(c, d), name


@pytest.mark.parametrize("causal", [False, True])
def test_full_lengths_equal_plain_attention(native, causal):
    """Every length at Sk: bit-identical to F.scaled_dot_product_attention."""
    q, k, v, g = (_values((2, 3, 70, 64), i) for i in range(4))
    tt = [t.to("tt") for t in (q, k, v, g)]
    got = _device(native, *tt, [70, 70], causal)
    ins = [t.detach().requires_grad_(True) for t in tt[:3]]
    out = F.scaled_dot_product_attention(*ins, is_causal=causal)
    want = [out.detach().cpu()] + [
        t.cpu() for t in torch.autograd.grad(out, ins, tt[3])
    ]
    for name, y, w in zip(["out", "dq", "dk", "dv"], got, want):
        assert torch.equal(y, w), name


def test_new_lengths_compile_nothing(native):
    """A new set of lengths, as a list or a tt tensor, runs the same program: no kernel
    compile, a descriptor-cache hit, and a tt tensor of lengths is not downloaded."""
    q, k, v = (_values((3, 2, 100, 64), i).to("tt") for i in range(3))
    with torch.no_grad():
        native.sdpa(q, k, v, key_lengths=[100, 50, 7])
        native.synchronize()
        before = native.counters()
        native.sdpa(q, k, v, key_lengths=[3, 99, 64])
        native.sdpa(q, k, v, key_lengths=torch.tensor([1, 2, 3]).to("tt"))
        native.synchronize()
        after = native.counters()
    assert after["kernel_compiles"] == before["kernel_compiles"]
    assert after["programs"] - before["programs"] == 2
    assert after["desc_hits"] - before["desc_hits"] == 2
    assert after["downloads"] == before["downloads"]


def test_head_dim_off_the_tile(native):
    """A head dim ktt_flash does not take runs the explicit-mask route, same result."""
    q, k, v = (_values((2, 2, 40, 40), i) for i in range(3))
    want = _reference(q, k, v, torch.zeros_like(q), [40, 9], False)[0]
    with torch.no_grad():
        got = native.sdpa(q.to("tt"), k.to("tt"), v.to("tt"), key_lengths=[40, 9])
    assert _rel(got.cpu(), want) <= 2e-2


@pytest.mark.parametrize("lengths", [[0, 5], [5, 41], [5]])
def test_bad_lengths_raise(native, lengths):
    """Lengths outside [1, Sk] or not one per batch raise."""
    q = _values((2, 2, 40, 32), 0).to("tt")
    with pytest.raises(ValueError):
        native.sdpa(q, q, q, key_lengths=lengths)
