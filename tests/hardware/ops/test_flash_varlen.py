"""Packed variable-length attention (kohakutt.sdpa_varlen and torch's varlen_attn on
tt): each sequence against CPU float64 attention of that sequence alone, over lengths 1
to 1500, boundaries off the 32-token tiles, MHA / GQA, dense and causal.

Bound: rel RMS <= 8e-3 per call (the dense kernel's bound, test_flash.py)."""

import pytest
import torch
import torch.nn.functional as F

BOUND = 8e-3
# A sequence of one or two tokens has few terms to average the bf16 rounding over.
SEQUENCE_BOUND = 3e-2
# dq, dk, dv against float64 autograd, rel RMS per call.
GRAD_BOUND = 2.5e-2

# Sequence lengths per case.
LENGTHS = {
    "one tile": [32],
    "ragged": [40],
    "mixed": [1, 31, 33, 5, 64, 100],
    "tiny x 50": [3] * 50,
    "aligned": [32, 64, 96],
    "equal": [201] * 8,
    "long": [700, 13, 1500],
}

# (lengths, h, hk, d, causal)
CASES = [
    ("one tile", 2, 2, 32, False),
    ("ragged", 1, 1, 32, False),
    ("ragged", 2, 2, 64, True),
    ("mixed", 2, 2, 64, False),
    ("mixed", 4, 2, 64, True),
    ("tiny x 50", 4, 4, 32, False),
    ("tiny x 50", 2, 1, 64, True),
    ("aligned", 2, 2, 64, False),
    ("aligned", 2, 2, 64, True),
    ("equal", 12, 12, 64, False),
    ("equal", 4, 4, 128, True),
    ("long", 4, 2, 64, False),
    ("long", 2, 2, 64, True),
]


def _offsets(lengths):
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    return cu


def _packed(lengths, h, hk, d, seed):
    g = torch.Generator().manual_seed(seed)
    total = sum(lengths)
    q = torch.randn(total, h, d, generator=g).to(torch.bfloat16)
    k = torch.randn(total, hk, d, generator=g).to(torch.bfloat16)
    v = torch.randn(total, hk, d, generator=g).to(torch.bfloat16)
    return q, k, v


def _reference(q, k, v, cu, causal):
    """Float64 attention of each sequence alone; q [T, H, D], k / v [T, Hk, D]."""
    rep = q.shape[1] // k.shape[1]
    out = []
    for a, b in zip(cu, cu[1:]):
        qs = q[a:b].double().transpose(0, 1)
        ks = k[a:b].double().repeat_interleave(rep, 1).transpose(0, 1)
        vs = v[a:b].double().repeat_interleave(rep, 1).transpose(0, 1)
        out.append(
            F.scaled_dot_product_attention(qs, ks, vs, is_causal=causal).transpose(0, 1)
        )
    return torch.cat(out)


def _rel(y, ref):
    return ((y.double() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("name,h,hk,d,causal", CASES)
def test_matches_each_sequence(native, name, h, hk, d, causal):
    lengths = LENGTHS[name]
    cu = _offsets(lengths)
    q, k, v = _packed(lengths, h, hk, d, len(lengths) + h + d)
    ref = _reference(q, k, v, cu, causal)
    y = native.sdpa_varlen(
        q.to("tt"), k.to("tt"), v.to("tt"), cu, is_causal=causal
    ).cpu()
    assert y.shape == (cu[-1], h, d)
    assert torch.isfinite(y).all()
    assert _rel(y, ref) <= BOUND
    for a, b in zip(cu, cu[1:]):
        assert _rel(y[a:b], ref[a:b]) <= SEQUENCE_BOUND, (a, b)


@pytest.mark.parametrize("causal", [False, True])
def test_torch_varlen_attn_runs_the_same_kernel(native, causal):
    from torch.nn.attention.varlen import varlen_attn

    lengths = LENGTHS["mixed"]
    cu = _offsets(lengths)
    q, k, v = (t.to("tt") for t in _packed(lengths, 2, 2, 64, 7))
    want = native.sdpa_varlen(q, k, v, cu, is_causal=causal).cpu()
    cu_t = torch.tensor(cu, dtype=torch.int32)
    got = varlen_attn(
        q,
        k,
        v,
        cu_t,
        cu_t,
        max(lengths),
        max(lengths),
        window_size=(-1, 0) if causal else (-1, -1),
    ).cpu()
    assert torch.equal(got, want)


@pytest.mark.parametrize("causal", [False, True])
def test_sequence_does_not_depend_on_its_neighbours(native, causal):
    """A sequence's output is bit-identical when every other sequence's q, k and v
    change (same lengths), for sequences sharing tiles with their neighbours."""
    lengths = [5, 40, 3, 33, 70, 1, 31]
    cu = _offsets(lengths)
    q, k, v = _packed(lengths, 2, 2, 64, 11)
    q2, k2, v2 = _packed(lengths, 2, 2, 64, 12)
    first = native.sdpa_varlen(
        q.to("tt"), k.to("tt"), v.to("tt"), cu, is_causal=causal
    ).cpu()
    for a, b in zip(cu, cu[1:]):
        qm, km, vm = q2.clone(), k2.clone(), v2.clone()
        qm[a:b], km[a:b], vm[a:b] = q[a:b], k[a:b], v[a:b]
        y = native.sdpa_varlen(
            qm.to("tt"), km.to("tt"), vm.to("tt"), cu, is_causal=causal
        ).cpu()
        assert torch.equal(y[a:b], first[a:b]), (a, b)


def test_head_views_run_one_program(native):
    """q, k, v as [T, H, D] views of [T, H D] tensors (a projection's output): the call
    is one program, with no head split or merge around it, for a first and for a new set
    of lengths."""
    h, d = 4, 64
    for lengths in ([40, 7, 90], [41, 9, 87, 30]):
        cu = _offsets(lengths)
        q, k, v = _packed(lengths, h, h, d, 9)
        want = native.sdpa_varlen(q.to("tt"), k.to("tt"), v.to("tt"), cu).cpu()
        views = [
            t.reshape(cu[-1], h * d).to("tt").view(cu[-1], h, d) for t in (q, k, v)
        ]
        native.sdpa_varlen(*views, cu)
        native.synchronize()
        before = native.counters()
        y = native.sdpa_varlen(*views, cu)
        native.synchronize()
        after = native.counters()
        assert after["programs"] - before["programs"] == 1
        assert after["ops"] - before["ops"] == 1
        assert torch.equal(y.cpu(), want)


def test_offsets_as_tensors_and_lists(native):
    lengths = LENGTHS["mixed"]
    cu = _offsets(lengths)
    q, k, v = (t.to("tt") for t in _packed(lengths, 2, 2, 64, 3))
    want = native.sdpa_varlen(q, k, v, cu).cpu()
    as_tensor = torch.tensor(cu, dtype=torch.int32)
    assert torch.equal(native.sdpa_varlen(q, k, v, as_tensor, as_tensor).cpu(), want)
    assert torch.equal(native.sdpa_varlen(q, k, v, as_tensor.to("tt")).cpu(), want)


def test_keys_of_other_lengths(native):
    """cu_seq_k differing from cu_seq_q (cross attention per sequence)."""
    lengths_q, lengths_k = [40, 7, 90], [25, 70, 33]
    cu_q, cu_k = _offsets(lengths_q), _offsets(lengths_k)
    g = torch.Generator().manual_seed(5)
    q = torch.randn(cu_q[-1], 2, 64, generator=g).to(torch.bfloat16)
    k = torch.randn(cu_k[-1], 2, 64, generator=g).to(torch.bfloat16)
    v = torch.randn(cu_k[-1], 2, 64, generator=g).to(torch.bfloat16)
    ref = []
    for (a, b), (c, e) in zip(zip(cu_q, cu_q[1:]), zip(cu_k, cu_k[1:])):
        qs, ks, vs = (t.double().transpose(0, 1) for t in (q[a:b], k[c:e], v[c:e]))
        ref.append(F.scaled_dot_product_attention(qs, ks, vs).transpose(0, 1))
    y = native.sdpa_varlen(q.to("tt"), k.to("tt"), v.to("tt"), cu_q, cu_k).cpu()
    assert _rel(y, torch.cat(ref)) <= BOUND


# (query lengths, key lengths or None, h, hk, d, causal)
BACKWARD_CASES = [
    ([32], None, 1, 1, 32, False),
    ([40], None, 1, 1, 32, False),
    ([1, 31, 33, 5, 64, 100], None, 2, 2, 64, False),
    ([1, 31, 33, 5, 64, 100], None, 4, 2, 64, True),
    ([3] * 50, None, 2, 1, 64, True),
    ([40, 7, 90], [25, 70, 33], 2, 2, 64, False),
    ([40, 0, 90], [25, 70, 33], 2, 1, 64, False),
    ([201] * 8, None, 12, 12, 64, False),
    ([700, 13, 1500], None, 2, 2, 64, True),
]


def _reference_grads(q, k, v, grad, cu_q, cu_k, causal):
    """Float64 gradients of sum(out * grad) for each sequence attending alone."""
    qd, kd, vd = (t.double().requires_grad_() for t in (q, k, v))
    rep = q.shape[1] // k.shape[1]
    scale = q.shape[2] ** -0.5
    outs = []
    for (a, b), (c, e) in zip(zip(cu_q, cu_q[1:]), zip(cu_k, cu_k[1:])):
        if a == b:
            continue
        qs = qd[a:b].transpose(0, 1)
        ks = kd[c:e].repeat_interleave(rep, 1).transpose(0, 1)
        vs = vd[c:e].repeat_interleave(rep, 1).transpose(0, 1)
        s = qs @ ks.transpose(1, 2) * scale
        if causal:
            n, m = b - a, e - c
            seen = torch.arange(m)[None, :] < (torch.arange(n)[:, None] + 1 + m - n)
            s = s.masked_fill(~seen, float("-inf"))
        outs.append((torch.softmax(s, -1) @ vs).transpose(0, 1))
    (torch.cat(outs) * grad.double()).sum().backward()
    return qd.grad, kd.grad, vd.grad


def _device_grads(native, q, k, v, grad, cu_q, cu_k, causal):
    qt, kt, vt = (t.to("tt").requires_grad_() for t in (q, k, v))
    native.sdpa_varlen(qt, kt, vt, cu_q, cu_k, is_causal=causal).backward(grad.to("tt"))
    return qt.grad.cpu(), kt.grad.cpu(), vt.grad.cpu()


@pytest.mark.parametrize("lq,lk,h,hk,d,causal", BACKWARD_CASES)
def test_backward_matches_each_sequence(native, lq, lk, h, hk, d, causal):
    """dq, dk, dv (packed) against float64 autograd; the keys of a sequence without
    queries get exact zeros."""
    cu_q = _offsets(lq)
    cu_k = cu_q if lk is None else _offsets(lk)
    g = torch.Generator().manual_seed(len(lq) + h + d)
    q = torch.randn(cu_q[-1], h, d, generator=g).to(torch.bfloat16)
    k = torch.randn(cu_k[-1], hk, d, generator=g).to(torch.bfloat16)
    v = torch.randn(cu_k[-1], hk, d, generator=g).to(torch.bfloat16)
    grad = torch.randn(cu_q[-1], h, d, generator=g).to(torch.bfloat16)
    want = _reference_grads(q, k, v, grad, cu_q, cu_k, causal)
    got = _device_grads(native, q, k, v, grad, cu_q, cu_k, causal)
    for y, ref, like in zip(got, want, (q, k, v)):
        assert y.shape == like.shape
        assert torch.isfinite(y.float()).all()
        assert _rel(y, ref) <= GRAD_BOUND
    for (a, b), (c, e) in zip(zip(cu_q, cu_q[1:]), zip(cu_k, cu_k[1:])):
        if a == b:
            assert not got[1][c:e].float().any() and not got[2][c:e].float().any()


def test_backward_through_torch_varlen_attn(native):
    from torch.nn.attention.varlen import varlen_attn

    lengths = LENGTHS["mixed"]
    cu = _offsets(lengths)
    q, k, v = _packed(lengths, 2, 2, 64, 7)
    grad = _packed(lengths, 2, 2, 64, 8)[0]
    want = _device_grads(native, q, k, v, grad, cu, cu, True)
    qt, kt, vt = (t.to("tt").requires_grad_() for t in (q, k, v))
    cu_t = torch.tensor(cu, dtype=torch.int32)
    out = varlen_attn(
        qt, kt, vt, cu_t, cu_t, max(lengths), max(lengths), window_size=(-1, 0)
    )
    out.backward(grad.to("tt"))
    for y, ref in zip((qt.grad, kt.grad, vt.grad), want):
        assert torch.equal(y.cpu(), ref)


def test_gradients_do_not_depend_on_neighbours(native):
    """A sequence's dq, dk and dv are bit-identical when every other sequence's q, k, v
    and output gradient change (same lengths)."""
    lengths = [5, 40, 3, 33, 70, 1, 31]
    cu = _offsets(lengths)
    q, k, v = _packed(lengths, 2, 2, 64, 11)
    grad = _packed(lengths, 2, 2, 64, 13)[0]
    q2, k2, v2 = _packed(lengths, 2, 2, 64, 12)
    grad2 = _packed(lengths, 2, 2, 64, 14)[0]
    first = _device_grads(native, q, k, v, grad, cu, cu, False)
    for a, b in zip(cu, cu[1:]):
        qm, km, vm, gm = q2.clone(), k2.clone(), v2.clone(), grad2.clone()
        qm[a:b], km[a:b], vm[a:b], gm[a:b] = q[a:b], k[a:b], v[a:b], grad[a:b]
        got = _device_grads(native, qm, km, vm, gm, cu, cu, False)
        for y, ref in zip(got, first):
            assert torch.equal(y[a:b], ref[a:b]), (a, b)


def test_backward_of_head_views_runs_two_programs(native):
    """With q, k, v and the output gradient as [T, H, D] views of [T, H D] tensors the
    backward is its two programs (dQ, then dK / dV), also for a new set of lengths."""
    h, d = 4, 64
    for lengths in ([40, 7, 90], [41, 9, 87, 30]):
        cu = _offsets(lengths)
        q, k, v = _packed(lengths, h, h, d, 9)
        grad = (
            _packed(lengths, h, h, d, 10)[0]
            .reshape(cu[-1], h * d)
            .to("tt")
            .view(cu[-1], h, d)
        )
        views = [
            t.reshape(cu[-1], h * d).to("tt").requires_grad_().view(cu[-1], h, d)
            for t in (q, k, v)
        ]
        for _ in range(2):
            out = native.sdpa_varlen(*views, cu)
            native.synchronize()
            before = native.counters()
            torch.autograd.grad(out, views, grad)
            native.synchronize()
            after = native.counters()
        assert after["programs"] - before["programs"] == 2


def test_log_sum_exp(native):
    """torch's varlen_attn returns each query's log-sum-exp [H, Tq]."""
    from torch.nn.attention.varlen import AuxRequest, varlen_attn

    lengths = LENGTHS["mixed"]
    cu = _offsets(lengths)
    q, k, v = _packed(lengths, 2, 2, 64, 7)
    cu_t = torch.tensor(cu, dtype=torch.int32)
    _, lse = varlen_attn(
        q.to("tt"),
        k.to("tt"),
        v.to("tt"),
        cu_t,
        cu_t,
        max(lengths),
        max(lengths),
        return_aux=AuxRequest(lse=True),
    )
    want = []
    for a, b in zip(cu, cu[1:]):
        s = (
            q[a:b].double().transpose(0, 1)
            @ k[a:b].double().transpose(0, 1).transpose(1, 2)
            / 8.0
        )
        want.append(torch.logsumexp(s, -1))
    assert lse.shape == (2, cu[-1])
    assert (lse.cpu().double() - torch.cat(want, 1)).abs().max() <= 3e-2


def test_log_sum_exp_of_a_centred_training_call(native, monkeypatch):
    """With KTT_ATTENTION_CENTER=1, tensors that require grad (the keys are centred) and
    q, k offset by +2, the returned log-sum-exp is that of the given keys."""
    from torch.nn.attention.varlen import AuxRequest, varlen_attn

    monkeypatch.setenv("KTT_ATTENTION_CENTER", "1")

    lengths = LENGTHS["mixed"]
    cu = _offsets(lengths)
    q, k, v = _packed(lengths, 2, 2, 64, 7)
    q, k = ((t.float() + 2.0).to(torch.bfloat16) for t in (q, k))
    cu_t = torch.tensor(cu, dtype=torch.int32)
    out, lse = varlen_attn(
        *(t.to("tt").requires_grad_() for t in (q, k, v)),
        cu_t,
        cu_t,
        max(lengths),
        max(lengths),
        return_aux=AuxRequest(lse=True),
    )
    want = []
    for a, b in zip(cu, cu[1:]):
        s = (
            q[a:b].double().transpose(0, 1)
            @ k[a:b].double().transpose(0, 1).transpose(1, 2)
            / 8.0
        )
        want.append(torch.logsumexp(s, -1))
    assert out.requires_grad
    assert lse.shape == (2, cu[-1])
    worst = (lse.detach().cpu().double() - torch.cat(want, 1)).abs().max().item()
    print(f"training log-sum-exp: max abs error {worst:.3e}")
    assert worst <= 5e-2


def _key_mask(kind, cu_k, seed):
    """A key mask [Tk] of `kind` ("bool" or "float") that keeps the first key of every
    sequence, and its float64 additive form (-inf for a key left out)."""
    g = torch.Generator().manual_seed(seed)
    tk = cu_k[-1]
    if kind == "bool":
        mask = torch.rand(tk, generator=g) < 0.7
        mask[cu_k[:-1]] = True
        return mask, torch.zeros(tk, dtype=torch.float64).masked_fill(
            ~mask, float("-inf")
        )
    mask = (torch.rand(tk, generator=g) * 4 - 2).to(torch.bfloat16)
    drop = torch.rand(tk, generator=g) < 0.2
    drop[cu_k[:-1]] = False
    mask = mask.masked_fill(drop, float("-inf"))
    return mask, mask.double()


# (query lengths, key lengths or None, h, hk, d, causal, mask kind, mask on tt)
MASK_CASES = [
    ([32], None, 1, 1, 32, False, "bool", False),
    ([40], None, 1, 1, 32, False, "float", False),
    ([1, 31, 33, 5, 64, 100], None, 2, 2, 64, False, "bool", True),
    ([1, 31, 33, 5, 64, 100], None, 4, 2, 64, True, "float", True),
    ([3] * 50, None, 2, 1, 64, True, "bool", False),
    ([40, 0, 90], [25, 70, 33], 2, 1, 64, False, "bool", False),
    ([201] * 8, None, 12, 12, 64, False, "bool", False),
    ([700, 13, 1500], None, 2, 2, 64, True, "float", False),
]


@pytest.mark.parametrize("lq,lk,h,hk,d,causal,kind,on_tt", MASK_CASES)
def test_key_mask_forward_and_backward(native, lq, lk, h, hk, d, causal, kind, on_tt):
    """attn_mask [Tk] (bool keep mask or additive) against float64 autograd with the
    mask added to each key's scores; a key a bool mask leaves out gets exact zero dv."""
    cu_q = _offsets(lq)
    cu_k = cu_q if lk is None else _offsets(lk)
    g = torch.Generator().manual_seed(len(lq) + h + d)
    q = torch.randn(cu_q[-1], h, d, generator=g).to(torch.bfloat16)
    k = torch.randn(cu_k[-1], hk, d, generator=g).to(torch.bfloat16)
    v = torch.randn(cu_k[-1], hk, d, generator=g).to(torch.bfloat16)
    grad = torch.randn(cu_q[-1], h, d, generator=g).to(torch.bfloat16)
    mask, additive = _key_mask(kind, cu_k, 21)

    qd, kd, vd = (t.double().requires_grad_() for t in (q, k, v))
    rep = h // hk
    outs = []
    for (a, b), (c, e) in zip(zip(cu_q, cu_q[1:]), zip(cu_k, cu_k[1:])):
        if a == b:
            continue
        qs = qd[a:b].transpose(0, 1)
        ks = kd[c:e].repeat_interleave(rep, 1).transpose(0, 1)
        vs = vd[c:e].repeat_interleave(rep, 1).transpose(0, 1)
        s = qs @ ks.transpose(1, 2) * d**-0.5 + additive[c:e]
        if causal:
            n, m = b - a, e - c
            seen = torch.arange(m)[None, :] < (torch.arange(n)[:, None] + 1 + m - n)
            s = s.masked_fill(~seen, float("-inf"))
        outs.append((torch.softmax(s, -1) @ vs).transpose(0, 1))
    ref = torch.cat(outs)
    (ref * grad.double()).sum().backward()

    qt, kt, vt = (t.to("tt").requires_grad_() for t in (q, k, v))
    out = native.sdpa_varlen(
        qt,
        kt,
        vt,
        cu_q,
        cu_k,
        is_causal=causal,
        attn_mask=mask.to("tt") if on_tt else mask,
    )
    out.backward(grad.to("tt"))
    assert _rel(out.detach().cpu(), ref.detach()) <= BOUND
    for y, want in zip((qt.grad, kt.grad, vt.grad), (qd.grad, kd.grad, vd.grad)):
        assert torch.isfinite(y.cpu().float()).all()
        assert _rel(y.cpu(), want) <= GRAD_BOUND
    if kind == "bool":
        assert not vt.grad.cpu()[~mask].float().any()
        assert not kt.grad.cpu()[~mask].float().any()


@pytest.mark.parametrize("masked_logit", [1e3, 1e4, 3e4, 1e5])
@pytest.mark.parametrize("kind", ["bool", "-inf"])
def test_key_mask_removes_large_logits(native, masked_logit, kind):
    """Masked keys whose logits (about masked_logit) far exceed the kept keys' (about 1)
    get no weight: each sequence is its kept keys' attention within BOUND."""
    lengths, h, d = [40, 90, 33], 2, 64
    cu = _offsets(lengths)
    g = torch.Generator().manual_seed(int(masked_logit))
    u = F.normalize(torch.randn(d, generator=g), dim=0)
    q = (torch.randn(cu[-1], h, d, generator=g) + 4 * u).to(torch.bfloat16)
    k = torch.randn(cu[-1], h, d, generator=g)
    keep = torch.rand(cu[-1], generator=g) >= 0.4
    keep[cu[:-1]] = True
    k = torch.where(~keep[:, None, None], u * masked_logit * d**0.5 / 4, k)
    k = k.to(torch.bfloat16)
    v = torch.randn(cu[-1], h, d, generator=g).to(torch.bfloat16)
    outs = []
    for a, b in zip(cu, cu[1:]):
        qs, ks, vs = (t[a:b].double().transpose(0, 1) for t in (q, k, v))
        o = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=keep[a:b])
        outs.append(o.transpose(0, 1))
    ref = torch.cat(outs)
    mask = keep
    if kind == "-inf":
        mask = torch.zeros(cu[-1], dtype=torch.bfloat16).masked_fill(
            ~keep, float("-inf")
        )
    y = native.sdpa_varlen(q.to("tt"), k.to("tt"), v.to("tt"), cu, attn_mask=mask)
    y = y.cpu()
    assert torch.isfinite(y.float()).all()
    assert _rel(y, ref) <= BOUND


def test_key_mask_keeping_every_key_changes_nothing(native):
    lengths = LENGTHS["mixed"]
    cu = _offsets(lengths)
    q, k, v = (t.to("tt") for t in _packed(lengths, 2, 2, 64, 7))
    want = native.sdpa_varlen(q, k, v, cu).cpu()
    keep = torch.ones(cu[-1], dtype=torch.bool)
    assert torch.equal(native.sdpa_varlen(q, k, v, cu, attn_mask=keep).cpu(), want)
    zeros = torch.zeros(cu[-1], dtype=torch.bfloat16)
    assert torch.equal(native.sdpa_varlen(q, k, v, cu, attn_mask=zeros).cpu(), want)


def test_key_mask_on_the_device_is_not_downloaded(native):
    """A tt mask is converted by device ops (no CPU fallback) and the attention itself
    stays one program."""
    lengths = [40, 7, 90]
    cu = _offsets(lengths)
    h, d = 4, 64
    q, k, v = (
        t.reshape(cu[-1], h * d).to("tt").view(cu[-1], h, d)
        for t in _packed(lengths, h, h, d, 9)
    )
    mask, _ = _key_mask("bool", cu, 5)
    on_cpu = native.sdpa_varlen(q, k, v, cu, attn_mask=mask).cpu()
    native.synchronize()
    torch.ops.ktt.reset_fallback_counts()
    y = native.sdpa_varlen(q, k, v, cu, attn_mask=mask.to("tt"))
    native.synchronize()
    assert torch.ops.ktt.fallback_counts() == []
    assert torch.equal(y.cpu(), on_cpu)


def test_rejects_what_it_does_not_take(native):
    q, k, v = (t.to("tt") for t in _packed([40], 2, 2, 64, 1))
    with pytest.raises(RuntimeError):
        native.sdpa_varlen(q, k, v, [0, 39])
    with pytest.raises(ValueError):
        native.sdpa_varlen(
            q, k, v, [0, 40], attn_mask=torch.ones(40, 40, dtype=torch.bool)
        )
    with pytest.raises(ValueError):
        native.sdpa_varlen(q, k, v, [0, 40], attn_mask=torch.ones(39, dtype=torch.bool))
