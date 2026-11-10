"""Packed variable-length attention against float64 attention and autograd of each
sequence alone: key tile rows no query sees, long plans, causal with other key lengths,
Dv != D, D 256, an explicit scale, inputs computed on the device, no_grad, and more
length sets than the plan cache keeps."""

import pytest
import torch

# test_flash_varlen.py's bounds (rel RMS per call; per sequence; gradients per call).
BOUND = 8e-3
SEQUENCE_BOUND = 3e-2
GRAD_BOUND = 2.5e-2

MIXED = [1, 31, 33, 5, 64, 100]


def _offsets(lengths):
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    return cu


def _inputs(lq, lk, h, hk, d, dv, seed):
    """bf16 q [Tq, h, d], k [Tk, hk, d], v [Tk, hk, dv] and an output gradient
    [Tq, h, dv]."""
    g = torch.Generator().manual_seed(seed)
    tq, tk = sum(lq), sum(lk)
    q = torch.randn(tq, h, d, generator=g).to(torch.bfloat16)
    k = torch.randn(tk, hk, d, generator=g).to(torch.bfloat16)
    v = torch.randn(tk, hk, dv, generator=g).to(torch.bfloat16)
    grad = torch.randn(tq, h, dv, generator=g).to(torch.bfloat16)
    return q, k, v, grad


def _reference(q, k, v, cu_q, cu_k, causal, scale=None, grad=None):
    """Float64 attention of each sequence alone, packed [Tq, H, Dv]; with `grad` also
    the gradients of sum(out * grad) for q, k and v."""
    qd, kd, vd = (t.double().requires_grad_(grad is not None) for t in (q, k, v))
    rep = q.shape[1] // k.shape[1]
    if scale is None:
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
    out = torch.cat(outs)
    if grad is None:
        return out
    (out * grad.double()).sum().backward()
    return out.detach(), (qd.grad, kd.grad, vd.grad)


def _rel(y, ref):
    return ((y.double() - ref).norm() / ref.norm()).item()


def _check_forward(y, ref, cu_q, shape, label):
    assert y.shape == shape
    assert torch.isfinite(y.float()).all()
    rel = _rel(y, ref)
    print(f"{label}: forward rel {rel:.3e}")
    assert rel <= BOUND, f"{label}: forward rel {rel:.3e} > {BOUND}"
    for a, b in zip(cu_q, cu_q[1:]):
        if a == b:
            continue
        rel = _rel(y[a:b], ref[a:b])
        assert rel <= SEQUENCE_BOUND, f"{label}: sequence [{a}, {b}) rel {rel:.3e}"


def _check_grads(got, want, like, label):
    failures = []
    for name, y, ref, t in zip(("dq", "dk", "dv"), got, want, like):
        assert y.shape == t.shape, name
        assert torch.isfinite(y.float()).all(), f"{label}: {name} not finite"
        rel = _rel(y, ref)
        print(f"{label}: {name} rel {rel:.3e}")
        if rel > GRAD_BOUND:
            failures.append(f"{name} rel {rel:.3e}")
    assert not failures, f"{label}: {', '.join(failures)} > {GRAD_BOUND}"


def _run(native, q, k, v, grad, cu_q, cu_k, causal, scale=None, backward=True):
    """The device output, and with `backward` (dq, dk, dv), all on the CPU."""
    qt, kt, vt = (t.to("tt").requires_grad_(backward) for t in (q, k, v))
    if scale is None:
        out = native.sdpa_varlen(qt, kt, vt, cu_q, cu_k, is_causal=causal)
    else:
        out = native.sdpa_varlen(qt, kt, vt, cu_q, cu_k, is_causal=causal, scale=scale)
    if not backward:
        return out.cpu(), None
    out.backward(grad.to("tt"))
    return out.detach().cpu(), (qt.grad.cpu(), kt.grad.cpu(), vt.grad.cpu())


def _forward_and_backward(
    native, lq, lk, h, hk, d, dv, causal, seed, scale=None, label=""
):
    cu_q, cu_k = _offsets(lq), _offsets(lk)
    q, k, v, grad = _inputs(lq, lk, h, hk, d, dv, seed)
    ref, want = _reference(q, k, v, cu_q, cu_k, causal, scale, grad)
    y, got = _run(native, q, k, v, grad, cu_q, cu_k, causal, scale)
    _check_forward(y, ref, cu_q, (cu_q[-1], h, dv), label)
    _check_grads(got, want, (q, k, v), label)
    return cu_q, cu_k, got


# 1. Key tile rows no query sees.


@pytest.mark.parametrize("lq,lk", [([32, 0, 32], [32, 64, 32]), ([32, 0], [32, 64])])
def test_key_tile_rows_no_query_sees(native, lq, lk):
    """A sequence with keys in whole tiles and no query: no query tile row sees its key
    tile rows. Its dk and dv are exact zeros."""
    cu_q, cu_k, got = _forward_and_backward(
        native, lq, lk, 2, 2, 64, 64, False, 31, label=f"lq {lq} lk {lk}"
    )
    for (a, b), (c, e) in zip(zip(cu_q, cu_q[1:]), zip(cu_k, cu_k[1:])):
        if a == b:
            for name, y in (("dk", got[1]), ("dv", got[2])):
                rows = y[c:e].float().flatten(1).abs().amax(1).nonzero().flatten() + c
                assert rows.numel() == 0, (
                    f"{name} of the query-less keys [{c}, {e}) is not zero in "
                    f"{rows.numel()} rows: {rows.tolist()[:16]}, "
                    f"max |value| {y[c:e].float().abs().max().item():.3e}"
                )


# 2. Second plan page.


def test_second_plan_page_forward_and_backward(native):
    """One sequence of 4200 tokens (132 query tile rows)."""
    _forward_and_backward(native, [4200], [4200], 1, 1, 32, 32, False, 32, label="4200")


def test_second_plan_page_causal_forward(native):
    lengths = [4100, 100]
    cu = _offsets(lengths)
    q, k, v, _ = _inputs(lengths, lengths, 1, 1, 32, 32, 33)
    ref = _reference(q, k, v, cu, cu, True)
    y, _ = _run(native, q, k, v, None, cu, cu, True, backward=False)
    _check_forward(y, ref, cu, (cu[-1], 1, 32), "4100 + 100 causal")


# 3. Causal with Lk != Lq.


def test_causal_with_more_keys_than_queries(native):
    """Query r sees its first r + 1 + Lk - Lq keys."""
    _forward_and_backward(
        native,
        [40, 7, 90],
        [50, 20, 100],
        2,
        2,
        64,
        64,
        True,
        34,
        label="causal Lk > Lq",
    )


# 4. Dv != D.


@pytest.mark.parametrize("d,dv", [(64, 32), (32, 128)])
def test_value_width_differs(native, d, dv):
    _forward_and_backward(
        native, MIXED, MIXED, 4, 2, d, dv, False, 35 + d, label=f"d {d} dv {dv}"
    )


# 5. D = 256.


def test_width_256(native):
    lengths = [40, 7, 90]
    _forward_and_backward(
        native, lengths, lengths, 2, 2, 256, 256, False, 36, label="d 256"
    )


# 6. Explicit scale.


@pytest.mark.parametrize("causal", [False, True])
def test_explicit_scale(native, causal):
    _forward_and_backward(
        native,
        MIXED,
        MIXED,
        2,
        2,
        64,
        64,
        causal,
        37,
        scale=0.05,
        label=f"scale 0.05 causal {causal}",
    )


# 7. Tile padding that is not fresh.

# The constants the device adds to q, k, v and the output gradient.
SHIFTS = (5.0, -3.0, 2.0, 4.0)


def _device_results(tensors, zero_mean):
    """tt tensors that are results of device ops ((x * 1.0) + c), so the rows past the
    last token of their last tile are what those ops left there, not an upload's. With
    `zero_mean` the host values are shifted by -c first, so the results have zero mean
    (the backward's documented limit is for inputs far from zero mean)."""
    out = []
    for t, c in zip(tensors, SHIFTS):
        host = (t.float() - c).to(torch.bfloat16) if zero_mean else t
        out.append((host.to("tt") * 1.0) + c)
    return out


@pytest.mark.parametrize("zero_mean", [True, False], ids=["zero mean", "offset"])
@pytest.mark.parametrize("causal", [False, True])
def test_inputs_from_device_ops_forward(native, causal, zero_mean):
    """137 tokens (9 in the last tile), q / k / v computed on the device. The reference
    takes the downloaded device tensors."""
    lengths = [40, 7, 90]
    cu = _offsets(lengths)
    qt, kt, vt, _ = _device_results(
        _inputs(lengths, lengths, 2, 2, 64, 64, 38), zero_mean
    )
    ref = _reference(qt.cpu(), kt.cpu(), vt.cpu(), cu, cu, causal)
    y = native.sdpa_varlen(qt, kt, vt, cu, is_causal=causal).cpu()
    _check_forward(y, ref, cu, (cu[-1], 2, 64), f"device inputs causal {causal}")


# dq and dk of inputs far from zero mean (flash.md, Limits).
OFFSET_BACKWARD = pytest.param(
    False, id="offset", marks=pytest.mark.xfail(strict=True, reason="offset inputs")
)


@pytest.mark.parametrize(
    "zero_mean", [pytest.param(True, id="zero mean"), OFFSET_BACKWARD]
)
@pytest.mark.parametrize("causal", [False, True])
def test_inputs_from_device_ops_backward(native, causal, zero_mean):
    """As above with the backward; the output gradient is a device result too."""
    lengths = [40, 7, 90]
    cu = _offsets(lengths)
    dev = _device_results(_inputs(lengths, lengths, 2, 2, 64, 64, 38), zero_mean)
    qt, kt, vt = (t.detach().requires_grad_() for t in dev[:3])
    gt = dev[3]
    host = [t.detach().cpu() for t in (qt, kt, vt)]
    ref, want = _reference(*host, cu, cu, causal, grad=gt.cpu())
    out = native.sdpa_varlen(qt, kt, vt, cu, is_causal=causal)
    out.backward(gt)
    label = f"device inputs causal {causal}"
    _check_forward(out.detach().cpu(), ref, cu, (cu[-1], 2, 64), label)
    _check_grads((qt.grad.cpu(), kt.grad.cpu(), vt.grad.cpu()), want, host, label)


# 8. no_grad.


@pytest.mark.parametrize("causal", [False, True])
def test_no_grad_forward_is_the_same(native, causal):
    lengths = [40, 7, 90]
    cu = _offsets(lengths)
    q, k, v, _ = _inputs(lengths, lengths, 2, 2, 64, 64, 39)
    qt, kt, vt = (t.to("tt").requires_grad_() for t in (q, k, v))
    with_grad = native.sdpa_varlen(qt, kt, vt, cu, is_causal=causal)
    assert with_grad.requires_grad
    with torch.no_grad():
        without = native.sdpa_varlen(qt, kt, vt, cu, is_causal=causal)
    assert not without.requires_grad
    assert torch.equal(without.cpu(), with_grad.detach().cpu())
    plain = native.sdpa_varlen(q.to("tt"), k.to("tt"), v.to("tt"), cu, is_causal=causal)
    assert torch.equal(plain.cpu(), with_grad.detach().cpu())


# 9. More length sets than the plan cache keeps.


def test_more_length_sets_than_the_plan_cache(native):
    """40 length sets (the cache keeps 32), then the first one again."""
    worst = (0.0, None)
    for n in [*range(33, 73), 33]:
        cu = [0, n]
        q, k, v, _ = _inputs([n], [n], 1, 1, 32, 32, n)
        ref = _reference(q, k, v, cu, cu, False)
        y = native.sdpa_varlen(q.to("tt"), k.to("tt"), v.to("tt"), cu).cpu()
        assert y.shape == (n, 1, 32)
        assert torch.isfinite(y.float()).all(), n
        rel = _rel(y, ref)
        worst = max(worst, (rel, n))
        assert rel <= BOUND, f"length {n}: forward rel {rel:.3e} > {BOUND}"
    print(f"length sets: worst forward rel {worst[0]:.3e} at length {worst[1]}")
