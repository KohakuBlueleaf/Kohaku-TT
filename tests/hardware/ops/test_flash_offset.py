"""Training attention under KTT_ATTENTION_CENTER=1 with q, k, v and the output gradient
offset from zero mean: output and gradients against float64 autograd, dense and packed.

Bounds, rel RMS: out 8e-3, dk and dv 2.5e-2, dq 2.5e-2 (5e-2 at offset 5)."""

import pytest
import torch
import torch.nn.functional as F

BOUND = 8e-3
GRAD_BOUND = 2.5e-2
DQ_BOUND = {0.0: 2.5e-2, 2.0: 2.5e-2, 5.0: 5e-2}


@pytest.fixture(autouse=True)
def _centring(monkeypatch):
    monkeypatch.setenv("KTT_ATTENTION_CENTER", "1")


# (b, h, hk, s, d, causal)
CASES = [
    (1, 1, 1, 64, 32, False),
    (2, 4, 4, 201, 64, False),
    (1, 12, 12, 512, 64, False),
    (2, 4, 4, 201, 64, True),
    (2, 4, 2, 128, 64, False),
]


def _rel(y, ref):
    return ((y.double() - ref).norm() / ref.norm()).item()


def _inputs(b, h, hk, s, d, offset, seed):
    """bf16 q [b, h, s, d], k / v [b, hk, s, d] and an output gradient [b, h, s, d],
    unit-variance noise plus `offset`."""
    g = torch.Generator().manual_seed(seed)
    shapes = ((b, h, s, d), (b, hk, s, d), (b, hk, s, d), (b, h, s, d))
    return [(torch.randn(n, generator=g) + offset).bfloat16() for n in shapes]


def _reference(q, k, v, dout, causal):
    """Float64 output and the gradients of sum(out * dout) for q, k and v."""
    rep = q.shape[1] // k.shape[1]
    qd, kd, vd = (t.double().requires_grad_() for t in (q, k, v))
    out = F.scaled_dot_product_attention(
        qd,
        kd.repeat_interleave(rep, 1),
        vd.repeat_interleave(rep, 1),
        is_causal=causal,
    )
    (out * dout.double()).sum().backward()
    return out.detach(), qd.grad, kd.grad, vd.grad


def _device(q, k, v, dout, causal):
    """The tt output and gradients (dq, dk, dv), on the CPU."""
    qt, kt, vt = (t.to("tt").requires_grad_() for t in (q, k, v))
    out = F.scaled_dot_product_attention(
        qt, kt, vt, is_causal=causal, enable_gqa=q.shape[1] != k.shape[1]
    )
    out.backward(dout.to("tt"))
    return [t.detach().cpu() for t in (out, qt.grad, kt.grad, vt.grad)]


def _check(b, h, hk, s, d, causal, offset):
    q, k, v, dout = _inputs(b, h, hk, s, d, offset, b + h + s + d)
    ref = _reference(q, k, v, dout, causal)
    got = _device(q, k, v, dout, causal)
    bounds = (BOUND, DQ_BOUND[offset], GRAD_BOUND, GRAD_BOUND)
    rels = []
    for name, x, r in zip(("out", "dq", "dk", "dv"), got, ref):
        assert x.shape == r.shape, name
        assert torch.isfinite(x.float()).all(), name
        rels.append(_rel(x, r))
    label = f"[{b}, {h} / {hk}, {s}, {d}] causal {causal} offset {offset}"
    print(
        f"{label}: out {rels[0]:.3e} dq {rels[1]:.3e} dk {rels[2]:.3e} "
        f"dv {rels[3]:.3e}"
    )
    for name, rel, bound in zip(("out", "dq", "dk", "dv"), rels, bounds):
        assert rel <= bound, f"{label}: {name} rel {rel:.3e} > {bound}"


@pytest.mark.parametrize("offset", [2.0, 5.0])
@pytest.mark.parametrize("b,h,hk,s,d,causal", CASES)
def test_offset_inputs_match_float64(native, b, h, hk, s, d, causal, offset):
    _check(b, h, hk, s, d, causal, offset)


def test_zero_mean_inputs_match_float64(native):
    _check(2, 4, 4, 201, 64, False, 0.0)


def _forward_ops(native, grads):
    """Device ops of one attention forward at (2, 4, 201, 64)."""
    q, k, v, _ = _inputs(2, 4, 4, 201, 64, 0.0, 3)
    qt, kt, vt = (t.to("tt").requires_grad_(grads) for t in (q, k, v))
    F.scaled_dot_product_attention(qt, kt, vt)
    native.synchronize()
    before = native.counters()["ops"]
    F.scaled_dot_product_attention(qt, kt, vt)
    native.synchronize()
    return native.counters()["ops"] - before


def test_inference_is_one_device_op(native):
    """A call without gradients runs no centring op."""
    assert _forward_ops(native, False) == 1


def test_training_without_the_switch_is_one_device_op(native, monkeypatch):
    monkeypatch.delenv("KTT_ATTENTION_CENTER")
    assert _forward_ops(native, True) == 1


@pytest.mark.parametrize("causal", [False, True])
def test_packed_offset_inputs_match_float64(native, causal):
    """Packed sequences of 40, 7 and 90 tokens, q / k / v / dO offset by 5, -3, 2, 4."""
    cu = [0, 40, 47, 137]
    g = torch.Generator().manual_seed(38)
    q, k, v, dout = (
        (torch.randn(137, 2, 64, generator=g) + c).bfloat16()
        for c in (5.0, -3.0, 2.0, 4.0)
    )
    qd, kd, vd = (t.double().requires_grad_() for t in (q, k, v))
    outs = [
        F.scaled_dot_product_attention(
            *(t[a:b].transpose(0, 1) for t in (qd, kd, vd)), is_causal=causal
        ).transpose(0, 1)
        for a, b in zip(cu, cu[1:])
    ]
    ref = torch.cat(outs)
    (ref * dout.double()).sum().backward()
    qt, kt, vt = (t.to("tt").requires_grad_() for t in (q, k, v))
    out = native.sdpa_varlen(qt, kt, vt, cu, is_causal=causal)
    out.backward(dout.to("tt"))
    assert _rel(out.detach().cpu(), ref.detach()) <= BOUND
    rels = [_rel(t.grad.cpu(), r.grad) for t, r in zip((qt, kt, vt), (qd, kd, vd))]
    print(f"packed causal {causal}: dq {rels[0]:.3e} dk {rels[1]:.3e} dv {rels[2]:.3e}")
    assert rels[0] <= DQ_BOUND[5.0] and rels[1] <= GRAD_BOUND and rels[2] <= GRAD_BOUND
