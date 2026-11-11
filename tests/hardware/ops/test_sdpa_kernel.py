"""kohakutt.sdpa_kernel: attention runs on the kernels a scope allows (flash, own, ttnn,
math), forward and backward; a call none of them takes raises; the routing outside a
scope is unchanged (each default call equals the same call with its kernel enforced, bit
for bit). Kernels are told apart by the attention_* counters."""

import pytest
import torch
import torch.nn.functional as F

KERNELS = ("flash", "own", "ttnn", "math")
# rel RMS against float64 per kernel (TT-NN's streaming kernel keeps bf16 dest
# registers: 2.3e-2 measured); gradients against float64 autograd.
BOUND = {"flash": 8e-3, "own": 2e-2, "ttnn": 3e-2, "math": 2e-2}
GRAD_BOUND = 3e-2


def _rel(y, ref):
    return ((y.double() - ref).norm() / ref.norm()).item()


def _qkv(b, h, hk, s, d, seed, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(b, n, s, d, generator=g).to(dtype) for n in (h, hk, hk))


def _routes(native, before):
    """{kernel: calls} of the attention counters (forward, backward_*) since before."""
    after = native.counters()
    keys = [k for k in after if k.startswith("attention_")]
    return {
        k[len("attention_") :]: after[k] - before[k]
        for k in keys
        if after[k] != before[k]
    }


def _reference(q, k, v, causal=False, mask=None):
    rep = q.shape[1] // k.shape[1]
    kr, vr = (t.double().repeat_interleave(rep, 1) for t in (k, v))
    return F.scaled_dot_product_attention(
        q.double(), kr, vr, attn_mask=mask, is_causal=causal
    )


def _run(q, k, v, **kw):
    return F.scaled_dot_product_attention(
        q.to("tt"), k.to("tt"), v.to("tt"), enable_gqa=q.shape[1] != k.shape[1], **kw
    ).cpu()


@pytest.mark.parametrize("kernel", KERNELS)
def test_each_kernel_takes_what_it_supports(native, kernel):
    q, k, v = _qkv(1, 4, 4, 128, 64, 1)
    before = native.counters()
    with native.sdpa_kernel(kernel):
        y = _run(q, k, v)
    assert _routes(native, before) == {kernel: 1}
    assert _rel(y, _reference(q, k, v)) <= BOUND[kernel]


def test_scope_with_several_kernels_takes_the_first_that_fits(native):
    q, k, v = _qkv(1, 2, 2, 64, 320, 2)
    before = native.counters()
    with native.sdpa_kernel("flash", "ttnn"):
        y = _run(q, k, v)
    assert _routes(native, before) == {"ttnn": 1}
    assert _rel(y, _reference(q, k, v)) <= BOUND["ttnn"]


# (name, (b, h, hk, s, d), dtype, causal, mask kind, KTT_FLASH)
DEFAULT_CASES = [
    ("flash", (1, 4, 4, 128, 64), torch.bfloat16, False, None, None),
    ("flash-mask", (2, 4, 4, 197, 64), torch.bfloat16, False, "bool", None),
    ("own-causal", (1, 4, 2, 128, 64), torch.bfloat16, True, None, "0"),
    ("ttnn-masked", (1, 4, 4, 128, 64), torch.bfloat16, False, "float", "0"),
    ("math-head-dim", (1, 4, 4, 128, 40), torch.bfloat16, False, None, None),
    ("math-fp32", (1, 4, 4, 128, 64), torch.float32, True, None, None),
]


@pytest.mark.parametrize(
    "name,shape,dtype,causal,kind,flash",
    DEFAULT_CASES,
    ids=[c[0] for c in DEFAULT_CASES],
)
def test_default_routing_unchanged(
    native, monkeypatch, name, shape, dtype, causal, kind, flash
):
    """The default call runs one kernel; enforcing that kernel gives the same bits."""
    if flash is not None:
        monkeypatch.setenv("KTT_FLASH", flash)
    b, h, hk, s, d = shape
    q, k, v = _qkv(b, h, hk, s, d, 3, dtype)
    mask = None
    if kind == "bool":
        mask = torch.rand(b, 1, s, s, generator=torch.Generator().manual_seed(4)) > 0.2
    elif kind == "float":
        mask = torch.randn(b, 1, s, s, generator=torch.Generator().manual_seed(4))
        mask = mask.to(dtype)
    kw = dict(is_causal=causal, attn_mask=None if mask is None else mask.to("tt"))
    before = native.counters()
    y = _run(q, k, v, **kw)
    routes = _routes(native, before)
    assert len(routes) == 1 and list(routes.values()) == [1], routes
    (kernel,) = routes
    assert kernel == name.split("-")[0]
    with native.sdpa_kernel(kernel):
        again = _run(q, k, v, **kw)
    assert torch.equal(y, again)


# (kernels, (b, h, hk, s, d), dtype, mask kind, causal, message)
REFUSED = [
    (("flash",), (1, 4, 4, 128, 40), torch.bfloat16, None, False, "multiple of 32"),
    (("flash", "own"), (1, 4, 4, 128, 64), torch.float32, None, False, "bf16"),
    (("own",), (1, 4, 4, 128, 64), torch.bfloat16, "bool", False, "own: a mask"),
    (("flash",), (1, 4, 4, 96, 320), torch.bfloat16, None, False, "up to 256"),
    (("ttnn", "own"), (1, 4, 4, 128, 64), torch.bfloat16, None, True, None),
]


@pytest.mark.parametrize("kernels,shape,dtype,kind,causal,message", REFUSED)
def test_refused_call_raises(native, kernels, shape, dtype, kind, causal, message):
    """A call no allowed kernel takes raises, naming the call and the refusal."""
    b, h, hk, s, d = shape
    q, k, v = _qkv(b, h, hk, s, d, 5, dtype)
    mask = None
    if kind == "bool":
        mask = (torch.rand(b, 1, s, s) > 0.2).to("tt")
    if message is None:
        # Causal with Sq != Sk: no fused kernel takes it.
        q = q[:, :, :64]
        message = "causal needs Sq == Sk"
    refused = pytest.raises(RuntimeError, match="no allowed kernel takes attention")
    with refused as err, native.sdpa_kernel(*kernels):
        _run(q, k, v, attn_mask=mask, is_causal=causal)
    assert message in str(err.value)
    assert f"[{b}, {h}, " in str(err.value)


def _train(native, q, k, v, grad, causal=False, mask=None):
    """(out, dq, dk, dv) of one training call on the card."""
    qt, kt, vt = (t.to("tt").requires_grad_() for t in (q, k, v))
    out = F.scaled_dot_product_attention(
        qt,
        kt,
        vt,
        is_causal=causal,
        attn_mask=None if mask is None else mask.to("tt"),
        enable_gqa=q.shape[1] != k.shape[1],
    )
    out.backward(grad.to("tt"))
    return out.detach().cpu(), qt.grad.cpu(), kt.grad.cpu(), vt.grad.cpu()


def _train_reference(q, k, v, grad, causal=False, mask=None):
    qd, kd, vd = (t.double().requires_grad_() for t in (q, k, v))
    rep = q.shape[1] // k.shape[1]
    out = F.scaled_dot_product_attention(
        qd,
        kd.repeat_interleave(rep, 1),
        vd.repeat_interleave(rep, 1),
        attn_mask=mask,
        is_causal=causal,
    )
    out.backward(grad.double())
    return out.detach(), qd.grad, kd.grad, vd.grad


@pytest.mark.parametrize("kernel", ["flash", "ttnn"])
def test_training_runs_forward_and_backward_on_the_kernel(native, kernel):
    q, k, v = _qkv(1, 4, 4, 128, 64, 6)
    grad = torch.randn(1, 4, 128, 64).to(torch.bfloat16)
    before = native.counters()
    with native.sdpa_kernel(kernel):
        got = _train(native, q, k, v, grad, causal=True)
    assert _routes(native, before) == {kernel: 1, f"backward_{kernel}": 1}
    want = _train_reference(q, k, v, grad, causal=True)
    assert _rel(got[0], want[0]) <= BOUND[kernel]
    for y, ref in zip(got[1:], want[1:]):
        assert _rel(y, ref) <= GRAD_BOUND


def test_masked_training_needs_math_for_its_backward(native):
    q, k, v = _qkv(1, 4, 4, 128, 64, 7)
    grad = torch.randn(1, 4, 128, 64).to(torch.bfloat16)
    mask = torch.rand(1, 1, 128, 128, generator=torch.Generator().manual_seed(8)) > 0.2
    refused = pytest.raises(RuntimeError, match="backward runs on 'math'")
    with refused, native.sdpa_kernel("flash"):
        _train(native, q, k, v, grad, mask=mask)
    before = native.counters()
    with native.sdpa_kernel("flash", "math"):
        got = _train(native, q, k, v, grad, mask=mask)
    assert _routes(native, before) == {"flash": 1, "backward_math": 1}
    want = _train_reference(q, k, v, grad, mask=mask)
    for y, ref in zip(got, want):
        assert _rel(y, ref) <= GRAD_BOUND


def test_backward_keeps_the_forward_kernel(native):
    """The scope open at backward time does not change the backward's kernel."""
    q, k, v = _qkv(1, 4, 4, 128, 64, 9)
    qt, kt, vt = (t.to("tt").requires_grad_() for t in (q, k, v))
    before = native.counters()
    with native.sdpa_kernel("flash"):
        out = F.scaled_dot_product_attention(qt, kt, vt)
    with native.sdpa_kernel("ttnn"):
        out.backward(torch.ones_like(out))
    torch.tt.synchronize()
    assert _routes(native, before) == {"flash": 1, "backward_flash": 1}


def test_varlen_needs_flash(native):
    q, k, v = (torch.randn(64, 2, 64).to(torch.bfloat16).to("tt") for _ in range(3))
    refused = pytest.raises(Exception, match="flash only")
    with refused, native.sdpa_kernel("ttnn", "math"):
        native.sdpa_varlen(q, k, v, [0, 30, 64])


def test_scope_reaches_compiled_code(native):
    q, k, v = (t.to("tt") for t in _qkv(1, 4, 4, 128, 64, 10))
    fn = torch.compile(
        lambda a, b, c: F.scaled_dot_product_attention(a, b, c), backend="ktt_tt"
    )
    before = native.counters()
    with native.sdpa_kernel("ttnn"):
        fn(q, k, v)
    fn(q, k, v)
    torch.tt.synchronize()
    assert _routes(native, before) == {"ttnn": 1, "flash": 1}


def test_trace_keeps_the_scope_of_its_capture(native):
    q, k, v = (t.to("tt") for t in _qkv(1, 4, 4, 128, 64, 11))
    with native.sdpa_kernel("ttnn"):
        want = F.scaled_dot_product_attention(q, k, v).cpu()
        step = native.capture(lambda a: F.scaled_dot_product_attention(a, k, v), q)
    assert step.numerics.attention_kernels == ("ttnn",)
    assert torch.equal(step(q).cpu(), want)
