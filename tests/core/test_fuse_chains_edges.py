"""The compile graph pass on CPU, edge cases: chains it must leave, in-place updates
inside a chain's span, dtypes, gradients, SDPA called with keywords, dynamic shapes, a
failing rewrite and the KTT_FUSE_CHAINS values. Outputs always equal eager's."""

import pytest
import torch
import torch._dynamo
import torch.nn.functional as F
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd

import kohakutt
from kohakutt.torch_backend import chains

needs_native = pytest.mark.skipif(
    not kohakutt.AVAILABLE, reason="needs the native build (torch.ops.ktt)"
)

LEVELS = ["dynamo", "aot"]


def compiled(fn, args, level, dynamic=False, grad=False):
    """(counts per graph, eager result, compiled result) of fn(*args)."""
    counts = []

    def fused(gm, example_inputs):
        counts.append(dict(chains.fuse_chains(gm).ktt_chains))
        return make_boxed_func(gm.forward) if level == "aot" else gm.forward

    backend = aot_autograd(fw_compiler=fused) if level == "aot" else fused
    torch._dynamo.reset()
    run = torch.compile(fn, backend=backend, dynamic=dynamic)
    with torch.enable_grad() if grad else torch.no_grad():
        want = fn(*args)
        got = run(*args)
    return counts, want, got


def total(counts, key):
    return sum(c[key] for c in counts)


def close(got, want, tol=2e-2):
    got = got if isinstance(got, tuple) else (got,)
    want = want if isinstance(want, tuple) else (want,)
    for g, w in zip(got, want, strict=True):
        assert g.dtype == w.dtype and g.shape == w.shape
        torch.testing.assert_close(
            g.float(), w.float(), rtol=tol, atol=tol, equal_nan=True
        )


def norm_chain(x, w, keepdim=True, power=2, dim=-1):
    h = x.float()
    h = h * torch.rsqrt(h.pow(power).mean(dim, keepdim=keepdim) + 1e-6)
    return w * h.to(x.dtype)


X = torch.randn(3, 64, generator=torch.Generator().manual_seed(0)).bfloat16()
W = torch.randn(64, generator=torch.Generator().manual_seed(1)).bfloat16()


def keepdim_false(x, w):
    h = x.float()
    return w * (h * torch.rsqrt(h.pow(2).mean(-1) + 1e-6)[:, None]).to(x.dtype)


def pow3(x, w):
    return norm_chain(x, w, power=3)


def other_dim(x, w):
    return norm_chain(x, w, dim=0)


def two_tensors(x, w):
    h, g = x.float(), (x * 2).float()
    return w * (h * torch.rsqrt((h * g).mean(-1, keepdim=True) + 1e-6)).to(x.dtype)


def eps_outside(x, w):
    h = x.float()
    return w * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True)) + 1e-6).to(x.dtype)


def divide_sqrt(x, w):
    h = x.float()
    return w * (h / torch.sqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype)


def mean_dtype(x, w):
    h = x.float()
    var = h.pow(2).mean(-1, keepdim=True, dtype=torch.float64).float()
    return w * (h * torch.rsqrt(var + 1e-6)).to(x.dtype)


def variance_used(x, w):
    h = x.float()
    var = h.pow(2).mean(-1, keepdim=True)
    return w * (h * torch.rsqrt(var + 1e-6)).to(x.dtype), var


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize(
    "fn",
    [
        keepdim_false,
        pow3,
        other_dim,
        two_tensors,
        eps_outside,
        divide_sqrt,
        mean_dtype,
        variance_used,
    ],
)
def test_norm_left(level, fn):
    counts, want, got = compiled(fn, (X, W), level)
    assert total(counts, "rms_norm") == 0
    close(got, want)


@needs_native
@pytest.mark.parametrize("level", LEVELS)
def test_norm_row_weight_outside(level):
    def row_weight(x, w):
        return norm_chain(x, w[None, :])

    counts, want, got = compiled(row_weight, (X, W), level)
    assert total(counts, "rms_norm") == 1
    close(got, want)


@needs_native
@pytest.mark.parametrize("level", LEVELS)
def test_norm_fp32_result_keeps_fp32(level):
    def fp32_out(x):
        h = x.float()
        return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)

    counts, want, got = compiled(fp32_out, (X,), level)
    assert total(counts, "rms_norm") == 1
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_norm_source_updated_in_place_is_left():
    def updated(x, w):
        h = x.float()
        n = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        x.add_(1.0)
        return w * n.to(x.dtype), x

    counts = []

    def fused(gm, example_inputs):
        counts.append(chains.fuse_chains(gm).ktt_chains)
        return gm.forward

    torch._dynamo.reset()
    with torch.no_grad():
        want = updated(X.clone(), W)
        got = torch.compile(updated, backend=fused)(X.clone(), W)
    assert total(counts, "rms_norm") == 0
    close(got, want)


def test_norm_needing_grad_is_left():
    x = X.float().requires_grad_()
    counts, want, got = compiled(norm_chain, (x, W.float()), "dynamo", grad=True)
    assert total(counts, "rms_norm") == 0
    close(got, want, tol=1e-6)


def test_norm_training_grads_equal_eager():
    w = torch.nn.Parameter(W.float())

    def step(x):
        return norm_chain(x, w).square().sum()

    loss = step(X.float())
    loss.backward()
    want = w.grad.clone()
    w.grad = None
    counts = []

    def fused(gm, example_inputs):
        counts.append(chains.fuse_chains(gm).ktt_chains)
        return make_boxed_func(gm.forward)

    torch._dynamo.reset()
    backend = aot_autograd(fw_compiler=fused, bw_compiler=fused)
    torch.compile(step, backend=backend)(X.float()).backward()
    torch.testing.assert_close(w.grad, want, rtol=1e-5, atol=1e-5)


# ----------------------------------------------------------------- attention
@pytest.fixture
def on_tt(monkeypatch):
    """The SDPA rewrites take CPU tensors as tt ones."""
    monkeypatch.setattr(chains, "_on_tt", lambda n: chains._val(n) is not None)


def qkv(heads=4, kv_heads=2, d=64, seq=8):
    g = torch.Generator().manual_seed(2)
    q = torch.randn(1, heads, 1, d, generator=g).bfloat16()
    k = torch.randn(1, kv_heads, seq, d, generator=g).bfloat16()
    v = torch.randn(1, kv_heads, seq, d, generator=g).bfloat16()
    return q, k, v


def repeat_kv(t, n):
    b, h, s, d = t.shape
    return t[:, :, None, :, :].expand(b, h, n, s, d).reshape(b, h * n, s, d)


def where_nodes(fn, args):
    found = []

    def backend(gm, example_inputs):
        chains.fuse_chains(gm)
        found.append(sum(n.target is torch.ops.aten.where.self for n in gm.graph.nodes))
        return gm.forward

    torch._dynamo.reset()
    with torch.no_grad():
        torch.compile(fn, backend=backend)(*args)
    return found[0]


MASK = torch.tensor([[[[True] * 6 + [False] * 2]]])


@pytest.mark.usefixtures("on_tt")
def test_mask_converted_once_for_two_calls():
    def two(q, k, v, m):
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return F.scaled_dot_product_attention(a, k, v, attn_mask=m)

    q, k, v = qkv(heads=2)
    counts, want, got = compiled(two, (q, k, v, MASK), "dynamo")
    assert total(counts, "mask") == 2
    assert where_nodes(two, (q, k, v, MASK)) == 1
    close(got, want)


@pytest.mark.usefixtures("on_tt")
def test_mask_updated_in_place_converted_again():
    def mutated(q, k, v, m):
        m = m.clone()
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        m[..., 1] = False
        return F.scaled_dot_product_attention(a, k, v, attn_mask=m)

    q, k, v = qkv(heads=2)
    counts, want, got = compiled(mutated, (q, k, v, MASK), "dynamo")
    assert where_nodes(mutated, (q, k, v, MASK)) == 2
    close(got, want)


@pytest.mark.usefixtures("on_tt")
@pytest.mark.parametrize("mask", [None, torch.zeros(1, 1, 1, 8).bfloat16()])
def test_float_or_no_mask_untouched(mask):
    def attend(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    counts, want, got = compiled(attend, qkv(heads=2), "dynamo")
    assert total(counts, "mask") == 0
    close(got, want)


@needs_native
@pytest.mark.usefixtures("on_tt")
@pytest.mark.parametrize("keywords", [False, True])
def test_repeat_kv_given_ungrouped(keywords):
    def attend(q, k, v):
        kk, vv = repeat_kv(k, 2), repeat_kv(v, 2)
        if keywords:
            return F.scaled_dot_product_attention(query=q, key=kk, value=vv)
        return F.scaled_dot_product_attention(q, kk, vv)

    counts, want, got = compiled(attend, qkv(), "dynamo")
    assert total(counts, "gqa") == 1
    close(got, want)


def tiled(q, k, v):
    b, h, s, d = k.shape
    kk = k[:, None].expand(b, 2, h, s, d).reshape(b, 2 * h, s, d)
    vv = v[:, None].expand(b, 2, h, s, d).reshape(b, 2 * h, s, d)
    return F.scaled_dot_product_attention(q, kk, vv)


def gathered(q, k, v):
    b, h, s, d = k.shape
    idx = torch.tensor([1, 0])
    kk = k[:, idx, None, :, :].expand(b, h, 2, s, d).reshape(b, 2 * h, s, d)
    return F.scaled_dot_product_attention(q, kk, repeat_kv(v, 2))


def repeat_updated(q, k, v):
    kk = repeat_kv(k, 2)
    kk.mul_(2.0)
    return F.scaled_dot_product_attention(q, kk, repeat_kv(v, 2))


def head_dim_48(q, k, v):
    q, k, v = q[..., :48], k[..., :48], v[..., :48]
    return F.scaled_dot_product_attention(q, repeat_kv(k, 2), repeat_kv(v, 2))


@pytest.mark.usefixtures("on_tt")
@pytest.mark.parametrize("fn", [tiled, gathered, repeat_updated])
def test_other_repeats_left(fn):
    counts, want, got = compiled(fn, qkv(), "dynamo")
    assert total(counts, "gqa") == 0
    close(got, want)


@needs_native
@pytest.mark.usefixtures("on_tt")
def test_padded_head_dim_rewritten():
    # A head dim the attention kernel pads by itself takes grouped heads as well.
    counts, want, got = compiled(head_dim_48, qkv(), "dynamo")
    assert total(counts, "gqa") == 1
    close(got, want)


@pytest.mark.usefixtures("on_tt")
def test_dynamic_shapes_compile():
    def attend(q, k, v):
        return F.scaled_dot_product_attention(q, repeat_kv(k, 2), repeat_kv(v, 2))

    counts, want, got = compiled(attend, qkv(), "dynamo", dynamic=True)
    assert total(counts, "gqa") == 0
    close(got, want)


# ------------------------------------------------------------------ robustness
def test_failing_rewrite_leaves_graph(monkeypatch):
    def broken(graph):
        # Edits the copy, then fails midway.
        graph.erase_node(next(n for n in graph.nodes if n.op == "output"))
        raise RuntimeError("boom")

    fns = [(n, k, broken if n == "norm" else f) for n, k, f in chains.REWRITE_FNS]
    monkeypatch.setattr(chains, "REWRITE_FNS", tuple(fns))
    with pytest.warns(UserWarning, match="'norm' failed"):
        counts, want, got = compiled(norm_chain, (X, W), "dynamo")
    assert total(counts, "rms_norm") == 0
    close(got, want)


@needs_native
def test_second_pass_changes_nothing():
    seen = []

    def twice(gm, example_inputs):
        seen.append(dict(chains.fuse_chains(gm).ktt_chains))
        seen.append(dict(chains.fuse_chains(gm).ktt_chains))
        return gm.forward

    torch._dynamo.reset()
    with torch.no_grad():
        torch.compile(norm_chain, backend=twice)(X, W)
    assert seen[0]["rms_norm"] == 1 and seen[1]["rms_norm"] == 0


@pytest.mark.parametrize(
    "value, want",
    [
        (None, {"norm", "rotary", "mask", "gqa"}),
        ("", {"norm", "rotary", "mask", "gqa"}),
        ("1", {"norm", "rotary", "mask", "gqa"}),
        (" On ", {"norm", "rotary", "mask", "gqa"}),
        ("TRUE", {"norm", "rotary", "mask", "gqa"}),
        ("0", set()),
        ("off", set()),
        ("All", {"norm", "rotary", "mask", "gqa"}),
        ("norm, Mask", {"norm", "mask"}),
        ("all,rotary", {"norm", "rotary", "mask", "gqa"}),
    ],
)
def test_knob_values(monkeypatch, value, want):
    if value is None:
        monkeypatch.delenv("KTT_FUSE_CHAINS", raising=False)
    else:
        monkeypatch.setenv("KTT_FUSE_CHAINS", value)
    assert chains.chosen() == want


@pytest.mark.parametrize("value", ["norm,typo", "norm;mask", "yes"])
def test_knob_unknown_raises(monkeypatch, value):
    monkeypatch.setenv("KTT_FUSE_CHAINS", value)
    with pytest.raises(ValueError, match="KTT_FUSE_CHAINS"):
        chains.chosen()
