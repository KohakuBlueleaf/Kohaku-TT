"""Regression: the compile backends rewrite RMSNorm and rotate-half RoPE chains written
in model code (Qwen2 / Llama, Gemma and Qwen3 forms) to one fused op each; results are
no worse than CPU's at the same dtype. KTT_FUSE_CHAINS=0 and other graphs: untouched."""

import pytest
import torch
import torch._dynamo

from kohakutt.torch_backend import compiler as tt_compile

BACKENDS = ["ktt_tt", "ktt_tt_aot"]
# Relative RMS of tt may exceed CPU's by this factor (as tests/hardware/ops/test_norm).
SLACK = 1.02


def _rel(a, ref):
    a, ref = a.double().flatten(), ref.double().flatten()
    return ((a - ref).norm() / ref.norm().clamp_min(1e-300)).item()


def qwen_norm(x, w):
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
    return w * h.to(x.dtype)


def gemma_norm(x, w):
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
    return (h * (1.0 + w.float())).type_as(x)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rope(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, (k * cos) + (rotate_half(k) * sin)


def _compiled(fn, backend, monkeypatch, counts):
    def counted(gm):
        out = fuse(gm)
        counts.append({k: gm.ktt_chains[k] for k in ("rms_norm", "rotary")})
        return out

    fuse = tt_compile.fuse_chains
    monkeypatch.setattr(tt_compile, "fuse_chains", counted)
    tt_compile.register()
    torch._dynamo.reset()
    return torch.compile(fn, backend=backend, dynamic=False)


def _check(fn, args, backend, monkeypatch, want, values=True):
    counts = []
    run = _compiled(fn, backend, monkeypatch, counts)
    with torch.no_grad():
        for _ in range(4):  # eager calls, capture, replays
            got = run(*[a.to("tt") for a in args])
    tt_compile.release()
    assert counts and counts[0] == want, counts
    if not values:
        return
    got = got if isinstance(got, tuple) else (got,)
    ref = fn(*[a.double() for a in args])
    cpu = fn(*args)
    ref, cpu = (ref, cpu) if isinstance(ref, tuple) else ((ref,), (cpu,))
    for g, c, r in zip(got, cpu, ref, strict=True):
        assert g.dtype == c.dtype
        t, e = _rel(g.cpu(), r), _rel(c, r)
        floor = 2e-6 if c.dtype == torch.float32 else 0.0
        assert t <= SLACK * e + floor, (t, e)


@pytest.mark.hw
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 1, 896), (2, 7, 128), (1, 1, 16, 128)])
@pytest.mark.parametrize("norm", [qwen_norm, gemma_norm])
def test_rms_norm_chain(native, monkeypatch, backend, dtype, shape, norm):
    g = torch.Generator().manual_seed(shape[-1] + len(shape))
    x = (torch.randn(shape, generator=g) * 2).to(dtype)
    w = (torch.randn(shape[-1], generator=g) * 0.5 + 1).to(dtype)
    _check(norm, (x, w), backend, monkeypatch, {"rms_norm": 1, "rotary": 0})


@pytest.mark.hw
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "heads, kv_heads, tokens, dim", [(14, 2, 1, 64), (16, 8, 1, 128), (4, 4, 7, 64)]
)
def test_rotary_chain(native, monkeypatch, backend, heads, kv_heads, tokens, dim):
    monkeypatch.setenv("KTT_FUSE_CHAINS", "all")
    g = torch.Generator().manual_seed(heads + tokens)
    q = torch.randn(1, heads, tokens, dim, generator=g).bfloat16()
    k = torch.randn(1, kv_heads, tokens, dim, generator=g).bfloat16()
    angle = torch.rand(1, tokens, dim // 2, generator=g) * 6
    angle = torch.cat((angle, angle), dim=-1)
    cos, sin = angle.cos().bfloat16(), angle.sin().bfloat16()
    _check(rope, (q, k, cos, sin), backend, monkeypatch, {"rms_norm": 0, "rotary": 2})


def norm_over_rows(x, w):
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(0, keepdim=True) + 1e-6)
    return w * h.to(x.dtype)


@pytest.mark.hw
@pytest.mark.parametrize("backend", BACKENDS)
def test_other_graphs_and_knob_untouched(native, monkeypatch, backend):
    g = torch.Generator().manual_seed(5)
    x, w = torch.randn(4, 64, generator=g), torch.randn(64, generator=g)
    none = {"rms_norm": 0, "rotary": 0}
    # Unfused chains: only the rewrite count is checked.
    _check(norm_over_rows, (x, w), backend, monkeypatch, none, values=False)
    monkeypatch.setenv("KTT_FUSE_CHAINS", "0")
    _check(qwen_norm, (x, w), backend, monkeypatch, none, values=False)
