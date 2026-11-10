"""Kohaku-TT's flash attention (ktt_flash, the default bf16 route): error against CPU
float64 over batch, MHA / GQA / MQA heads, ragged lengths, head dims 32-256, causal, and
additive / bool masks broadcast over batch and heads; training gradients (dq, dk, dv)
against float64 autograd.

Bound: rel RMS <= 8e-3 against float64. At HiFi2 the matmul keeps 7 of Q's 8 significand
bits and P is bf16, which alone gives about 5.5e-3 on these inputs; CPU bf16 (fp32 math,
bf16 output) is about 1.7e-3.
"""

import pytest
import torch
import torch.nn.functional as F

BOUND = 8e-3

# (b, h, hk, s, d, causal)
CASES = [
    (1, 2, 2, 32, 32, False),
    (1, 2, 2, 256, 64, False),
    (2, 4, 4, 512, 64, True),
    (1, 4, 2, 96, 96, False),
    (3, 4, 1, 300, 128, False),
    (1, 4, 4, 1000, 64, False),
    (1, 4, 4, 1000, 64, True),
    (1, 8, 8, 1024, 128, True),
    (4, 8, 4, 640, 128, False),
    (1, 16, 4, 1024, 128, True),
    (8, 16, 16, 256, 64, False),
    (1, 2, 1, 1100, 256, False),
    (2, 2, 2, 96, 256, True),
]

# (b, h, hk, s, d, mask kind): row = [b, 1, s, s] additive, head = [b, h, s, s]
# additive, bcast = [b, 1, 1, s] additive, bool = [b, 1, s, s] keep mask.
MASK_CASES = [
    (2, 4, 4, 512, 64, "row"),
    (2, 4, 4, 256, 64, "head"),
    (2, 8, 2, 1024, 128, "bcast"),
    (2, 4, 4, 300, 64, "bool"),
]


def _rel(y, ref):
    return ((y.double() - ref).norm() / ref.norm()).item()


def _qkv(b, h, hk, s, d, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, h, s, d, generator=g).to(torch.bfloat16)
    k = torch.randn(b, hk, s, d, generator=g).to(torch.bfloat16)
    v = torch.randn(b, hk, s, d, generator=g).to(torch.bfloat16)
    return q, k, v, g


@pytest.mark.parametrize("b,h,hk,s,d,causal", CASES)
def test_matches_float64(native, b, h, hk, s, d, causal):
    q, k, v, _ = _qkv(b, h, hk, s, d, b + h + s + d)
    rep = h // hk
    kr, vr = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    ref = F.scaled_dot_product_attention(
        q.double(), kr.double(), vr.double(), is_causal=causal
    )
    y = F.scaled_dot_product_attention(
        q.to("tt"), k.to("tt"), v.to("tt"), is_causal=causal, enable_gqa=rep > 1
    ).cpu()
    assert y.shape == (b, h, s, d)
    assert torch.isfinite(y).all()
    assert _rel(y, ref) <= BOUND


@pytest.mark.parametrize("log2_logit", [10, 14, 17])
@pytest.mark.parametrize("s", [192, 201])
def test_large_logits(native, s, log2_logit):
    """Logits of about 2^log2_logit (dinov3 reaches about 1e5): the row reference jumps
    by far more than 127 between key chunks. The output is finite at every fidelity."""
    q, k, v, _ = _qkv(1, 4, 4, s, 64, s + log2_logit)
    m = 2.0 ** (log2_logit / 2)
    q, k = (q.float() * m).bfloat16(), (k.float() * m).bfloat16()
    for fidelity in ("HiFi4", "HiFi2", "LoFi"):
        with native.numerics(fidelity=fidelity):
            y = F.scaled_dot_product_attention(q.to("tt"), k.to("tt"), v.to("tt")).cpu()
        assert torch.isfinite(y).all(), fidelity


@pytest.mark.parametrize("b,h,hk,s,d,kind", MASK_CASES)
def test_masks(native, b, h, hk, s, d, kind):
    q, k, v, g = _qkv(b, h, hk, s, d, b + h + s + d + 1)
    keys = torch.arange(s)
    keep = torch.ones(b, 1, 1, s, dtype=torch.bool)
    for bi in range(b):
        # Even batches padded on the right, odd batches on the left (the first key
        # block masked).
        keep[
            bi, ..., (keys >= int(0.7 * s)) if bi % 2 == 0 else (keys < int(0.4 * s))
        ] = False
    if kind == "bool":
        mask = keep.expand(b, 1, s, s) & (torch.rand(b, 1, s, s, generator=g) >= 0.1)
        bias = torch.zeros(b, 1, s, s, dtype=torch.float64).masked_fill(
            ~mask, float("-inf")
        )
    else:
        rows = 1 if kind == "bcast" else s
        heads = h if kind == "head" else 1
        mask = (0.5 * torch.randn(b, heads, rows, s, generator=g)).to(torch.bfloat16)
        mask = mask.masked_fill(~keep, float("-inf"))
        bias = mask.double()
    rep = h // hk
    kr, vr = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    ref = F.scaled_dot_product_attention(
        q.double(), kr.double(), vr.double(), attn_mask=bias
    )
    y = F.scaled_dot_product_attention(
        q.to("tt"), k.to("tt"), v.to("tt"), attn_mask=mask.to("tt"), enable_gqa=rep > 1
    ).cpu()
    assert torch.isfinite(y).all()
    assert _rel(y, ref) <= BOUND


APPROX_BOUND = (
    3e-2  # approximate exponential (TT-NN SDPA's approximate path: about 2.3e-2)
)

# (b, h, hk, s, d, causal, mask kind or None)
APPROX_CASES = [
    (1, 4, 4, 512, 64, False, None),
    (2, 4, 4, 512, 64, True, None),
    (2, 4, 4, 512, 64, False, "row"),
]


@pytest.mark.parametrize("b,h,hk,s,d,causal,kind", APPROX_CASES)
def test_approx_exp(native, b, h, hk, s, d, causal, kind):
    q, k, v, g = _qkv(b, h, hk, s, d, b + h + s + d + 2)
    mask = None
    bias = None
    if kind == "row":
        mask = (0.5 * torch.randn(b, 1, s, s, generator=g)).to(torch.bfloat16)
        mask[0, ..., int(0.7 * s) :] = float("-inf")
        mask[1, ..., : int(0.4 * s)] = float("-inf")
        bias = mask.double()
    ref = F.scaled_dot_product_attention(
        q.double(), k.double(), v.double(), attn_mask=bias, is_causal=causal
    )
    with native.numerics(approx=True):
        y = F.scaled_dot_product_attention(
            q.to("tt"),
            k.to("tt"),
            v.to("tt"),
            attn_mask=None if mask is None else mask.to("tt"),
            is_causal=causal,
        ).cpu()
    assert torch.isfinite(y).all()
    assert _rel(y, ref) <= APPROX_BOUND


# (b, h): the most (batch, head) pairs of one query tile whose jobs fit a core's runtime
# args on the 110-core grid, then a count past it (today's route: TT-NN SDPA, bf16 Dst,
# approximate exp).
@pytest.mark.parametrize("b,h,bound", [(41, 110, BOUND), (44, 110, 5e-2)])
def test_many_pairs_short_queries(native, b, h, bound):
    q, k, v, _ = _qkv(b, h, h, 32, 32, b + h)
    ref = F.scaled_dot_product_attention(q.double(), k.double(), v.double())
    y = F.scaled_dot_product_attention(q.to("tt"), k.to("tt"), v.to("tt")).cpu()
    assert torch.isfinite(y).all()
    assert _rel(y, ref) <= bound


def _train(q, k, v, dout, causal, rep, device):
    """Output and gradients (dq, dk, dv) of attention trained on `device` ("cpu":
    float64 with K / V repeated over each group)."""
    if device == "cpu":
        qq, kk, vv = (x.double().requires_grad_() for x in (q, k, v))
        y = F.scaled_dot_product_attention(
            qq,
            kk.repeat_interleave(rep, 1),
            vv.repeat_interleave(rep, 1),
            is_causal=causal,
        )
        y.backward(dout.double())
    else:
        qq, kk, vv = (x.to(device).detach().requires_grad_() for x in (q, k, v))
        y = F.scaled_dot_product_attention(
            qq, kk, vv, is_causal=causal, enable_gqa=rep > 1
        )
        y.backward(dout.to(device))
    return [t.detach().cpu() for t in (y, qq.grad, kk.grad, vv.grad)]


# Backward (the training forward keeps each row's log-sum-exp; the backward recomputes
# P). Bound: rel RMS <= BW_BOUND per gradient against float64. At HiFi2 the matrix unit
# keeps 7 of Q's and dO's 8 significand bits in S and dP, and P and dS are bf16
# operands. A CPU emulation of exactly these operand precisions gives dq / dk 1.0e-2 -
# 1.1e-2 and dv 6.6e-3 - 7.9e-3 with the device's own log-sum-exp (1.2e-2 - 1.4e-2 and
# 8.8e-3 - 1.1e-2 with the float64 one); the device measures the same. CPU bf16:
# 2.4e-3 - 4.6e-3. After an approximate-exp forward: 1.3e-2 - 1.9e-2 (APPROX_BOUND).
BW_BOUND = 2e-2

# Log-sum-exp the training forward returns, against float64 (nats): Q's truncation in S
# moves it by about 3e-3 RMS (CPU emulation).
LSE_RMS_BOUND = 6e-3

# (b, h, hk, s, d, causal)
BW_CASES = [
    (1, 2, 2, 32, 32, False),
    (2, 4, 4, 256, 64, False),
    (2, 4, 4, 512, 64, True),
    (1, 8, 2, 512, 128, False),
    (1, 8, 2, 384, 64, True),
    (2, 4, 1, 256, 64, False),
    (1, 4, 4, 1024, 128, True),
    (1, 2, 2, 256, 256, False),
]


@pytest.mark.parametrize("b,h,hk,s,d,causal", BW_CASES)
def test_backward_matches_float64(native, b, h, hk, s, d, causal):
    q, k, v, g = _qkv(b, h, hk, s, d, b + h + s + d + 3)
    dout = torch.randn(b, h, s, d, generator=g).to(torch.bfloat16)
    rep = h // hk
    ref = _train(q, k, v, dout, causal, rep, "cpu")
    got = _train(q, k, v, dout, causal, rep, "tt")
    for name, x, r in zip(("out", "dq", "dk", "dv"), got, ref):
        assert x.shape == r.shape, name
        assert torch.isfinite(x).all(), name
        assert _rel(x, r) <= (BOUND if name == "out" else BW_BOUND), name


# Lengths that are not a multiple of 32 (one ragged last tile of queries and of keys).
# (b, h, s, d, causal)
RAGGED_BW_CASES = [
    (1, 2, 40, 32, False),
    (1, 2, 40, 32, True),
    (1, 2, 31, 64, False),
    (2, 12, 201, 64, False),
    (1, 4, 201, 64, True),
]


def _padded(x, value):
    """x on the device with `value` in its tile padding (0 at the values of 1 -
    ones)."""
    ones = torch.ones(x.shape, dtype=x.dtype).to("tt")
    return x.to("tt") + (1.0 - ones) * value


@pytest.mark.parametrize("b,h,s,d,causal", RAGGED_BW_CASES)
def test_backward_ragged_length(native, b, h, s, d, causal):
    """Gradients at a ragged length against float64, on the fused route (one forward and
    two backward programs), and unchanged bit for bit by what the tile padding of q, k,
    v and the incoming gradient holds."""
    q, k, v, g = _qkv(b, h, h, s, d, b + h + s + d + 5)
    dout = torch.randn(b, h, s, d, generator=g).to(torch.bfloat16)
    ref = _train(q, k, v, dout, causal, 1, "cpu")
    got = {}
    for pad in (0.0, 3.0):
        qq, kk, vv = (_padded(x, pad).detach().requires_grad_() for x in (q, k, v))
        grad = _padded(dout, pad).detach()
        native.synchronize()
        before = native.counters()["programs"]
        y = F.scaled_dot_product_attention(qq, kk, vv, is_causal=causal)
        y.backward(grad)
        native.synchronize()
        programs = native.counters()["programs"] - before
        got[pad] = [t.detach().cpu() for t in (y, qq.grad, kk.grad, vv.grad)]
        assert programs == 3, (pad, programs)
    for name, x, r in zip(("out", "dq", "dk", "dv"), got[0.0], ref):
        assert x.shape == r.shape, name
        assert torch.isfinite(x).all(), name
        assert _rel(x, r) <= (BOUND if name == "out" else BW_BOUND), name
    for name, x, y in zip(("dq", "dk", "dv"), got[0.0][1:], got[3.0][1:]):
        assert torch.equal(x, y), name


@pytest.mark.parametrize("causal", [False, True])
def test_training_log_sum_exp(native, causal):
    b, h, s, d = 2, 4, 512, 64
    q, k, v, _ = _qkv(b, h, h, s, d, 23)
    scores = q.double() @ k.double().transpose(-1, -2) / d**0.5
    if causal:
        scores = scores.masked_fill(torch.ones(s, s).triu(1).bool(), float("-inf"))
    ref = torch.logsumexp(scores, -1)
    qq, kk, vv = (x.to("tt").detach().requires_grad_() for x in (q, k, v))
    res = torch.ops.aten._scaled_dot_product_fused_attention_overrideable(
        qq, kk, vv, None, 0.0, causal, False
    )
    lse = res[1].detach().cpu().double()
    assert lse.shape == (b, h, s)
    assert (lse - ref).pow(2).mean().sqrt().item() <= LSE_RMS_BOUND


@pytest.mark.parametrize("causal", [False, True])
def test_backward_after_approx_forward(native, causal):
    b, h, s, d = 2, 4, 512, 64
    q, k, v, g = _qkv(b, h, h, s, d, 17)
    dout = torch.randn(b, h, s, d, generator=g).to(torch.bfloat16)
    ref = _train(q, k, v, dout, causal, 1, "cpu")
    with native.numerics(approx=True):
        got = _train(q, k, v, dout, causal, 1, "tt")
    for name, x, r in zip(("out", "dq", "dk", "dv"), got, ref):
        assert torch.isfinite(x).all(), name
        assert _rel(x, r) <= APPROX_BOUND, name


def test_backward_masked_keys_exactly_zero(native):
    """Causal, dO zero from query 80 on: keys from 80 on are seen only by masked scores
    or by queries without gradient, so their dK and dV are exactly zero (masked P
    exactly 0).
    """
    b, h, s, d = 2, 4, 256, 64
    q, k, v, g = _qkv(b, h, h, s, d, 11)
    dout = torch.randn(b, h, s, d, generator=g).to(torch.bfloat16)
    dout[:, :, 80:] = 0
    _, dq, dk, dv = _train(q, k, v, dout, True, 1, "tt")
    assert (dk[:, :, 80:] == 0).all()
    assert (dv[:, :, 80:] == 0).all()
    assert (dq[:, :, 80:] == 0).all()
    assert (dk[:, :, :80] != 0).any()


def test_backward_no_mixing_across_pairs(native):
    """B x H = 128 pairs, each with its own value and output-gradient scale: a gradient
    taking another pair's data lands far outside the bound."""
    b, h, s, d = 8, 16, 256, 64
    q, k, v, g = _qkv(b, h, h, s, d, 13)
    scale = (1.0 + torch.arange(b * h, dtype=torch.float32)).reshape(b, h, 1, 1)
    v = (v.float() * scale).to(torch.bfloat16)
    dout = (torch.randn(b, h, s, d, generator=g) / scale).to(torch.bfloat16)
    ref = _train(q, k, v, dout, False, 1, "cpu")
    got = _train(q, k, v, dout, False, 1, "tt")
    for x, r in zip(got[1:], ref[1:]):
        per_pair = (x.double() - r).flatten(2).norm(dim=2) / r.flatten(2).norm(dim=2)
        assert per_pair.max().item() <= BW_BOUND


def test_later_kernels_see_default_constants(native, monkeypatch):
    """ktt_flash programs (forward, training forward and backward) end with the vector
    unit's shared constant (LReg 11 = -1) restored: the attention training kernels of
    the KTT_FLASH=0 route, which read it, run right after them on the same cores and
    stay finite and within the bound of float64."""
    b, h, s, d = 2, 4, 256, 64
    q, k, v, g = _qkv(b, h, h, s, d, 19)
    dout = torch.randn(b, h, s, d, generator=g).to(torch.bfloat16)
    F.scaled_dot_product_attention(q.to("tt"), k.to("tt"), v.to("tt")).cpu()
    _train(q, k, v, dout, False, 1, "tt")
    monkeypatch.setenv("KTT_FLASH", "0")
    ref = _train(q, k, v, dout, False, 1, "cpu")
    got = _train(q, k, v, dout, False, 1, "tt")
    for name, x, r in zip(("out", "dq", "dk", "dv"), got, ref, strict=True):
        assert torch.isfinite(x).all(), name
        assert _rel(x, r) <= BOUND, name


def test_tt_train_backward_reads_flash_statistics(native, monkeypatch):
    """The training forward's statistics have tt-train's layout: with KTT_FLASH=0 set
    after a ktt_flash forward, tt-train's backward runs on them and stays within the
    bound of float64.
    """
    b, h, s, d = 2, 4, 256, 64
    q, k, v, g = _qkv(b, h, h, s, d, 21)
    dout = torch.randn(b, h, s, d, generator=g).to(torch.bfloat16)
    ref = _train(q, k, v, dout, False, 1, "cpu")
    qq, kk, vv = (x.to("tt").detach().requires_grad_() for x in (q, k, v))
    y = F.scaled_dot_product_attention(qq, kk, vv)
    torch.tt.synchronize()
    monkeypatch.setenv("KTT_FLASH", "0")
    y.backward(dout.to("tt"))
    got = [t.detach().cpu() for t in (y, qq.grad, kk.grad, vv.grad)]
    for name, x, r in zip(("out", "dq", "dk", "dv"), got, ref, strict=True):
        assert torch.isfinite(x).all(), name
        assert _rel(x, r) <= BW_BOUND, name


@pytest.mark.parametrize("causal", [False, True])
def test_training_statistics_fill_every_column(native, causal):
    """The training forward's statistics [B, H, S, 32] hold row r's log-sum-exp in every
    column of row r (each column sums the row's partials in its own order: equal within
    fp32 rounding).
    """
    b, h, s, d = 2, 4, 512, 64
    q, k, v, _ = _qkv(b, h, h, s, d, 29)
    qq, kk, vv = (x.to("tt").detach().requires_grad_() for x in (q, k, v))
    lse = torch.ops.aten._scaled_dot_product_fused_attention_overrideable(
        qq, kk, vv, None, 0.0, causal, False
    )[1].detach()
    stats = lse.as_strided((b, h, s, 32), (h * s * 32, s * 32, 32, 1), 0).cpu()
    assert torch.isfinite(stats).all()
    assert (
        stats - stats[..., :1]
    ).abs().max().item() <= 4e-6 * stats.abs().max().item()


# Key-row orientation of the backward's dK / dV program (KTT_FLASH_BWD_ORIENT=kq, read
# per call), at HiFi4 (every operand bit used) on ordinary logits;
# docs/reference/kernels/flash.md gives its limit at large logits.
# (b, h, hk, s, d, causal)
KEY_ROW_CASES = [
    (2, 4, 4, 256, 64, False),
    (1, 4, 4, 384, 64, True),
    (1, 8, 2, 256, 128, False),
]


@pytest.mark.parametrize("b,h,hk,s,d,causal", KEY_ROW_CASES)
def test_backward_key_rows_at_hifi4(native, monkeypatch, b, h, hk, s, d, causal):
    q, k, v, g = _qkv(b, h, hk, s, d, b + h + s + d + 5)
    dout = torch.randn(b, h, s, d, generator=g).to(torch.bfloat16)
    rep = h // hk
    ref = _train(q, k, v, dout, causal, rep, "cpu")
    with native.numerics(fidelity="HiFi4"):
        natural = _train(q, k, v, dout, causal, rep, "tt")
        monkeypatch.setenv("KTT_FLASH_BWD_ORIENT", "kq")
        got = _train(q, k, v, dout, causal, rep, "tt")
    for name, x, r, n in zip(("out", "dq", "dk", "dv"), got, ref, natural, strict=True):
        assert torch.isfinite(x).all(), name
        assert _rel(x, r) <= (BOUND if name == "out" else BW_BOUND), name
        # The two orientations differ by operand roles and summation order only.
        assert _rel(x, n.double()) <= BW_BOUND, name
    # dQ's program does not depend on the orientation.
    assert torch.equal(got[1], natural[1])


def _large(b, h, hk, s, d, seed):
    """q, k, v, dO with dinov3-like scores (raw q.k up to about 1e6, scale x row max up
    to about 1.3e5): the even queries carry a massive component (about 300) on channels
    0-7, four sink keys share the value 251 there (its last bit is one, so 7-bit
    truncation moves the scores).
    """
    q, k, v, g = _qkv(b, h, hk, s, d, seed)
    big = 300 * (1 + 0.2 * torch.randn(b, h, s, 1, generator=g))
    q = q.float()
    q[:, :, ::2, :8] += big[:, :, ::2]
    k[:, :, [0, 5, 77, s - 3], :8] = 251.0
    dout = torch.randn(b, h, s, d, generator=g).to(torch.bfloat16)
    return q.to(torch.bfloat16), k, v, dout


def _keep(x, bits):
    """bf16 x truncated to its top `bits` significant bits (the matrix unit's operand at
    a fidelity), as float64."""
    m = (1 << (8 - bits)) - 1
    return (x.view(torch.int16) & ~m).view(torch.bfloat16).double()


# Large scores, default backward orientation. Reference: float64 SDPA on the operands
# the matrix unit keeps for S at that fidelity (row operand Q: 7 significant bits below
# HiFi4; column operand K: 5 bits at LoFi). Measured (dense / causal): HiFi2 out
# 5.1e-3 / 4.2e-3, dk 8.4e-3 / 1.4e-2, dv 6.7e-3 / 6.3e-3, dq on channels 8+
# 1.1e-2 / 1.2e-2; LoFi out 2.3e-2, dk 5.2e-2 / 7.6e-2, dv 2.7e-2 / 2.6e-2, dq on
# channels 8+ 3.3e-2 / 4.1e-2. dq on the massive channels 0-7 is a sum of bf16 dS over
# the sinks that cancels to nearly zero: 0.13 - 0.36 at HiFi2 / HiFi4
# (LARGE_DQ_MASSIVE_BOUND), not bounded at LoFi (K keeps 5 bits there).
LARGE_FIDELITY_BITS = {"LoFi": (7, 5), "HiFi2": (7, 8), "HiFi4": (8, 8)}
LARGE_DQ_MASSIVE_BOUND = 0.5
# Per fidelity: out and dv, dk, dq on channels 8+.
LARGE_BOUNDS = {
    "LoFi": (5e-2, 0.15, 8e-2),
    "HiFi2": (BW_BOUND,) * 3,
    "HiFi4": (BW_BOUND,) * 3,
}


@pytest.mark.parametrize("fidelity", ["LoFi", "HiFi2", "HiFi4"])
@pytest.mark.parametrize("approx", [False, True])
def test_large_scores_forward(native, fidelity, approx):
    b, h, hk, s, d = 1, 4, 4, 201, 64
    q, k, v, _ = _large(b, h, hk, s, d, 31)
    bq, bk = LARGE_FIDELITY_BITS[fidelity]
    ref = F.scaled_dot_product_attention(_keep(q, bq), _keep(k, bk), v.double())
    with native.numerics(fidelity=fidelity, approx=approx):
        y = F.scaled_dot_product_attention(q.to("tt"), k.to("tt"), v.to("tt")).cpu()
    assert torch.isfinite(y).all()
    assert _rel(y, ref) <= max(
        LARGE_BOUNDS[fidelity][0], APPROX_BOUND if approx else 0.0
    )


@pytest.mark.parametrize("fidelity", ["LoFi", "HiFi2", "HiFi4"])
@pytest.mark.parametrize("causal", [False, True])
def test_large_scores_training(native, fidelity, causal):
    b, h, hk, s, d = 1, 4, 4, 256, 64
    q, k, v, dout = _large(b, h, hk, s, d, 37)
    bq, bk = LARGE_FIDELITY_BITS[fidelity]
    qq, kk, vv = (x.double().requires_grad_() for x in (_keep(q, bq), _keep(k, bk), v))
    y = F.scaled_dot_product_attention(qq, kk, vv, is_causal=causal)
    y.backward(dout.double())
    ref = [y.detach(), qq.grad, kk.grad, vv.grad]
    with native.numerics(fidelity=fidelity):
        got = _train(q, k, v, dout, causal, 1, "tt")
    out_bound, dk_bound, dq_bound = LARGE_BOUNDS[fidelity]
    for name, x, r in zip(("out", "dq", "dk", "dv"), got, ref, strict=True):
        assert torch.isfinite(x).all(), name
        if name == "dq":
            assert _rel(x[..., 8:], r[..., 8:]) <= dq_bound, name
            if fidelity != "LoFi":
                assert _rel(x[..., :8], r[..., :8]) <= LARGE_DQ_MASSIVE_BOUND, name
        else:
            assert _rel(x, r) <= (dk_bound if name == "dk" else out_bound), name


# Forward's against backward's attention weights at large logits, without a reference:
# with an all-ones output gradient dv[k, :] = c_k = sum_q P[q, k], and
# sum_q out[q] = sum_k c_k v_k. Measured rel difference (default orientation):
# 3.6e-3 - 5.4e-3 at HiFi2, 1.8e-2 at LoFi (K's and P's 5-bit operand slices); P's rows
# sum to 1 within 2e-3 / 1.4e-2.
P_CONSISTENCY_BOUND = {"HiFi2": (2e-2, 1e-2), "LoFi": (5e-2, 4e-2)}


@pytest.mark.parametrize("log2_logit", [14, 17])
@pytest.mark.parametrize("fidelity", ["HiFi2", "LoFi"])
def test_large_logits_training(native, fidelity, log2_logit):
    """Logits of about 2^log2_logit: the gradients are finite and the backward rebuilds
    the forward's attention weights (default orientation: the same scores, bit for
    bit)."""
    b, h, s, d = 1, 4, 192, 64
    q, k, v, _ = _qkv(b, h, h, s, d, s + log2_logit)
    m = 2.0 ** (log2_logit / 2)
    q, k = (q.float() * m).bfloat16(), (k.float() * m).bfloat16()
    ones = torch.ones(b, h, s, d, dtype=torch.bfloat16)
    with native.numerics(fidelity=fidelity):
        y, dq, dk, dv = _train(q, k, v, ones, False, 1, "tt")
    for name, x in zip(("out", "dq", "dk", "dv"), (y, dq, dk, dv), strict=True):
        assert torch.isfinite(x).all(), name
    columns = dv.double()[..., :1]
    weights_bound, sum_bound = P_CONSISTENCY_BOUND[fidelity]
    assert _rel((columns * v.double()).sum(-2), y.double().sum(-2)) <= weights_bound
    assert abs(columns.sum().item() / (b * h * s) - 1.0) <= sum_bound


def test_no_mixing_across_pairs(native):
    """Every (batch, head) pair gets its own value scale: an output row taking another
    pair's keys or values lands far outside the bound."""
    b, h, s, d = 8, 16, 512, 64
    q, k, v, _ = _qkv(b, h, h, s, d, 7)
    scale = (1.0 + torch.arange(b * h, dtype=torch.float32)).reshape(b, h, 1, 1)
    v = (v.float() * scale).to(torch.bfloat16)
    ref = F.scaled_dot_product_attention(q.double(), k.double(), v.double())
    y = F.scaled_dot_product_attention(q.to("tt"), k.to("tt"), v.to("tt")).cpu()
    per_pair = (y.double() - ref).flatten(2).norm(dim=2) / ref.flatten(2).norm(dim=2)
    assert per_pair.max().item() <= BOUND
