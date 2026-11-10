"""A numpy model of the core binding's API (names, shape rules, dtype codes) for
testing the executor without a card. bf16 rounds like the device; the rest is float64.
"""

from __future__ import annotations

import builtins

import numpy as np

from kohakutt.tinygrad_backend.core import (  # noqa: F401  (re-exported constants)
    BF16,
    BFP4,
    BFP8,
    BINARY,
    FLAT,
    FP32,
    INT32,
    ITEMSIZE,
    NCHW,
    NHWC,
    UNARY,
    CoreError,
)

_open = False
_counters = {"uploads": 0, "downloads": 0, "ops": 0}
NP = {BF16: np.float32, FP32: np.float32, INT32: np.int32}


def _bf16_round(x: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(x, np.float32).view(np.uint32)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return rounded.view(np.float32)


def _pad2(shape):
    shape = tuple(int(s) for s in shape)
    return (1,) * (2 - len(shape)) + shape if len(shape) < 2 else shape


class Tensor:
    def __init__(self, data: np.ndarray, dtype: int):
        self.data, self.dtype = data, dtype
        self._shape = tuple(data.shape)

    @property
    def shape(self):
        return self._shape

    @property
    def numel(self):
        return int(np.prod(self._shape, dtype=np.int64))


def _mk(arr, dtype) -> Tensor:
    _counters["ops"] += 1
    arr = np.asarray(arr)
    arr = _bf16_round(arr) if dtype == BF16 else arr.astype(NP[dtype])
    return Tensor(arr.reshape(_pad2(arr.shape)), dtype)


def open(ids=(0,), rows=1, cols=1):
    global _open
    _open = True


def close():
    global _open
    _open = False


def is_open():
    return _open


def synchronize():
    pass


def counters():
    return {
        **{
            k: 0
            for k in (
                "op_ns",
                "programs",
                "desc_hits",
                "kernel_compiles",
                "program_entries",
            )
        },
        **_counters,
    }


def memory():
    return {"allocated": 0, "peak": 0}


def clear_program_cache():
    pass


class numerics:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *exc):
        pass


def upload(array, dtype=FP32, fmt=NCHW):
    _counters["uploads"] += 1
    array = np.asarray(array)
    if dtype == BF16:
        array = (
            (array.astype(np.uint32) << 16).view(np.float32)
            if array.dtype == np.uint16
            else array
        )
    return Tensor(
        (
            _bf16_round(array).reshape(_pad2(array.shape))
            if dtype == BF16
            else array.astype(NP[dtype]).reshape(_pad2(array.shape))
        ),
        dtype,
    )


def download(t, chip=0):
    _counters["downloads"] += 1
    if t.dtype == BF16:
        return (np.ascontiguousarray(t.data).view(np.uint32) >> 16).astype(np.uint16)
    return t.data.copy()


def empty(shape, dtype=FP32, fmt=NCHW):
    return Tensor(np.zeros(_pad2(shape), NP[dtype]), dtype)


def clone(t):
    return Tensor(t.data.copy(), t.dtype)


def copy_into(dst, src):
    dst.data[...] = src.data


def full(shape, value, dtype=FP32):
    return _mk(np.full(_pad2(shape), value), dtype)


def _draw(shape, dtype, seed, low=0.0, high=1.0):
    """Uniforms in [low, high), a fixed function of `seed` (bf16 rounded down)."""
    u = np.random.default_rng(seed).random(_pad2(shape), dtype=np.float32)
    u = low + (high - low) * u
    if dtype == BF16:
        bits = u.view(np.uint32) & np.uint32(0xFFFF0000)
        return Tensor(bits.view(np.float32), BF16)
    return _mk(u, dtype)


def uniform(shape, dtype, low, high, seed):
    return _draw(shape, dtype, seed, low, high)


def normal(shape, dtype, seed):
    z = np.random.default_rng(seed).standard_normal(_pad2(shape))
    return _mk(z.astype(np.float32), dtype)


def philox_uniform(shape, dtype, key, word=0):
    return _draw(shape, dtype, [key, word])


def reshape(t, sizes):
    if int(np.prod(sizes)) != t.numel:
        raise CoreError(f"reshape: cannot reshape {t.shape} to {tuple(sizes)}")
    return _mk(t.data.reshape(_pad2(sizes)), t.dtype)


def permute(t, dims):
    return _mk(t.data.transpose(dims), t.dtype)


def slice(t, begins, ends, steps=None):
    steps = steps or [1] * len(begins)
    idx = tuple(builtins.slice(b, e, s) for b, e, s in zip(begins, ends, steps))
    return _mk(t.data[idx], t.dtype)


def concat(parts, dim):
    return _mk(np.concatenate([p.data for p in parts], dim), parts[0].dtype)


def repeat(t, repeats):
    return _mk(np.tile(t.data, repeats), t.dtype)


def typecast(t, dtype):
    return _mk(t.data, dtype)


_UN = {
    "neg": np.negative,
    "abs": np.abs,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "rsqrt": lambda x: 1 / np.sqrt(x),
    "reciprocal": lambda x: 1 / x,
    "tanh": np.tanh,
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "relu": lambda x: np.maximum(x, 0),
    "sin": np.sin,
    "cos": np.cos,
    "exp2": np.exp2,
    "log2": np.log2,
    "trunc": np.trunc,
    "floor": np.floor,
    "ceil": np.ceil,
    "gelu": lambda x: 0.5
    * x
    * (1 + np.vectorize(__import__("math").erf)(x / np.sqrt(2))),
    "silu": lambda x: x / (1 + np.exp(-x)),
    "sign": np.sign,
}
_BI = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "pow": np.power,
    "eq": np.equal,
    "ne": np.not_equal,
    "lt": np.less,
    "le": np.less_equal,
    "gt": np.greater,
    "ge": np.greater_equal,
    "bit_and": np.bitwise_and,
    "bit_or": np.bitwise_or,
    "bit_xor": np.bitwise_xor,
    "atan2": np.arctan2,
}
_CMP = {"eq", "ne", "lt", "le", "gt", "ge"}


def unary(op, a):
    if op not in UNARY or op not in _UN:
        raise CoreError(f"unary {op}")
    with np.errstate(all="ignore"):
        return _mk(_UN[op](a.data.astype(np.float64)), a.dtype)


def binary(op, a, b):
    with np.errstate(all="ignore"):
        try:
            r = _BI[op](a.data.astype(np.float64), b.data.astype(np.float64))
        except ValueError as e:
            raise CoreError(f"binary {op}: {e}") from e
    return _mk(r.astype(np.float64), BF16 if op in _CMP else a.dtype)


def binary_scalar(op, a, s):
    with np.errstate(all="ignore"):
        r = _BI[op](a.data.astype(np.float64), float(s))
    return _mk(r, BF16 if op in _CMP else a.dtype)


def where(cond, a, b):
    if not (cond.shape == a.shape == b.shape):
        raise CoreError(
            f"where needs equal shapes, got {cond.shape} {a.shape} {b.shape}"
        )
    return _mk(np.where(cond.data != 0, a.data, b.data), a.dtype)


def clamp(a, lo=None, hi=None):
    return _mk(np.clip(a.data, lo, hi), a.dtype)


def _reduce(fn, a, dims, out):
    r = fn(a.data.astype(np.float64), axis=tuple(dims), keepdims=True)
    return _mk(r.reshape(_pad2(out)), a.dtype)


def sum(a, dims, out_shape):
    return _reduce(np.sum, a, dims, out_shape)


def amax(a, dims, out_shape):
    return _reduce(np.max, a, dims, out_shape)


def softmax(a, dim):
    x = a.data.astype(np.float64)
    e = np.exp(x - x.max(dim, keepdims=True))
    return _mk(e / e.sum(dim, keepdims=True), a.dtype)


def cumsum(a, dim):
    return _mk(np.cumsum(a.data.astype(np.float64), dim), a.dtype)


def argmax_last(a, out_shape):
    return _mk(np.argmax(a.data, -1).reshape(_pad2(out_shape)), INT32)


def gather(a, dim, index):
    return _mk(np.take_along_axis(a.data, index.data.astype(np.int64), dim), a.dtype)


def version(t):
    return id(t)


def embedding_rows(table, version, index, count):
    return _mk(table.data[index.data.reshape(-1)[:count].astype(np.int64)], table.dtype)


def mm(a, b, transpose_a=False, transpose_b=False, out_dtype=None, fast=False):
    x, y = a.data.astype(np.float64), b.data.astype(np.float64)
    if transpose_a:
        x = np.swapaxes(x, -1, -2)
    if transpose_b:
        y = np.swapaxes(y, -1, -2)
    return _mk(np.matmul(x, y), a.dtype if out_dtype is None else out_dtype)


def linear(x, w, bias=None, kn=False):
    r = x.data.astype(np.float64) @ (w.data if kn else w.data.T).astype(np.float64)
    if bias is not None:
        r = r + bias.data
    return _mk(r, x.dtype)


# ---- fused ops (reference formulas)


def layer_norm(x, normalized, eps, gamma=None, beta=None):
    d = x.data.astype(np.float64)
    axes = tuple(range(d.ndim - normalized, d.ndim))
    mean = d.mean(axes, keepdims=True)
    rstd = 1 / np.sqrt(((d - mean) ** 2).mean(axes, keepdims=True) + eps)
    y = (d - mean) * rstd
    if gamma is not None:
        y = y * gamma.data
    if beta is not None:
        y = y + beta.data
    return [_mk(y, x.dtype), _mk(mean, FP32), _mk(rstd, FP32)]


def layer_norm_backward(
    grad,
    x,
    mean,
    rstd,
    normalized,
    gamma,
    want_input=True,
    want_gamma=True,
    want_beta=True,
):
    g, d = grad.data.astype(np.float64), x.data.astype(np.float64)
    axes = tuple(range(d.ndim - normalized, d.ndim))
    xh = (d - mean.data) * rstd.data
    gg = g * (gamma.data if gamma is not None else 1)
    dx = rstd.data * (
        gg - gg.mean(axes, keepdims=True) - xh * (gg * xh).mean(axes, keepdims=True)
    )
    lead = tuple(range(d.ndim - normalized))
    return [
        _mk(dx, x.dtype) if want_input else None,
        (
            _mk(
                (g * xh)
                .sum(lead, keepdims=False)
                .reshape(_pad2(d.shape[-normalized:])),
                x.dtype,
            )
            if want_gamma
            else None
        ),
        (
            _mk(g.sum(lead).reshape(_pad2(d.shape[-normalized:])), x.dtype)
            if want_beta
            else None
        ),
    ]


def rms_norm(x, eps, gamma=None, beta=None):
    d = x.data.astype(np.float64)
    y = d / np.sqrt((d**2).mean(-1, keepdims=True) + eps)
    if gamma is not None:
        y = y * gamma.data
    if beta is not None:
        y = y + beta.data
    return _mk(y, x.dtype)


def rms_norm_train(x, eps, gamma=None):
    d = x.data.astype(np.float64)
    rstd = 1 / np.sqrt((d**2).mean(-1, keepdims=True) + eps)
    y = d * rstd * (gamma.data if gamma is not None else 1)
    return [_mk(y, x.dtype), None, _mk(rstd, FP32)]


def rms_norm_backward(grad, x, rstd, gamma, want_input=True, want_gamma=True):
    g, d = grad.data.astype(np.float64), x.data.astype(np.float64)
    gg = g * (gamma.data if gamma is not None else 1)
    xh = d * rstd.data
    dx = rstd.data * (gg - xh * (gg * xh).mean(-1, keepdims=True))
    lead = tuple(range(d.ndim - 1))
    return [
        _mk(dx, x.dtype) if want_input else None,
        (
            _mk((g * xh).sum(lead).reshape(_pad2((d.shape[-1],))), x.dtype)
            if want_gamma
            else None
        ),
        None,
    ]


def _attn(q, k, v, mask, causal, scale):
    s = np.matmul(q, np.swapaxes(k, -1, -2)) * (
        scale if scale is not None else 1 / np.sqrt(q.shape[-1])
    )
    if mask is not None:
        s = s + mask
    if causal:
        s = s + np.triu(np.full(s.shape[-2:], -np.inf), 1)
    e = np.exp(s - s.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True), s


def sdpa(q, k, v, mask=None, causal=False, scale=None):
    p, _ = _attn(
        q.data.astype(np.float64),
        k.data.astype(np.float64),
        v.data.astype(np.float64),
        None if mask is None else mask.data,
        causal,
        scale,
    )
    return _mk(np.matmul(p, v.data.astype(np.float64)), q.dtype)


def sdpa_train(q, k, v, causal=False):
    qd, kd, vd = (t.data.astype(np.float64) for t in (q, k, v))
    p, s = _attn(qd, kd, vd, None, causal, None)
    lse = np.log(np.exp(s - s.max(-1, keepdims=True)).sum(-1)) + s.max(-1)
    return [
        _mk(np.matmul(p, vd), q.dtype),
        _mk(np.broadcast_to(lse[..., None], lse.shape + (32,)).copy(), FP32),
    ]


def sdpa_train_backward(grad, out, q, k, v, stats, causal=False):
    qd, kd, vd, g = (t.data.astype(np.float64) for t in (q, k, v, grad))
    p, _ = _attn(qd, kd, vd, None, causal, None)
    scale = 1 / np.sqrt(qd.shape[-1])
    dv = np.matmul(np.swapaxes(p, -1, -2), g)
    dp = np.matmul(g, np.swapaxes(vd, -1, -2))
    ds = p * (dp - (dp * p).sum(-1, keepdims=True)) * scale
    return [
        _mk(np.matmul(ds, kd), q.dtype),
        _mk(np.matmul(np.swapaxes(ds, -1, -2), qd), k.dtype),
        _mk(dv, v.dtype),
    ]


def _conv_patches(x, w, stride, padding, dilation):
    n, c, h, wd = x.shape
    o, _, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding[0],) * 2, (padding[1],) * 2))
    oh = (h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    ow = (wd + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    cols = np.zeros((n, c, kh, kw, oh, ow))
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j] = xp[
                :,
                :,
                i * dilation[0] : i * dilation[0] + stride[0] * oh : stride[0],
                j * dilation[1] : j * dilation[1] + stride[1] * ow : stride[1],
            ]
    return cols, oh, ow


def conv2d(x, w, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    cols, oh, ow = _conv_patches(
        x.data.astype(np.float64), w.data, stride, padding, dilation
    )
    return _mk(np.einsum("ncijhw,ocij->nohw", cols, w.data.astype(np.float64)), x.dtype)


def conv2d_input_grad(
    go, w, x_shape, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1
):
    n, c, h, wd = x_shape
    o, _, kh, kw = w.shape
    g, wt = go.data.astype(np.float64), w.data.astype(np.float64)
    dxp = np.zeros((n, c, h + 2 * padding[0], wd + 2 * padding[1]))
    oh, ow = g.shape[2:]
    for i in range(kh):
        for j in range(kw):
            dxp[
                :,
                :,
                i * dilation[0] : i * dilation[0] + stride[0] * oh : stride[0],
                j * dilation[1] : j * dilation[1] + stride[1] * ow : stride[1],
            ] += np.einsum("nohw,oc->nchw", g, wt[:, :, i, j])
    return _mk(
        dxp[:, :, padding[0] : padding[0] + h, padding[1] : padding[1] + wd], go.dtype
    )


def conv2d_weight_grad(
    go, x, w_shape, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1
):
    cols, oh, ow = _conv_patches(
        x.data.astype(np.float64), np.zeros(w_shape), stride, padding, dilation
    )
    return _mk(
        np.einsum("ncijhw,nohw->ocij", cols, go.data.astype(np.float64)), go.dtype
    )


def adam_step(
    p, g, m, v, beta1, beta2, scale, eps, step, decay=1.0, l2=0.0, negate=False
):
    pd, gd, md, vd = (t.data.astype(np.float64) for t in (p, g, m, v))
    gd = (-gd if negate else gd) + l2 * pd
    md = beta1 * md + (1 - beta1) * gd
    vd = beta2 * vd + (1 - beta2) * gd * gd
    return [
        _mk(decay * pd - step * md / (np.sqrt(vd) * scale + eps), p.dtype),
        _mk(md, m.dtype),
        _mk(vd, v.dtype),
    ]
