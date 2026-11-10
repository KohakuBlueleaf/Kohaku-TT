"""ctypes binding over the C ABI (ktt_* functions, contracts in ktt_capi.h) that
`_ktt_native.so` exports. It is the one bridge of the process: the torch backend and
this binding share it, and the device opens and closes through kohakutt.open / close.
"""

from __future__ import annotations

import ctypes as ct
from pathlib import Path

import numpy as np

ABI_VERSION = 1

BF16, FP32, INT32, BFP8, BFP4 = range(5)
NCHW, NHWC, FLAT = range(3)
ITEMSIZE = {BF16: 2, FP32: 4, INT32: 4}

UNARY = {
    name: i
    for i, name in enumerate(
        [
            "neg",
            "abs",
            "exp",
            "log",
            "sqrt",
            "rsqrt",
            "reciprocal",
            "tanh",
            "sigmoid",
            "relu",
            "gelu",
            "gelu_tanh",
            "silu",
            "sin",
            "cos",
            "erf",
            "acos",
            "asin",
            "asinh",
            "atan",
            "atanh",
            "acosh",
            "cosh",
            "sinh",
            "tan",
            "exp2",
            "expm1",
            "log2",
            "log10",
            "log1p",
            "erfc",
            "erfinv",
            "lgamma",
            "digamma",
            "sign",
            "floor",
            "ceil",
            "trunc",
            "frac",
            "hardsigmoid",
            "hardswish",
            "mish",
            "isnan",
            "isinf",
            "isfinite",
            "i0",
        ]
    )
}
BINARY = {
    name: i
    for i, name in enumerate(
        [
            "add",
            "sub",
            "mul",
            "div",
            "maximum",
            "minimum",
            "pow",
            "eq",
            "ne",
            "lt",
            "le",
            "gt",
            "ge",
            "bit_and",
            "bit_or",
            "bit_xor",
            "atan2",
            "shl",
            "shr",
            "gcd",
            "lcm",
        ]
    )
}
COUNTER_FIELDS = [
    "uploads",
    "downloads",
    "ops",
    "op_ns",
    "programs",
    "desc_hits",
    "alloc_ns",
    "args_ns",
    "desc_ns",
    "enqueue_ns",
    "program_entries",
    "kernel_compiles",
]


class CoreError(RuntimeError):
    """A bridge call failed; the message is the bridge's."""


_lib = None


def library_path() -> Path | None:
    """The torch extension, which carries the C ABI, loaded; None if not built."""
    from kohakutt.torch_backend import registration

    return registration.native_library()


def available() -> bool:
    return library_path() is not None


def _sig(lib, name, restype, *argtypes):
    fn = getattr(lib, name)
    fn.restype, fn.argtypes = restype, list(argtypes)
    return fn


def load():
    """The extension's C ABI (once per process); CoreError if it is not built."""
    global _lib
    if _lib is not None:
        return _lib
    path = library_path()
    if path is None:
        raise CoreError(
            "the native layer is not built "
            "(cmake -S src/kohakutt_native -B build/native; ninja -C build/native)"
        )
    # The extension is already loaded by torch: this is the same image.
    lib = ct.CDLL(str(path))
    h, i64p, vp = ct.c_void_p, ct.POINTER(ct.c_int64), ct.c_void_p
    ci, cl, cd = ct.c_int, ct.c_int64, ct.c_double
    hp = ct.POINTER(ct.c_void_p)
    for name, res, args in (
        ("ktt_abi_version", ci, ()),
        ("ktt_last_error", ct.c_char_p, ()),
        ("ktt_open", ci, (ct.POINTER(ci), ci, ci, ci)),
        ("ktt_close", ci, ()),
        ("ktt_is_open", ci, ()),
        ("ktt_synchronize", ci, ()),
        ("ktt_mark_exiting", ci, ()),
        ("ktt_mesh_size", ci, ()),
        ("ktt_counters", ci, (i64p,)),
        ("ktt_memory", ci, (i64p,)),
        ("ktt_clear_program_cache", ci, ()),
        ("ktt_numerics_push", ci, (ci, ci, ci, ci)),
        ("ktt_numerics_pop", ci, ()),
        ("ktt_release", None, (h,)),
        ("ktt_retain", h, (h,)),
        ("ktt_upload", h, (vp, i64p, ci, ci, ci)),
        ("ktt_download", ci, (h, vp, ci)),
        ("ktt_empty", h, (i64p, ci, ci, ci)),
        ("ktt_clone", h, (h,)),
        ("ktt_copy_into", ci, (h, h)),
        ("ktt_shape", ci, (h, i64p, ci)),
        ("ktt_dtype", ci, (h,)),
        ("ktt_format", ci, (h,)),
        ("ktt_reshape", h, (h, i64p, ci)),
        ("ktt_permute", h, (h, i64p, ci)),
        ("ktt_slice", h, (h, i64p, i64p, i64p, ci)),
        ("ktt_concat", h, (hp, ci, cl)),
        ("ktt_repeat", h, (h, i64p, ci)),
        ("ktt_typecast", h, (h, ci)),
        ("ktt_bitcast", h, (h, ci)),
        ("ktt_unary", h, (ci, h)),
        ("ktt_binary", h, (ci, h, h)),
        ("ktt_binary_scalar", h, (ci, h, cd)),
        ("ktt_where", h, (h, h, h)),
        ("ktt_clamp", h, (h, ci, cd, ci, cd)),
        ("ktt_full", h, (i64p, ci, cd, ci)),
        ("ktt_uniform", h, (h, cd, cd, ct.c_uint32)),
        ("ktt_normal", h, (i64p, ci, ci, ct.c_uint32)),
        ("ktt_philox_uniform", h, (i64p, ci, ci, ct.c_uint64, ct.c_uint32)),
        ("ktt_sum", h, (h, i64p, ci, i64p, ci)),
        ("ktt_amax", h, (h, i64p, ci, i64p, ci)),
        ("ktt_softmax", h, (h, cl)),
        ("ktt_argmax_last", h, (h, i64p, ci)),
        ("ktt_cumsum", h, (h, cl)),
        ("ktt_embedding_rows", h, (h, cl, h, cl)),
        ("ktt_gather", h, (h, cl, h)),
        ("ktt_mm", h, (h, h, ci, ci, ci, ci)),
        ("ktt_linear", h, (h, h, h, ci)),
        ("ktt_layer_norm", ci, (h, cl, cd, h, h, hp)),
        ("ktt_layer_norm_backward", ci, (h, h, h, h, cl, h, ci, ci, ci, hp)),
        ("ktt_rms_norm_train", ci, (h, cd, h, hp)),
        ("ktt_rms_norm_backward", ci, (h, h, h, h, ci, ci, hp)),
        ("ktt_rms_norm", h, (h, cd, h, h)),
        ("ktt_sdpa", h, (h, h, h, h, ci, ci, cd)),
        ("ktt_sdpa_train", ci, (h, h, h, ci, hp)),
        ("ktt_sdpa_train_backward", ci, (h, h, h, h, h, h, ci, hp)),
        ("ktt_conv2d", h, (h, h, i64p, i64p, i64p, cl)),
        ("ktt_conv2d_input_grad", h, (h, h, i64p, i64p, i64p, i64p, cl)),
        ("ktt_conv2d_weight_grad", h, (h, h, i64p, i64p, i64p, i64p, cl)),
        ("ktt_adam_step", ci, (h, h, h, h, ct.POINTER(cd), ci, hp)),
    ):
        _sig(lib, name, res, *args)
    if lib.ktt_abi_version() != ABI_VERSION:
        raise CoreError(f"{path}: ABI {lib.ktt_abi_version()}, expected {ABI_VERSION}")
    _lib = lib
    return lib


def _err(what: str):
    msg = (_lib.ktt_last_error() or b"").decode(errors="replace")
    raise CoreError(f"{what}: {msg}")


def _i64(values):
    arr = (ct.c_int64 * max(len(values), 1))(*[int(v) for v in values])
    return arr, len(values)


class Tensor:
    """A bridge handle (a device tensor); released when the object is collected."""

    __slots__ = ("_h", "_shape", "_dtype", "_ver", "__weakref__")

    def __init__(self, raw: int):
        self._h = ct.c_void_p(raw)
        self._shape = None
        self._dtype = None
        self._ver = 0

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h and h.value and _lib is not None:
            _lib.ktt_release(h)

    @property
    def raw(self) -> ct.c_void_p:
        return self._h

    @property
    def shape(self) -> tuple[int, ...]:
        """The device shape (a 1-D logical shape is stored as (1, n))."""
        if self._shape is None:
            buf = (ct.c_int64 * 16)()
            n = _lib.ktt_shape(self._h, buf, 16)
            if n < 0:
                _err("shape")
            self._shape = tuple(buf[:n])
        return self._shape

    @property
    def dtype(self) -> int:
        if self._dtype is None:
            self._dtype = _lib.ktt_dtype(self._h)
        return self._dtype

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    def __repr__(self):
        return f"<ktt.Tensor {self.shape} dtype={self.dtype}>"


def _wrap(raw, what: str) -> Tensor:
    if not raw:
        _err(what)
    return Tensor(raw)


def _handles(raws, n, what, rc):
    if rc != 0:
        _err(what)
    return [Tensor(r) if r else None for r in raws[:n]]


# ---- device


def open(ids=(0,), rows: int = 1, cols: int = 1):
    """Opens the mesh of chips `ids` (rows x cols) as the process's device, through
    kohakutt.mesh so torch and this binding agree on what is open."""
    from kohakutt.device import session

    load()
    try:
        session.mesh(list(ids), (rows, cols))
    except RuntimeError as error:
        raise CoreError(f"open: {error}") from error


def close():
    """Closes the live mesh (the one kohakutt.close closes)."""
    from kohakutt.device import session

    if _lib is not None and session._LIVE:
        session.close()
    elif _lib is not None and _lib.ktt_close() != 0:
        _err("close")


def is_open() -> bool:
    """True when a device is open in this process, whichever frontend opened it."""
    return bool(load().ktt_is_open())


def synchronize():
    if _lib.ktt_synchronize() != 0:
        _err("synchronize")


def counters() -> dict[str, int]:
    out = (ct.c_int64 * 12)()
    if _lib.ktt_counters(out) != 0:
        _err("counters")
    return dict(zip(COUNTER_FIELDS, out))


def memory() -> dict[str, int]:
    out = (ct.c_int64 * 2)()
    if _lib.ktt_memory(out) != 0:
        _err("memory")
    return {"allocated": out[0], "peak": out[1]}


def clear_program_cache():
    _lib.ktt_clear_program_cache()


class numerics:
    """Numerics scope: fidelity 0..3 (LoFi..HiFi4), SFPU approx, strict, attention."""

    def __init__(self, fidelity=1, approx=False, strict=False, attention=0):
        self.args = (fidelity, int(approx), int(strict), attention)

    def __enter__(self):
        if _lib.ktt_numerics_push(*self.args) != 0:
            _err("numerics_push")

    def __exit__(self, *exc):
        _lib.ktt_numerics_pop()


# ---- memory


def upload(array: np.ndarray, dtype: int = FP32, fmt: int = NCHW) -> Tensor:
    """Host row-major data (in `dtype`'s byte layout; bf16 as uint16) as a tensor."""
    array = np.ascontiguousarray(array)
    shape, n = _i64(array.shape)
    return _wrap(_lib.ktt_upload(array.ctypes.data, shape, n, dtype, fmt), "upload")


def download(t: Tensor, chip: int = 0) -> np.ndarray:
    """The tensor's values as a host array of its device shape (bf16 as uint16 bits)."""
    np_dt = {BF16: np.uint16, FP32: np.float32, INT32: np.int32}.get(t.dtype)
    if np_dt is None:
        raise CoreError(
            f"download of block-float dtype {t.dtype} is not supported; cast it first"
        )
    out = np.empty(t.shape, dtype=np_dt)
    if _lib.ktt_download(t.raw, out.ctypes.data, chip) != 0:
        _err("download")
    return out


def empty(shape, dtype: int = FP32, fmt: int = NCHW) -> Tensor:
    s, n = _i64(shape)
    return _wrap(_lib.ktt_empty(s, n, dtype, fmt), "empty")


def clone(t: Tensor) -> Tensor:
    return _wrap(_lib.ktt_clone(t.raw), "clone")


def copy_into(dst: Tensor, src: Tensor):
    if _lib.ktt_copy_into(dst.raw, src.raw) != 0:
        _err("copy_into")


def full(shape, value: float, dtype: int = FP32) -> Tensor:
    s, n = _i64(shape)
    return _wrap(_lib.ktt_full(s, n, float(value), dtype), "full")


# ---- random draws (equal seeds or keys give equal values)


def uniform(shape, dtype: int, low: float, high: float, seed: int) -> Tensor:
    """U(low, high) of `shape` from the device's per-core generators."""
    like = empty(shape, dtype)
    return _wrap(_lib.ktt_uniform(like.raw, low, high, seed), "uniform")


def normal(shape, dtype: int, seed: int) -> Tensor:
    s, n = _i64(shape)
    return _wrap(_lib.ktt_normal(s, n, dtype, seed), "normal")


def philox_uniform(shape, dtype: int, key: int, word: int = 0) -> Tensor:
    """U(0, 1) from Philox4x32-10: word `word` of the counter at each element index."""
    s, n = _i64(shape)
    return _wrap(_lib.ktt_philox_uniform(s, n, dtype, key, word), "philox_uniform")


# ---- views


def reshape(t: Tensor, sizes) -> Tensor:
    s, n = _i64(sizes)
    return _wrap(_lib.ktt_reshape(t.raw, s, n), "reshape")


def permute(t: Tensor, dims) -> Tensor:
    s, n = _i64(dims)
    return _wrap(_lib.ktt_permute(t.raw, s, n), "permute")


def slice(t: Tensor, begins, ends, steps=None) -> Tensor:
    steps = steps if steps is not None else [1] * len(begins)
    b, n = _i64(begins)
    e, _ = _i64(ends)
    s, _ = _i64(steps)
    return _wrap(_lib.ktt_slice(t.raw, b, e, s, n), "slice")


def concat(parts, dim: int) -> Tensor:
    arr = (ct.c_void_p * len(parts))(*[p.raw for p in parts])
    return _wrap(_lib.ktt_concat(arr, len(parts), dim), "concat")


def repeat(t: Tensor, repeats) -> Tensor:
    r, n = _i64(repeats)
    return _wrap(_lib.ktt_repeat(t.raw, r, n), "repeat")


def typecast(t: Tensor, dtype: int) -> Tensor:
    return _wrap(_lib.ktt_typecast(t.raw, dtype), "typecast")


# ---- elementwise


def unary(op: str, a: Tensor) -> Tensor:
    return _wrap(_lib.ktt_unary(UNARY[op], a.raw), f"unary {op}")


def binary(op: str, a: Tensor, b: Tensor) -> Tensor:
    return _wrap(_lib.ktt_binary(BINARY[op], a.raw, b.raw), f"binary {op}")


# ops with a scalar form in the bridge
SCALAR_FORMS = frozenset(
    {"add", "sub", "mul", "div", "pow", "maximum", "minimum"}
    | {"eq", "ne", "lt", "le", "gt", "ge"}
)


def binary_scalar(op: str, a: Tensor, s: float) -> Tensor:
    if op not in SCALAR_FORMS:
        raise CoreError(f"binary_scalar {op}: the bridge has no scalar form of this op")
    return _wrap(
        _lib.ktt_binary_scalar(BINARY[op], a.raw, float(s)), f"binary_scalar {op}"
    )


def where(cond: Tensor, a: Tensor, b: Tensor) -> Tensor:
    return _wrap(_lib.ktt_where(cond.raw, a.raw, b.raw), "where")


def clamp(a: Tensor, lo=None, hi=None) -> Tensor:
    return _wrap(
        _lib.ktt_clamp(a.raw, lo is not None, lo or 0.0, hi is not None, hi or 0.0),
        "clamp",
    )


# ---- reductions and index


def sum(a: Tensor, dims, out_shape) -> Tensor:
    d, nd = _i64(dims)
    o, no = _i64(out_shape)
    return _wrap(_lib.ktt_sum(a.raw, d, nd, o, no), "sum")


def amax(a: Tensor, dims, out_shape) -> Tensor:
    d, nd = _i64(dims)
    o, no = _i64(out_shape)
    return _wrap(_lib.ktt_amax(a.raw, d, nd, o, no), "amax")


def softmax(a: Tensor, dim: int) -> Tensor:
    return _wrap(_lib.ktt_softmax(a.raw, dim), "softmax")


def argmax_last(a: Tensor, out_shape) -> Tensor:
    o, no = _i64(out_shape)
    return _wrap(_lib.ktt_argmax_last(a.raw, o, no), "argmax_last")


def cumsum(a: Tensor, dim: int) -> Tensor:
    return _wrap(_lib.ktt_cumsum(a.raw, dim), "cumsum")


_versions = iter(range(1, 1 << 62))


def version(t: Tensor) -> int:
    """A number naming this tensor object's contents (never written in place)."""
    if not t._ver:
        t._ver = next(_versions)
    return t._ver


def embedding_rows(table: Tensor, version: int, index: Tensor, count: int) -> Tensor:
    return _wrap(
        _lib.ktt_embedding_rows(table.raw, version, index.raw, count), "embedding_rows"
    )


def gather(a: Tensor, dim: int, index: Tensor) -> Tensor:
    return _wrap(_lib.ktt_gather(a.raw, dim, index.raw), "gather")


# ---- matmul


def mm(
    a: Tensor,
    b: Tensor,
    transpose_a=False,
    transpose_b=False,
    out_dtype=None,
    fast=False,
) -> Tensor:
    od = -1 if out_dtype is None else out_dtype
    return _wrap(_lib.ktt_mm(a.raw, b.raw, transpose_a, transpose_b, od, fast), "mm")


def linear(
    x: Tensor, w: Tensor, bias: Tensor | None = None, kn: bool = False
) -> Tensor:
    return _wrap(
        _lib.ktt_linear(x.raw, w.raw, bias.raw if bias else None, kn), "linear"
    )


# ---- fused ops


def _raw(t):
    return t.raw if t is not None else None


def layer_norm(x, normalized, eps, gamma=None, beta=None):
    """(y, mean, rstd)."""
    out = (ct.c_void_p * 3)()
    return _handles(
        out,
        3,
        "layer_norm",
        _lib.ktt_layer_norm(x.raw, normalized, eps, _raw(gamma), _raw(beta), out),
    )


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
    """(dx, dgamma, dbeta); unrequested entries are None."""
    out = (ct.c_void_p * 3)()
    rc = _lib.ktt_layer_norm_backward(
        grad.raw,
        x.raw,
        mean.raw,
        rstd.raw,
        normalized,
        _raw(gamma),
        want_input,
        want_gamma,
        want_beta,
        out,
    )
    return _handles(out, 3, "layer_norm_backward", rc)


def rms_norm_train(x, eps, gamma=None):
    """(y, None, rstd)."""
    out = (ct.c_void_p * 3)()
    return _handles(
        out, 3, "rms_norm_train", _lib.ktt_rms_norm_train(x.raw, eps, _raw(gamma), out)
    )


def rms_norm_backward(grad, x, rstd, gamma, want_input=True, want_gamma=True):
    """(dx, dgamma, None)."""
    out = (ct.c_void_p * 3)()
    rc = _lib.ktt_rms_norm_backward(
        grad.raw, x.raw, rstd.raw, _raw(gamma), want_input, want_gamma, out
    )
    return _handles(out, 3, "rms_norm_backward", rc)


def rms_norm(x, eps, gamma=None, beta=None):
    return _wrap(_lib.ktt_rms_norm(x.raw, eps, _raw(gamma), _raw(beta)), "rms_norm")


def sdpa(q, k, v, mask=None, causal=False, scale=None):
    return _wrap(
        _lib.ktt_sdpa(
            q.raw, k.raw, v.raw, _raw(mask), causal, scale is not None, scale or 0.0
        ),
        "sdpa",
    )


def sdpa_train(q, k, v, causal=False):
    """(out, stats)."""
    out = (ct.c_void_p * 2)()
    return _handles(
        out, 2, "sdpa_train", _lib.ktt_sdpa_train(q.raw, k.raw, v.raw, causal, out)
    )


def sdpa_train_backward(grad, out, q, k, v, stats, causal=False):
    """(dq, dk, dv)."""
    res = (ct.c_void_p * 3)()
    rc = _lib.ktt_sdpa_train_backward(
        grad.raw, out.raw, q.raw, k.raw, v.raw, stats.raw, causal, res
    )
    return _handles(res, 3, "sdpa_train_backward", rc)


def _pair(v):
    return (ct.c_int64 * 2)(*v)


def conv2d(x, w, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    return _wrap(
        _lib.ktt_conv2d(
            x.raw, w.raw, _pair(stride), _pair(padding), _pair(dilation), groups
        ),
        "conv2d",
    )


def conv2d_input_grad(
    go, w, x_shape, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1
):
    xs, _ = _i64(x_shape)
    return _wrap(
        _lib.ktt_conv2d_input_grad(
            go.raw, w.raw, xs, _pair(stride), _pair(padding), _pair(dilation), groups
        ),
        "conv2d_input_grad",
    )


def conv2d_weight_grad(
    go, x, w_shape, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1
):
    ws, _ = _i64(w_shape)
    return _wrap(
        _lib.ktt_conv2d_weight_grad(
            go.raw, x.raw, ws, _pair(stride), _pair(padding), _pair(dilation), groups
        ),
        "conv2d_weight_grad",
    )


def adam_step(
    p, g, m, v, beta1, beta2, scale, eps, step, decay=1.0, l2=0.0, negate=False
):
    """(p', m', v'), or None when the inputs do not qualify for the fused program."""
    params = (ct.c_double * 7)(beta1, beta2, scale, eps, step, decay, l2)
    out = (ct.c_void_p * 3)()
    rc = _lib.ktt_adam_step(p.raw, g.raw, m.raw, v.raw, params, negate, out)
    if rc == -2:
        return None
    return _handles(out, 3, "adam_step", rc)
