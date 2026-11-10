"""Library-op overrides: sdpa, layernorm, RMSNorm, conv2d become one fused bridge-op
kernel (`LibPlan`) each, plus one for gradients, and so do max / avg pooling; rand
draws on the device. Other calls take tinygrad's path; `enable(False)` stops them all.
"""

from __future__ import annotations

import json
import os

import numpy as np

from kohakutt.tinygrad_backend import compat as C
from kohakutt.tinygrad_backend.compat import Tensor, UOp, dtypes

_enabled = False
_patched = False
_attn_mode = os.environ.get(
    "KTT_TINYGRAD_ATTN", "auto"
)  # "infer": forward-only kernel (no gradients)
_orig: dict = {}
FLOATS = (dtypes.float32, dtypes.bfloat16)


def enabled() -> bool:
    return _enabled


def enable(on: bool = True):
    """Switch all overrides (patches the Tensor / nn methods once)."""
    global _enabled
    _enabled = bool(on)
    if _enabled:
        _patch()


def _fits(*tensors: Tensor) -> bool:
    return (
        _enabled
        and all(
            isinstance(t.device, str) and t.device.split(":")[0] == C.DEVICE_NAME
            for t in tensors
        )
        and all(all(isinstance(d, int) for d in t.shape) for t in tensors)
    )


# ---- library kernels


class LibPlan:
    """A kernel naming one fused bridge op; `spec` is its JSON description."""

    def __init__(self, ast: UOp):
        self.ast = ast
        self.spec = json.loads(str(ast.src[2].arg)[4:])
        self.params = [p for p in C.kernel_params(ast.src[0]) if not p.is_var]
        self.kind = "lib:" + self.spec["op"]
        self.calls = [self.kind]
        self.n_out = self.spec["n_out"]

    # ---- bridge

    def run_bridge(self, be, bufs: dict, var_vals: dict):
        from kohakutt.tinygrad_backend import executor

        core = be.core
        shapes = self.spec["shapes"]
        ins = []
        for i, p in enumerate(self.params):
            if i < self.n_out:
                continue
            ins.append(self._input(be, bufs[p.slot], p.dtype, shapes[i], executor))
        fn = getattr(self, "_b_" + self.spec["op"])
        outs = fn(core, be, ins)
        for i, (p, h) in enumerate(zip(self.params[: self.n_out], outs)):
            if h is None:
                continue
            want = be.codes[p.dtype]
            if h.dtype != want:
                h = core.typecast(h, want)
            n_declared = int(np.prod(shapes[i], dtype=np.int64))
            if h.numel != n_declared:
                raise executor.Unsupported(
                    f"{self.kind}: output {i} has {h.numel} elements, "
                    f"{n_declared} declared"
                )
            be.set_canon(bufs[p.slot], h, p.dtype)

    def _draw(self, core, code):
        """The next device draw of output 0: U(0, 1) keyed by `next_key()`."""
        shape, key = _pad2(self.spec["shapes"][0]), next_key()
        if self.spec["philox"]:
            return core.philox_uniform(shape, code, key)
        return core.uniform(shape, code, 0.0, 1.0, (key & 0xFFFFFFFF) | 1)

    def _input(self, be, st, td, shape, executor):
        code = be.codes.get(td)
        if code is None:
            raise executor.Unsupported(f"dtype {td}")
        return be.buffer_view(st, td, code, shape)

    def _need_device(self, bufs_in, executor):
        pass

    def _bf16(self, core, be, tensors):
        """The tensors as bf16 when the op is bf16-only and the buffers fp32."""
        if not self.spec.get("cast"):
            return tensors
        return [
            (
                t
                if t is None or t.dtype == be.raw_core.BF16
                else core.typecast(t, be.raw_core.BF16)
            )
            for t in tensors
        ]

    def _b_rand(self, core, be, ins):
        return [self._draw(core, be.codes[self.params[0].dtype])]

    def _b_max_pool2d(self, core, be, ins):
        return [pool_forward(core, ins[0], self.spec)]

    _b_avg_pool2d = _b_max_pool2d

    def _b_max_pool2d_bw(self, core, be, ins):
        return [pool_backward(core, *ins, self.spec)]

    _b_avg_pool2d_bw = _b_max_pool2d_bw

    def _b_sdpa(self, core, be, ins):
        q, k, v = self._bf16(core, be, ins)
        return [core.sdpa(q, k, v, None, self.spec["causal"], None)]

    def _b_sdpa_train(self, core, be, ins):
        q, k, v = self._bf16(core, be, ins)
        out, stats = core.sdpa_train(q, k, v, self.spec["causal"])
        return [out, stats]

    def _b_sdpa_train_bw(self, core, be, ins):
        g, out, q, k, v = self._bf16(core, be, ins[:5])
        return list(
            core.sdpa_train_backward(g, out, q, k, v, ins[5], self.spec["causal"])
        )

    def _b_layer_norm(self, core, be, ins):
        (x,) = ins
        y, mean, rstd = core.layer_norm(x, 1, self.spec["eps"], None, None)
        return [y, mean, rstd]

    def _b_layer_norm_bw(self, core, be, ins):
        g, x, mean, rstd = ins
        dx, _, _ = core.layer_norm_backward(
            g, x, mean, rstd, 1, None, True, False, False
        )
        return [dx]

    def _b_rms_norm(self, core, be, ins):
        (x,) = ins
        y, _, rstd = core.rms_norm_train(x, self.spec["eps"], None)
        return [y, rstd]

    def _b_rms_norm_bw(self, core, be, ins):
        g, x, rstd = ins
        dx, _, _ = core.rms_norm_backward(g, x, rstd, None, True, False)
        return [dx]

    def _conv_args(self):
        s = self.spec
        return (
            tuple(s["stride"]),
            tuple(s["padding"]),
            tuple(s["dilation"]),
            s["groups"],
        )

    def _b_conv2d(self, core, be, ins):
        x, w = ins
        return [core.conv2d(x, w, *self._conv_args())]

    def _b_conv2d_bw(self, core, be, ins):
        g, x, w = self._bf16(core, be, ins)
        stride, pad, dil, groups = self._conv_args()
        if self.spec["need_dx"]:
            return [
                core.conv2d_input_grad(g, w, list(x.shape), stride, pad, dil, groups)
            ]
        return [core.conv2d_weight_grad(g, x, list(w.shape), stride, pad, dil, groups)]

    # ---- host (reference formulas, used when the device path is unavailable)

    def run_host(self, be, bufs: dict, var_vals: dict):
        from kohakutt.tinygrad_backend import core_sim as S

        if self.spec["op"] == "rand":
            p = self.params[0]
            bf16 = p.dtype == dtypes.bfloat16
            bits = S.download(self._draw(S, S.BF16 if bf16 else S.FP32))
            host = be.host_bytes(bufs[p.slot])
            host.view(np.uint16 if bf16 else np.float32)[:] = bits.reshape(-1)
            be.touch(bufs[p.slot])
            return
        shapes = self.spec["shapes"]
        tensors = []
        for i, p in enumerate(self.params):
            if i < self.n_out:
                tensors.append(None)
                continue
            raw = (
                be.host_bytes(bufs[p.slot]).view(C.np_dtype(p.dtype)).reshape(shapes[i])
            )
            tensors.append(S.upload(raw.astype(np.float32), S.FP32))
        ins = tensors[self.n_out :]
        op = self.spec["op"]
        if op == "sdpa":
            outs = [S.sdpa(*ins, None, self.spec["causal"])]
        elif op == "sdpa_train":
            outs = S.sdpa_train(*ins, self.spec["causal"])
        elif op == "sdpa_train_bw":
            g, out, q, k, v, stats = ins
            outs = S.sdpa_train_backward(g, out, q, k, v, stats, self.spec["causal"])
        elif op == "layer_norm":
            outs = S.layer_norm(ins[0], 1, self.spec["eps"])
        elif op == "layer_norm_bw":
            g, x, mean, rstd = ins
            outs = [
                S.layer_norm_backward(g, x, mean, rstd, 1, None, True, False, False)[0]
            ]
        elif op == "rms_norm":
            y, _, rstd = S.rms_norm_train(ins[0], self.spec["eps"])
            outs = [y, rstd]
        elif op == "rms_norm_bw":
            g, x, rstd = ins
            outs = [S.rms_norm_backward(g, x, rstd, None, True, False)[0]]
        elif op in ("max_pool2d", "avg_pool2d"):
            outs = [pool_forward(S, ins[0], self.spec)]
        elif op in ("max_pool2d_bw", "avg_pool2d_bw"):
            outs = [pool_backward(S, *ins, self.spec)]
        elif op == "conv2d":
            outs = [S.conv2d(ins[0], ins[1], *self._conv_args()[:3])]
        elif op == "conv2d_bw":
            g, x, w = ins
            stride, pad, dil, groups = self._conv_args()
            outs = [
                (
                    S.conv2d_input_grad(g, w, list(x.shape), stride, pad, dil)
                    if self.spec["need_dx"]
                    else S.conv2d_weight_grad(g, x, list(w.shape), stride, pad, dil)
                )
            ]
        else:
            raise NotImplementedError(op)
        for p, o in zip(self.params[: self.n_out], outs):
            if o is None:
                continue
            host = be.host_bytes(bufs[p.slot])
            host.view(C.np_dtype(p.dtype))[:] = (
                S.download(o).reshape(-1).astype(C.np_dtype(p.dtype))
            )
            be.touch(bufs[p.slot])


def make_lib_plan(ast: UOp) -> LibPlan:
    return LibPlan(ast)


def lib_call(
    spec: dict, outs: list[Tensor], ins: list[Tensor], grad_fxn=None
) -> list[Tensor]:
    """One custom kernel over buffers `outs` (written) and `ins`; returns them all."""
    ins = [t.contiguous() for t in ins]
    spec = {
        **spec,
        "n_out": len(outs),
        "shapes": [list(t.shape) for t in (*outs, *ins)],
    }
    text = "lib " + json.dumps(spec, sort_keys=True)

    def body(*params):
        return C.source_program(C.kernel_sink("ktt_" + spec["op"], *params), text)

    return outs[0].custom_kernel(*outs[1:], *ins, fxn=body, grad_fxn=grad_fxn)


def _empty(like: Tensor, shape, dtype=None) -> Tensor:
    return Tensor.empty(*shape, dtype=dtype or like.dtype, device=like.device)


def _after(kernel: UOp, i: int) -> Tensor:
    return Tensor(kernel.src[1 + i].after(kernel))


# ---- scaled_dot_product_attention


def _sdpa(
    self, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, enable_gqa=False
):
    orig = _orig["sdpa"]
    q = self
    ok = (
        _fits(q, key, value)
        and attn_mask is None
        and not dropout_p
        and not enable_gqa
        and q.ndim == key.ndim == value.ndim == 4
        and q.dtype == key.dtype == value.dtype
        and q.dtype in FLOATS
        and q.shape[-1] == key.shape[-1]
        and key.shape[:3] == value.shape[:3]
        and q.shape[:2] == key.shape[:2]
        and (not is_causal or q.shape[2] == key.shape[2])
    )
    if not ok:
        return orig(self, key, value, attn_mask, dropout_p, is_causal, enable_gqa)
    out_shape = (*q.shape[:3], value.shape[-1])
    cast = q.dtype != dtypes.bfloat16  # the fused attention kernels take bf16 only
    causal = bool(is_causal)
    if _attn_mode == "infer":
        return lib_call(
            {"op": "sdpa", "causal": causal, "cast": cast},
            [_empty(q, out_shape)],
            [q, key, value],
        )[0]
    # No requires_grad in tinygrad 0.14: the training kernel (with row statistics)
    # is used whenever it fits.
    if (
        q.shape[2] % 32
        or key.shape[2] % 32
        or q.shape[-1] != value.shape[-1]
        or q.shape[2] != key.shape[2]
    ):
        return orig(self, key, value, attn_mask, dropout_p, is_causal, enable_gqa)

    def grad(gradient: UOp, kernel: UOp):
        o_t, stats_t, q_t, k_t, v_t = (_after(kernel, i) for i in range(5))
        dq, dk, dv = (_empty(q_t, t.shape) for t in (q_t, k_t, v_t))
        res = lib_call(
            {"op": "sdpa_train_bw", "causal": causal, "cast": cast},
            [dq, dk, dv],
            [Tensor(gradient), o_t, q_t, k_t, v_t, stats_t],
        )
        return (None, None, res[0].uop, res[1].uop, res[2].uop)

    out, stats = _empty(q, out_shape), _empty(q, (*q.shape[:3], 32), dtypes.float32)
    return lib_call(
        {"op": "sdpa_train", "causal": causal, "cast": cast},
        [out, stats],
        [q, key, value],
        grad,
    )[0]


# ---- layernorm / RMSNorm


def _stats_shape(x: Tensor):
    return (*x.shape[:-1], 1)


def _layernorm(self, axis=-1, eps=1e-5):
    orig = _orig["layernorm"]
    last = self.ndim - 1
    ok = (
        _fits(self)
        and self.dtype in FLOATS
        and self.ndim >= 2
        and axis in (-1, last, (-1,), (last,))
    )
    if not ok:
        return orig(self, axis, eps)
    x = self
    y, mean, rstd = (
        _empty(x, x.shape),
        _empty(x, _stats_shape(x), dtypes.float32),
        _empty(x, _stats_shape(x), dtypes.float32),
    )

    def grad(gradient: UOp, kernel: UOp):
        mean_t, rstd_t, x_t = _after(kernel, 1), _after(kernel, 2), _after(kernel, 3)
        dx = _empty(x_t, x_t.shape)
        res = lib_call(
            {"op": "layer_norm_bw"}, [dx], [Tensor(gradient), x_t, mean_t, rstd_t]
        )
        return (None, None, None, res[0].uop)

    return lib_call(
        {"op": "layer_norm", "eps": float(eps)}, [y, mean, rstd], [x], grad
    )[0]


def _rms_norm(self_mod, x: Tensor):
    orig = _orig["rms_norm"]
    ok = _fits(x) and x.dtype in FLOATS and x.ndim >= 2
    if not ok:
        return orig(self_mod, x)
    eps = float(self_mod.eps)
    y, rstd = _empty(x, x.shape), _empty(x, _stats_shape(x), dtypes.float32)

    def grad(gradient: UOp, kernel: UOp):
        rstd_t, x_t = _after(kernel, 1), _after(kernel, 2)
        dx = _empty(x_t, x_t.shape)
        res = lib_call({"op": "rms_norm_bw"}, [dx], [Tensor(gradient), x_t, rstd_t])
        return (None, None, res[0].uop)

    return lib_call({"op": "rms_norm", "eps": eps}, [y, rstd], [x], grad)[0]


# ---- conv2d


def _pair(v):
    if isinstance(v, int):
        return (v, v)
    v = tuple(v)
    return v if len(v) == 2 and all(isinstance(i, int) for i in v) else None


def _conv2d(
    self, weight, bias=None, groups=1, stride=1, dilation=1, padding=0, dtype=None
):
    orig = _orig["conv2d"]
    st, di, pd = _pair(stride), _pair(dilation), _pair(padding)
    ok = (
        _fits(self, weight)
        and dtype is None
        and self.ndim == 4
        and weight.ndim == 4
        and self.dtype == weight.dtype
        and self.dtype in FLOATS
        and None not in (st, di, pd)
    )
    if not ok or weight.shape[1] * groups != self.shape[1]:
        return orig(self, weight, bias, groups, stride, dilation, padding, dtype)
    n, _, h, w = self.shape
    o, _, kh, kw = weight.shape
    oh = (h + 2 * pd[0] - di[0] * (kh - 1) - 1) // st[0] + 1
    ow = (w + 2 * pd[1] - di[1] * (kw - 1) - 1) // st[1] + 1
    spec = {
        "stride": list(st),
        "padding": list(pd),
        "dilation": list(di),
        "groups": int(groups),
        "cast": self.dtype != dtypes.bfloat16,
    }  # cast: the backward kernels take bf16

    def grad(gradient: UOp, kernel: UOp):
        # one kernel per gradient: the scheduler runs only those the caller asks for
        x_t, w_t = _after(kernel, 1), _after(kernel, 2)
        dx = lib_call(
            {"op": "conv2d_bw", **spec, "need_dx": True, "need_dw": False},
            [_empty(x_t, x_t.shape)],
            [Tensor(gradient), x_t, w_t],
        )
        dw = lib_call(
            {"op": "conv2d_bw", **spec, "need_dx": False, "need_dw": True},
            [_empty(w_t, w_t.shape)],
            [Tensor(gradient), x_t, w_t],
        )
        return (None, dx[0].uop, dw[0].uop)

    out = lib_call(
        {"op": "conv2d", **spec}, [_empty(self, (n, o, oh, ow))], [self, weight], grad
    )[0]
    return out if bias is None else out + bias.reshape(1, -1, 1, 1)


# ---- max_pool2d / avg_pool2d


def _max_pool2d(
    self,
    kernel_size=(2, 2),
    stride=None,
    dilation=1,
    padding=0,
    ceil_mode=False,
    return_indices=False,
):
    """Non-overlapping pooling (stride == kernel, no padding, dividing sizes) as
    reshape + max; other 2-D float pooling as the `max_pool2d` bridge kernel, whose
    gradient adds each window's share back by scatter. Else tinygrad's."""
    orig = _orig["max_pool2d"]
    args = (kernel_size, stride, dilation, padding, ceil_mode, return_indices)
    if not _fits(self) or return_indices:
        return orig(self, *args)
    g = _pool_geometry(self, kernel_size, stride, dilation, padding, ceil_mode)
    if g is None:
        return orig(self, *args)
    if _pool_tiles(self, g):
        return _pool_blocks(self, g).max(axis=(-3, -1))
    if self.dtype not in FLOATS:
        return orig(self, *args)
    spec = {"op": "max_pool2d", **g, "low": float(self.dtype.min)}
    return _pool_call(self, spec)


def _avg_pool2d(
    self,
    kernel_size=(2, 2),
    stride=None,
    dilation=1,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
):
    """Non-overlapping pooling as reshape + mean; other 2-D float pooling as the
    `avg_pool2d` bridge kernel (gradient: scatter of the shares). Else tinygrad's."""
    orig = _orig["avg_pool2d"]
    args = (kernel_size, stride, dilation, padding, ceil_mode, count_include_pad)
    g = _pool_geometry(self, kernel_size, stride, dilation, padding, ceil_mode)
    if not _fits(self) or self.dtype not in FLOATS or g is None:
        return orig(self, *args)
    if _pool_tiles(self, g):
        return _pool_blocks(self, g).mean(axis=(-3, -1))
    spec = {"op": "avg_pool2d", **g, "include_pad": bool(count_include_pad)}
    return _pool_call(self, spec)


def _pool_geometry(x, kernel_size, stride, dilation, padding, ceil_mode):
    """kernel, stride, dilation, pads (top, bottom, left, right; "reg": before
    ceil_mode) and output size of a 2-D pool over x's last two dims, or None."""
    k, d = _pair(kernel_size), _pair(dilation)
    s = k if stride is None else _pair(stride)
    if None in (k, s, d) or x.ndim < 2 or min(*k, *s, *d) < 1:
        return None
    try:
        reg = list(C.resolve_pool_pads(padding, 2))
    except (TypeError, ValueError):
        return None
    pads = list(x._apply_ceil_mode(reg, k, s, d)) if ceil_mode else reg
    if not all(isinstance(p, int) and p >= 0 for p in pads + reg):
        return None
    (pt, pb), (pl, pr) = C.flat_to_grouped(pads)
    (rt, rb), (rl, rr) = C.flat_to_grouped(reg)
    h, w = x.shape[-2:]
    oh = (h + pt + pb - d[0] * (k[0] - 1) - 1) // s[0] + 1
    ow = (w + pl + pr - d[1] * (k[1] - 1) - 1) // s[1] + 1
    if oh < 1 or ow < 1 or x.numel() == 0:
        return None
    return {
        "kernel": list(k),
        "stride": list(s),
        "dilation": list(d),
        "pads": [pt, pb, pl, pr],
        "reg": [rt, rb, rl, rr],
        "out": [oh, ow],
    }


def _pool_tiles(x, g) -> bool:
    """True when the windows tile the input: stride = kernel, no padding or dilation."""
    k = g["kernel"]
    return (
        g["stride"] == k
        and g["dilation"] == [1, 1]
        and not any(g["pads"])
        and x.shape[-2] % k[0] == 0
        and x.shape[-1] % k[1] == 0
    )


def _pool_blocks(x, g):
    (kh, kw), (h, w) = g["kernel"], x.shape[-2:]
    return x.reshape(*x.shape[:-2], h // kh, kh, w // kw, kw)


def _pool_call(x, spec):
    """One pooling kernel over x; its gradient is the matching `*_bw` kernel."""

    def grad(gradient: UOp, kernel: UOp):
        out_t, x_t = _after(kernel, 0), _after(kernel, 1)
        dx = lib_call(
            {**spec, "op": spec["op"] + "_bw"},
            [_empty(x_t, x_t.shape)],
            [Tensor(gradient), x_t, out_t],
        )
        return (None, dx[0].uop)

    out_shape = (*x.shape[:-2], *spec["out"])
    return lib_call(spec, [_empty(x, out_shape)], [x], grad)[0]


# ---- pooling kernels on a core binding (the bridge or its numpy model)


def _planes(core, t, hw):
    return core.reshape(t, (t.numel // (hw[0] * hw[1]), *hw))


def _padded(core, x, pads, value):
    """x [n, h, w] padded by (top, bottom, left, right) with `value`."""
    pt, pb, pl, pr = pads
    n, h, w = x.shape

    def band(shape):
        return core.full(shape, value, x.dtype)

    if pl or pr:
        left = [band((n, h, pl))] if pl else []
        x = core.concat([*left, x, *([band((n, h, pr))] if pr else [])], 2)
    if pt or pb:
        top = [band((n, pt, w + pl + pr))] if pt else []
        x = core.concat([*top, x, *([band((n, pb, w + pl + pr))] if pb else [])], 1)
    return x


def _taps(core, xp, spec):
    """Per kernel tap (row-major), the [n, oh, ow] input values it reads."""
    (kh, kw), (sh, sw) = spec["kernel"], spec["stride"]
    (dh, dw), (oh, ow) = spec["dilation"], spec["out"]
    n = xp.shape[0]
    return [
        core.slice(
            xp,
            [0, i * dh, j * dw],
            [n, i * dh + (oh - 1) * sh + 1, j * dw + (ow - 1) * sw + 1],
            [1, sh, sw],
        )
        for i in range(kh)
        for j in range(kw)
    ]


def _tap_index(spec, wp):
    """[1, taps * oh * ow] int32: the padded flat position each tap reads per output."""
    (kh, kw), (sh, sw) = spec["kernel"], spec["stride"]
    (dh, dw), (oh, ow) = spec["dilation"], spec["out"]
    rows, cols = np.arange(oh)[:, None] * sh, np.arange(ow)[None, :] * sw
    flat = [
        ((rows + i * dh) * wp + cols + j * dw).reshape(-1)
        for i in range(kh)
        for j in range(kw)
    ]
    return np.concatenate(flat).astype(np.int32)[None]


def _fold(core, shares, spec, x):
    """[n, h, w]: per input element, the sum of the tap `shares` [n, oh, ow] read
    from it (scatter-add over the padded input, padding cut off)."""
    n, h, w = x.shape
    pt, pb, pl, pr = spec["pads"]
    hp, wp = h + pt + pb, w + pl + pr
    oh, ow = spec["out"]
    index = core.upload(_tap_index(spec, wp), core.INT32)
    if n > 1:
        index = core.repeat(index, [n, 1])
    src = core.concat([core.reshape(s, (n, oh * ow)) for s in shares], 1)
    acc = core.full((n, hp * wp), 0.0, shares[0].dtype)
    acc = core.reshape(core.scatter(acc, 1, index, src, "add"), (n, hp, wp))
    if hp == h and wp == w:
        return acc
    return core.slice(acc, [0, pt, pl], [n, pt + h, pl + w])


def _avg_divisor(spec, h, w):
    """[oh, ow] float32: the cells each window averages over (tinygrad's rule)."""
    (kh, kw), (sh, sw) = spec["kernel"], spec["stride"]
    (dh, dw), (oh, ow) = spec["dilation"], spec["out"]
    pt, _, pl, _ = spec["pads"]
    rt, rb, rl, rr = spec["reg"]
    if spec["include_pad"]:
        rows, cols = (0, rt + h + rb), (0, rl + w + rr)
    else:
        rows, cols = (pt, pt + h), (pl, pl + w)

    def inside(n, s, k, d, span):
        at = np.arange(n)[:, None] * s + np.arange(k)[None, :] * d
        return ((at >= span[0]) & (at < span[1])).sum(1)

    count = np.outer(inside(oh, sh, kh, dh, rows), inside(ow, sw, kw, dw, cols))
    return count.astype(np.float32)


def pool_forward(core, x, spec):
    """Max or average pool of x [..., h, w] per `spec`, as [planes, oh, ow]."""
    hw = spec["shapes"][1][-2:]
    x = _planes(core, x, hw)
    low = spec["low"] if spec["op"] == "max_pool2d" else 0.0
    taps = _taps(core, _padded(core, x, spec["pads"], low), spec)
    out = taps[0]
    for t in taps[1:]:
        out = core.binary("maximum" if spec["op"] == "max_pool2d" else "add", out, t)
    if spec["op"] == "max_pool2d":
        return out
    return _divide(core, out, spec, *hw)


def _divide(core, t, spec, h, w):
    div = _avg_divisor(spec, h, w)
    if (div == div.flat[0]).all():
        return core.binary_scalar("div", t, float(div.flat[0]))
    d = core.upload(div[None], core.FP32)
    if t.dtype != core.FP32:
        d = core.typecast(d, t.dtype)
    if t.shape[0] > 1:
        d = core.repeat(d, [t.shape[0], 1, 1])
    return core.binary("div", t, d)


def pool_backward(core, grad, x, out, spec):
    """The input gradient [planes, h, w] of pool_forward given its output gradient."""
    hw = spec["shapes"][2][-2:]
    x = _planes(core, x, hw)
    grad = _planes(core, grad, spec["out"])
    if spec["op"] == "avg_pool2d_bw":
        share = _divide(core, grad, spec, *hw)
        return _fold(core, [share] * int(np.prod(spec["kernel"])), spec, x)
    out = _planes(core, out, spec["out"])
    taps = _taps(core, _padded(core, x, spec["pads"], spec["low"]), spec)
    hits = [core.typecast(core.binary("eq", t, out), x.dtype) for t in taps]
    count = hits[0]
    for m in hits[1:]:
        count = core.binary("add", count, m)
    shares = [core.binary("mul", core.binary("div", m, count), grad) for m in hits]
    return _fold(core, shares, spec, x)


# ---- random numbers

M64 = (1 << 64) - 1
_draws = {"seed": None, "n": 0}  # the tinygrad seed counted from, device draws since


def rand_route() -> str:
    """KTT_TINYGRAD_RAND: "card" (default) draws on the device; "host" runs tinygrad's
    own generator on the host, which gives the CPU backend's stream."""
    route = os.environ.get("KTT_TINYGRAD_RAND", "card")
    if route not in ("card", "host"):
        raise ValueError(f"KTT_TINYGRAD_RAND={route!r}: use 'card' or 'host'")
    return route


def _mix(z: int) -> int:
    """The splitmix64 output function of state `z`."""
    z = (z + 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def next_key() -> int:
    """The 64-bit key of the next device draw: a hash of `Tensor.manual_seed`'s seed
    and the number of draws run since it was set."""
    seed = Tensor._seed
    if _draws["seed"] != seed:
        _draws.update(seed=seed, n=0)
    n = _draws["n"]
    _draws["n"] = n + 1
    return _mix(_mix(seed & M64) ^ n)


def _manual_seed(seed=0):
    _orig["manual_seed"](seed)
    _draws.update(seed=None, n=0)


def _pad2(shape):
    shape = tuple(int(s) for s in shape)
    return (1,) * (2 - len(shape)) + shape if len(shape) < 2 else shape


def _card_draw(device, dtype, dims) -> bool:
    if not (_enabled and rand_route() == "card"):
        return False
    if not all(isinstance(s, int) and s > 0 for s in dims):
        return False
    if device is not None and not isinstance(device, str):
        return False
    on_card = C.canonicalize_device(device).split(":")[0] == C.DEVICE_NAME
    return on_card and C.to_dtype(dtype or dtypes.default_float) in FLOATS


def _rand(cls, *shape, device=None, dtype=None, contiguous=True):
    """U(0, 1) drawn on the device for KTT float32 / bfloat16 tensors."""
    dims = C.argfix(*shape)
    if not _card_draw(device, dtype, dims):
        return _orig["rand"](*shape, device=device, dtype=dtype, contiguous=contiguous)
    out = Tensor.empty(
        *dims,
        dtype=C.to_dtype(dtype or dtypes.default_float),
        device=C.canonicalize_device(device),
    )
    spec = {"op": "rand", "philox": os.environ.get("KTT_PHILOX") == "1"}
    return lib_call(spec, [out], [])[0]


def _dropout(self, p=0.5):
    """tinygrad's dropout with the mask fused into the select (no stored bool mask)."""
    if not (
        _fits(self)
        and isinstance(p, (int, float))
        and 0 < p < 1
        and C.training()
        and _card_draw(self.device, dtypes.default_float, self.shape)
    ):
        return _orig["dropout"](self, p)
    keep = self.rand_like(dtype=dtypes.default_float) >= p
    return keep.where(self, 0) / (1.0 - p)


def _patch():
    global _patched
    if _patched:
        return
    nn = C.nn
    _orig["rand"], Tensor.rand = Tensor.rand, classmethod(_rand)
    _orig["dropout"], Tensor.dropout = Tensor.dropout, _dropout
    _orig["manual_seed"] = Tensor.manual_seed
    Tensor.manual_seed = staticmethod(_manual_seed)
    _orig["max_pool2d"], Tensor.max_pool2d = Tensor.max_pool2d, _max_pool2d
    _orig["avg_pool2d"], Tensor.avg_pool2d = Tensor.avg_pool2d, _avg_pool2d
    _orig["sdpa"], Tensor.scaled_dot_product_attention = (
        Tensor.scaled_dot_product_attention,
        _sdpa,
    )
    _orig["layernorm"], Tensor.layernorm = Tensor.layernorm, _layernorm
    _orig["conv2d"], Tensor.conv2d = Tensor.conv2d, _conv2d
    _orig["rms_norm"], nn.RMSNorm._norm = nn.RMSNorm._norm, _rms_norm
    _patched = True
