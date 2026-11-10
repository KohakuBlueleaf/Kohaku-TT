"""Library-op overrides: sdpa, layernorm, RMSNorm, conv2d become one fused bridge-op
kernel (`LibPlan`) each, plus one for gradients; max_pool2d becomes reshape + max.
Calls that do not fit take tinygrad's path; `enable(False)` stops all overrides.
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
    reshape + max, whose gradient has no strided view; other calls: tinygrad's."""
    k = _pair(kernel_size)
    s = k if stride is None else _pair(stride)
    ok = (
        _fits(self)
        and not return_indices
        and self.ndim >= 2
        and k is not None
        and s == k
        and dilation in (1, (1, 1))
        and padding in (0, (0, 0), (0, 0, 0, 0))
        and self.shape[-2] % k[0] == 0
        and self.shape[-1] % k[1] == 0
    )
    if not ok:
        return _orig["max_pool2d"](
            self, kernel_size, stride, dilation, padding, ceil_mode, return_indices
        )
    *lead, h, w = self.shape
    x = self.reshape(*lead, h // k[0], k[0], w // k[1], k[1])
    return x.max(axis=(-3, -1))


def _patch():
    global _patched
    if _patched:
        return
    nn = C.nn
    _orig["max_pool2d"], Tensor.max_pool2d = Tensor.max_pool2d, _max_pool2d
    _orig["sdpa"], Tensor.scaled_dot_product_attention = (
        Tensor.scaled_dot_product_attention,
        _sdpa,
    )
    _orig["layernorm"], Tensor.layernorm = Tensor.layernorm, _layernorm
    _orig["conv2d"], Tensor.conv2d = Tensor.conv2d, _conv2d
    _orig["rms_norm"], nn.RMSNorm._norm = nn.RMSNorm._norm, _rms_norm
    _patched = True
