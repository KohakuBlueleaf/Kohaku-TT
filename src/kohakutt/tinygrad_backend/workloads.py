"""Operator workloads (name, function of tensors, input shapes) and a runner.
`run_case` evaluates one on CPU and on the device: max relative error, kernel count
and library-call counts of the device run.
"""

from __future__ import annotations

import numpy as np

from kohakutt.tinygrad_backend import compat as C
from kohakutt.tinygrad_backend import device as D

Tensor = C.Tensor


def _layernorm(x):
    return x.layernorm()


def _rmsnorm(x, w):
    return x * (x.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * w


def _embedding(w):
    return w[Tensor([3, 1, 7], device=w.device)]


WORKLOADS = [
    ("linear+bias", lambda x, w, b: x @ w.T + b, [(64, 256), (512, 256), (512,)]),
    (
        "linear+bias+gelu",
        lambda x, w, b: (x @ w.T + b).gelu(),
        [(64, 256), (512, 256), (512,)],
    ),
    ("layernorm", _layernorm, [(64, 256)]),
    ("rmsnorm (manual)", _rmsnorm, [(64, 256), (256,)]),
    ("softmax", lambda x: x.softmax(-1), [(64, 256)]),
    ("elementwise gelu(x*y+z)", lambda x, y, z: (x * y + z).gelu(), [(64, 256)] * 3),
    (
        "sdpa",
        lambda q, k, v: q.scaled_dot_product_attention(k, v),
        [(1, 4, 128, 64)] * 3,
    ),
    (
        "sdpa causal",
        lambda q, k, v: q.scaled_dot_product_attention(k, v, is_causal=True),
        [(1, 4, 128, 64)] * 3,
    ),
    (
        "conv2d k3 p1",
        lambda x, w, b: x.conv2d(w, b, padding=1),
        [(2, 16, 32, 32), (32, 16, 3, 3), (32,)],
    ),
    (
        "conv2d k3 s2",
        lambda x, w: x.conv2d(w, stride=2),
        [(2, 16, 32, 32), (32, 16, 3, 3)],
    ),
    ("transpose", lambda x: x.T.contiguous(), [(64, 256)]),
    (
        "permute 4d + reshape",
        lambda x: x.permute(0, 2, 1, 3).reshape(4 * 64, 32 * 16).contiguous(),
        [(4, 32, 64, 16)],
    ),
    (
        "pad + slice",
        lambda x: x.pad(((1, 2), (3, 0)))[2:, 1:-1].contiguous(),
        [(64, 256)],
    ),
    ("cat", lambda x, y: x.cat(y, dim=1), [(64, 256), (64, 128)]),
    ("sum all 4096x4096", lambda x: x.sum(), [(4096, 4096)]),
    ("max pool 2x2", lambda x: x.max_pool2d(2), [(2, 16, 32, 32)]),
    ("cumsum", lambda x: x.cumsum(-1), [(8, 256)]),
    ("embedding-like gather", _embedding, [(16, 64)]),
    ("argmax", lambda x: x.argmax(-1).float(), [(64, 256)]),
]


def _nn_layernorm(x, w, b):
    ln = C.nn.LayerNorm(x.shape[-1])
    ln.weight, ln.bias = w, b
    return ln(x)


def _nn_rmsnorm(x, w):
    rn = C.nn.RMSNorm(x.shape[-1])
    rn.weight = w
    return rn(x)


def _block(x, wqkv, wo, w1, w2, g1, b1, g2, b2, heads=4):
    """Pre-norm transformer block (causal attention, GELU MLP) on x [B, T, D]."""
    B, T, D = x.shape
    ln1, ln2 = C.nn.LayerNorm(D), C.nn.LayerNorm(D)
    ln1.weight, ln1.bias, ln2.weight, ln2.bias = g1, b1, g2, b2
    qkv = (
        (ln1(x) @ (wqkv * 0.06).T)
        .reshape(B, T, 3, heads, D // heads)
        .permute(2, 0, 3, 1, 4)
    )
    a = (
        qkv[0]
        .scaled_dot_product_attention(qkv[1], qkv[2], is_causal=True)
        .transpose(1, 2)
        .reshape(B, T, D)
    )
    x = x + a @ (wo * 0.06).T
    return x + (ln2(x) @ (w1 * 0.06).T).gelu() @ (w2 * 0.03).T


BLOCK_SHAPES = [
    (1, 128, 256),
    (768, 256),
    (256, 256),
    (1024, 256),
    (256, 1024),
    (256,),
    (256,),
    (256,),
    (256,),
]

# Workloads the overrides change. Gradient cases compare d(sum(f(x)^2))/d(inputs).
OVERRIDE_WORKLOADS = [
    ("ov nn.LayerNorm", _nn_layernorm, [(64, 256), (256,), (256,)]),
    ("ov nn.RMSNorm", _nn_rmsnorm, [(64, 256), (256,)]),
    (
        "ov sdpa",
        lambda q, k, v: q.scaled_dot_product_attention(k, v),
        [(1, 4, 128, 64)] * 3,
    ),
    (
        "ov sdpa causal",
        lambda q, k, v: q.scaled_dot_product_attention(k, v, is_causal=True),
        [(1, 4, 128, 64)] * 3,
    ),
    (
        "ov conv2d k3 p1",
        lambda x, w, b: x.conv2d(w, b, padding=1),
        [(2, 16, 32, 32), (32, 16, 3, 3), (32,)],
    ),
    (
        "ov conv2d k3 s2",
        lambda x, w: x.conv2d(w, stride=2),
        [(2, 16, 32, 32), (32, 16, 3, 3)],
    ),
    ("ov transformer block fwd", _block, BLOCK_SHAPES),
]
GRAD_WORKLOADS = [
    ("grad nn.LayerNorm", _nn_layernorm, [(64, 256), (256,), (256,)]),
    ("grad nn.RMSNorm", _nn_rmsnorm, [(64, 256), (256,)]),
    (
        "grad sdpa causal",
        lambda q, k, v: q.scaled_dot_product_attention(k, v, is_causal=True),
        [(1, 4, 128, 64)] * 3,
    ),
    (
        "grad conv2d k3 p1",
        lambda x, w, b: x.conv2d(w, b, padding=1),
        [(2, 16, 32, 32), (32, 16, 3, 3), (32,)],
    ),
    ("grad transformer block", _block, BLOCK_SHAPES),
]


def grads(device, fn, arrays):
    ts = [Tensor(a, device=device) for a in arrays]
    for t in ts:
        t.requires_grad = True
    fn(*ts).square().sum().backward()
    Tensor.realize(*[t.grad for t in ts])
    return [t.grad.numpy() for t in ts]


def run_grad_case(fn, shapes, device="KTT"):
    """(per-input max relative gradient error, kernels, calls) of `device` vs CPU."""
    arrays = inputs(shapes)
    ref = grads("CPU", fn, arrays)
    D.reset_stats()
    got = grads(device, fn, arrays)
    errs = [
        float(np.max(np.abs(g - r)) / (np.max(np.abs(r)) + 1e-9))
        for g, r in zip(got, ref)
    ]
    return errs, sum(D.KERNELS.values()), dict(D.CALLS)


def inputs(shapes, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(s).astype(np.float32) for s in shapes]


def on(device, fn, arrays):
    return fn(*[Tensor(a, device=device).realize() for a in arrays])


REF_SCALE = {
    "sum all 4096x4096": 4096.0
}  # sqrt(N): the size of a sum of N standard normals


def run_case(fn, shapes, device="KTT", name=None):
    """(max abs error / max(|ref|, REF_SCALE[name]), kernels, calls) vs CPU."""
    arrays = inputs(shapes)
    ref = on("CPU", fn, arrays).numpy()
    D.reset_stats()
    out = on(device, fn, arrays).numpy()
    err = float(
        np.max(np.abs(out - ref))
        / (max(np.max(np.abs(ref)), REF_SCALE.get(name, 0.0)) + 1e-9)
    )
    return err, sum(D.KERNELS.values()), dict(D.CALLS)
