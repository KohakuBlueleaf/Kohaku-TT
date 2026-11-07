"""Golden cases for numeric kernels: seeded CPU inputs -> tt outputs (CPU tensors).
Goldens are written by make_goldens.py from a build whose numerics were checked as stats
against the CPU baseline; test_goldens.py requires the same bits from later builds."""

import torch
import torch.nn.functional as F


def _inputs(seed, *shapes, dtype=torch.bfloat16):
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(*s, generator=gen).to(dtype) for s in shapes]


def _grads(fn, inputs, grad_seed):
    """fn(*inputs) on tt with requires_grad inputs, its output and the input grads."""
    xs = [x.to("tt").requires_grad_() for x in inputs]
    out = fn(*xs)
    gen = torch.Generator().manual_seed(grad_seed)
    g = torch.randn(out.shape, generator=gen).to(out.dtype)
    out.backward(g.to("tt"))
    return {
        "out": out.detach().cpu(),
        **{f"grad{i}": x.grad.cpu() for i, x in enumerate(xs)},
    }


def sdpa_flash():
    q, k, v = _inputs(0, (2, 4, 96, 64), (2, 4, 96, 64), (2, 4, 96, 64))
    with torch.no_grad():
        out = F.scaled_dot_product_attention(
            q.to("tt"), k.to("tt"), v.to("tt"), is_causal=True
        )
    return {"out": out.cpu()}


def sdpa_train():
    q, k, v = _inputs(1, (2, 4, 128, 64), (2, 4, 128, 64), (2, 4, 128, 64))
    return _grads(
        lambda a, b, c: F.scaled_dot_product_attention(a, b, c, is_causal=True),
        [q, k, v],
        2,
    )


def sdpa_recompute():
    q, k, v, m = _inputs(
        3, (1, 2, 64, 32), (1, 2, 80, 32), (1, 2, 80, 32), (1, 1, 64, 80)
    )
    return _grads(
        lambda a, b, c: F.scaled_dot_product_attention(a, b, c, attn_mask=m.to("tt")),
        [q, k, v],
        4,
    )


def fft():
    (x,) = _inputs(5, (3, 1031), dtype=torch.float32)
    spec = torch.fft.rfft(x.to("tt"))
    return {
        "rfft": torch.view_as_real(spec.cpu()),
        "irfft": torch.fft.irfft(spec, n=1031).cpu(),
    }


def conv_patchify():
    x, w, b = _inputs(6, (2, 3, 64, 64), (32, 3, 16, 16), (32,))
    return _grads(lambda a, c, d: F.conv2d(a, c, d, stride=16), [x, w, b], 7)


def conv_taps():
    x, w = _inputs(8, (2, 8, 20, 20), (16, 8, 3, 3))
    return _grads(lambda a, c: F.conv2d(a, c, padding=1), [x, w], 9)


def layer_norm():
    x, w, b = _inputs(10, (4, 77, 256), (256,), (256,))
    return _grads(lambda a, c, d: F.layer_norm(a, (256,), c, d), [x, w, b], 11)


def cross_entropy():
    (logits,) = _inputs(12, (128, 1000), dtype=torch.float32)
    target = torch.randint(0, 1000, (128,), generator=torch.Generator().manual_seed(13))
    xt = logits.to("tt").requires_grad_()
    loss = F.cross_entropy(xt, target.to("tt"))
    loss.backward()
    return {"loss": loss.detach().cpu().reshape(1), "grad": xt.grad.cpu()}


def matmul_softmax_gelu():
    a, b = _inputs(14, (256, 384), (384, 512))
    out = F.gelu((a.to("tt") @ b.to("tt")).softmax(-1) * 64)
    return {"out": out.cpu()}


CASES = {
    "sdpa_flash": sdpa_flash,
    "sdpa_train": sdpa_train,
    "sdpa_recompute": sdpa_recompute,
    "fft": fft,
    "conv_patchify": conv_patchify,
    "conv_taps": conv_taps,
    "layer_norm": layer_norm,
    "cross_entropy": cross_entropy,
    "matmul_softmax_gelu": matmul_softmax_gelu,
}
