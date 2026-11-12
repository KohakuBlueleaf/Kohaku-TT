"""Attention with per-batch key lengths on tt: the kernel route (ktt::_sdpa_lengths and
its backward) and the explicit-mask route for shapes the kernel does not take."""

import torch
import torch.nn.functional as F

from kohakutt.core.numerics import current_numerics


def device_lengths(key_lengths, batch, keys, device):
    """`key_lengths` as an int32 [batch] tensor on `device`. A list or a CPU tensor is
    checked (1 <= length <= keys) and uploaded; a tt tensor is used as it is (no
    download: the kernel clamps its values to [1, keys])."""
    if isinstance(key_lengths, torch.Tensor) and key_lengths.device.type != "cpu":
        lengths = key_lengths.to(torch.int32)
    else:
        lengths = torch.as_tensor(key_lengths, dtype=torch.int32)
        if lengths.numel() and (lengths.min() < 1 or lengths.max() > keys):
            raise ValueError(
                f"kohakutt.sdpa: key_lengths must lie in [1, {keys}], "
                f"got {lengths.tolist()}"
            )
        lengths = lengths.to(device)
    if lengths.dim() != 1 or lengths.size(0) != batch:
        raise ValueError(
            f"kohakutt.sdpa: key_lengths must hold one length per batch ({batch}), "
            f"got shape {tuple(lengths.shape)}"
        )
    return lengths


def _device_fits(q, k, v, is_causal, grads):
    return torch.ops.ktt._sdpa_lengths_supported(q, k, v, bool(is_causal), grads)


def kernel_takes(q, k, v, is_causal, fits=_device_fits):
    """True when ktt_flash takes q, k, v: bf16 tt [B, H, S, D], H a multiple of Hk,
    flash allowed by the scope, and the kernel's own check (`fits`: L1, runtime
    arguments, head dims; with gradients its backward, causal Sq == Sk) passes.
    Otherwise the explicit-mask route runs, never a deferred stream error."""
    tensors = (q, k, v)
    grads = torch.is_grad_enabled() and any(t.requires_grad for t in tensors)
    return (
        "flash" in current_numerics().attention_kernels
        and all(t.device.type == "tt" and t.dtype == torch.bfloat16 for t in tensors)
        and all(t.dim() == 4 for t in tensors)
        and q.size(1) % k.size(1) == 0
        and bool(fits(q, k, v, is_causal, grads))
    )


class _Lengths(torch.autograd.Function):
    """The kernel route: output and dq, dk, dv; the lengths get no gradient."""

    @staticmethod
    def forward(ctx, q, k, v, lengths, causal, scale):
        out = torch.ops.ktt._sdpa_lengths(q, k, v, lengths, causal, scale, True)
        ctx.save_for_backward(q, k, v, lengths, out)
        ctx.args = (causal, scale)
        return out

    @staticmethod
    def backward(ctx, grad):
        q, k, v, lengths, out = ctx.saved_tensors
        causal, scale = ctx.args
        dq, dk, dv = torch.ops.ktt._sdpa_lengths_backward(
            grad.contiguous(), out, q, k, v, lengths, causal, scale
        )
        return dq, dk, dv, None, None, None


def kernel(q, k, v, lengths, is_causal, scale):
    """The ktt_flash route; records autograd when q, k or v requires grad."""
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        return _Lengths.apply(q, k, v, lengths, bool(is_causal), scale)
    return torch.ops.ktt._sdpa_lengths(q, k, v, lengths, bool(is_causal), scale, False)


def masked(q, k, v, lengths, is_causal, scale):
    """The explicit-mask route: a bool mask [B, 1, Sq, Sk] built on the device."""
    sq, sk = q.size(-2), k.size(-2)
    keys = torch.arange(sk, device=q.device, dtype=torch.int32)
    keep = (keys[None, :] < lengths[:, None])[:, None, None, :]
    if is_causal:
        rows = torch.arange(sq, device=q.device, dtype=torch.int32)
        keep = keep & (keys[None, :] <= rows[:, None])[None, None]
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=keep, scale=scale, enable_gqa=q.size(1) != k.size(1)
    )
