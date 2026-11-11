"""Attention entry points of the package beyond torch's own."""

import torch
import torch.nn.functional as F

from kohakutt.torch_backend import lengths as _lengths
from kohakutt.torch_backend import varlen


def sdpa(q, k, v, key_lengths=None, is_causal=False, scale=None):
    """F.scaled_dot_product_attention of q [B, H, Sq, D], k / v [B, Hk, Sk, D / Dv]
    with the key mask key < key_lengths[b] (B ints in [1, Sk]: list, CPU or tt tensor),
    built nowhere: the kernel skips and masks those keys. Query rows past a length see
    the real keys; docs/reference/kernels/flash.md, Key lengths."""
    gqa = q.size(1) != k.size(1)
    if key_lengths is None:
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, scale=scale, enable_gqa=gqa
        )
    lengths = _lengths.device_lengths(key_lengths, q.size(0), k.size(-2), q.device)
    if _lengths.kernel_takes(q, k, v, is_causal):
        return _lengths.kernel(q, k, v, lengths, is_causal, scale)
    return _lengths.masked(q, k, v, lengths, is_causal, scale)


def sdpa_varlen(
    q,
    k,
    v,
    cu_seq_q,
    cu_seq_k=None,
    max_q=None,
    max_k=None,
    is_causal=False,
    scale=None,
    attn_mask=None,
):
    """Scaled dot-product attention over packed variable-length sequences on tt.

    q [Tq, H, D], k [Tk, Hkv, D] and v [Tk, Hkv, Dv] (bf16, H a multiple of Hkv) hold
    the tokens of every sequence back to back; nothing is padded. cu_seq_q / cu_seq_k
    are the cumulative sequence offsets [0, ..., total] (a list or an int tensor;
    cu_seq_k defaults to cu_seq_q). Each query attends to the keys of its own sequence;
    with is_causal query r of a sequence sees its first min(Lk, r + 1 + Lk - Lq) keys.
    Returns [Tq, H, Dv].

    attn_mask: a key mask [Tk], one value per packed key, for every query of the key's
    sequence and every head. Bool: True keeps the key. Float: added to the key's scores
    (clamped at -5e5). It gets no gradient. Other mask shapes are not taken.

    max_q / max_k (the longest sequence) are accepted for torch's signature and not
    needed. The contract is torch.nn.attention.varlen.varlen_attn's, which runs the same
    kernel on tt tensors.
    """
    cu_q = varlen.offsets(cu_seq_q)
    cu_k = cu_q if cu_seq_k is None else varlen.offsets(cu_seq_k)
    if attn_mask is not None:
        softmax_scale = q.size(-1) ** -0.5 if scale is None else float(scale)
        mask = varlen.key_mask(attn_mask, softmax_scale, k.size(0), q.device)
        return varlen.masked(q, k, v, mask, cu_q, cu_k, is_causal, softmax_scale)
    lengths_q = [b - a for a, b in zip(cu_q, cu_q[1:])]
    lengths_k = [b - a for a, b in zip(cu_k, cu_k[1:])]
    out, _, _ = torch.ops.torch_attn._varlen_attn(
        q,
        k,
        v,
        torch.tensor(cu_q, dtype=torch.int32),
        torch.tensor(cu_k, dtype=torch.int32),
        max(lengths_q, default=0) if max_q is None else max_q,
        max(lengths_k, default=0) if max_k is None else max_k,
        bool(is_causal),
        scale,
        [-1, 0] if is_causal else [-1, -1],
        q.size(1) != k.size(1),
    )
    return out
