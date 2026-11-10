"""Packed variable-length attention on tt: the tt kernel of torch's
torch_attn::_varlen_attn (torch.nn.attention.varlen.varlen_attn), which
kohakutt.sdpa_varlen also calls, and the per-length-set plan constants it keeps on the
device."""

import collections

import torch

# Plans kept on the device, most recently used last: (offsets of queries, offsets of
# keys, causal, scale, key side) -> (records, mask tiles) as tt tensors.
_PLANS = collections.OrderedDict()
_MAX_PLANS = 32


def offsets(cu_seq):
    """The cumulative sequence offsets as a tuple of ints (a list, or a CPU or tt int
    tensor; a tt tensor is downloaded)."""
    if isinstance(cu_seq, torch.Tensor):
        return tuple(int(x) for x in cu_seq.detach().cpu().tolist())
    return tuple(int(x) for x in cu_seq)


def plan(cu_q, cu_k, is_causal, scale, device, key_side=False):
    """The plan tensors of an offset set and softmax scale on `device`: built on the
    host and uploaded on first use, then reused. `key_side`: the plan by key tile row
    (the backward's dK / dV program) instead of by query tile row."""
    key = (cu_q, cu_k, bool(is_causal), float(scale), bool(key_side))
    if key in _PLANS:
        _PLANS.move_to_end(key)
        return _PLANS[key]
    records, masks = torch.ops.ktt._varlen_plan(
        list(cu_q), list(cu_k), bool(is_causal), float(scale), bool(key_side)
    )
    entry = (records.to(device), masks.to(device))
    _PLANS[key] = entry
    while len(_PLANS) > _MAX_PLANS:
        _PLANS.popitem(last=False)
    return entry


def release():
    """Drops the plans kept on the device (before it closes)."""
    _PLANS.clear()


def _unsupported(**given):
    names = [name for name, value in given.items() if value is not None]
    if names:
        raise NotImplementedError(
            f"kohakutt: varlen attention on tt does not support {', '.join(names)}"
        )


def _forward(
    query,
    key,
    value,
    cu_seq_q,
    cu_seq_k,
    max_q,
    max_k,
    is_causal=False,
    scale=None,
    window_size=None,
    enable_gqa=False,
    seqused_k=None,
    block_table=None,
    num_splits=None,
    backend=1,
):
    """torch_attn::_varlen_attn on tt: (output [Tq, H, Dv], log-sum-exp [H, Tq], rng
    state)."""
    _unsupported(seqused_k=seqused_k, block_table=block_table)
    causal = _causal(is_causal, window_size)
    cu_q = offsets(cu_seq_q)
    cu_k = cu_q if cu_seq_k is None else offsets(cu_seq_k)
    if scale is None:
        scale = query.size(-1) ** -0.5
    records, masks = plan(cu_q, cu_k, causal, scale, query.device)
    out, lse = torch.ops.ktt._sdpa_varlen(
        query, key, value, records, masks, list(cu_q), list(cu_k), causal, scale, True
    )
    return out, lse, torch.zeros((2,), dtype=torch.uint64)


def _causal(is_causal, window_size):
    """True for causal attention; raises for windows other than (-1, -1) and (-1, 0)."""
    window = list(window_size) if window_size is not None else [-1, -1]
    if window not in ([-1, -1], [-1, 0]):
        raise NotImplementedError(
            "kohakutt: varlen attention on tt takes window_size (-1, -1) or (-1, 0)"
        )
    return bool(is_causal) or window == [-1, 0]


def _backward(
    grad_out,
    query,
    key,
    value,
    out,
    lse,
    cu_seq_q,
    cu_seq_k,
    max_q,
    max_k,
    is_causal,
    rng_state,
    scale=None,
    window_size=None,
    backend=1,
):
    """torch_attn::_varlen_attn_backward on tt: (dq, dk, dv), packed as query, key,
    value. `out` must be the forward's output (its row statistics stay on the device
    with it; `lse` is not read)."""
    causal = _causal(is_causal, window_size)
    cu_q = offsets(cu_seq_q)
    cu_k = cu_q if cu_seq_k is None else offsets(cu_seq_k)
    if scale is None:
        scale = query.size(-1) ** -0.5
    records, masks = plan(cu_q, cu_k, causal, scale, query.device)
    key_records, key_masks = plan(
        cu_q, cu_k, causal, scale, query.device, key_side=True
    )
    return torch.ops.ktt._sdpa_varlen_backward(
        grad_out,
        out,
        query,
        key,
        value,
        records,
        masks,
        key_records,
        key_masks,
        list(cu_q),
        list(cu_k),
        causal,
        scale,
    )


# The logit of a key a bool mask leaves out, and the floor of an additive mask.
_MASKED_LOGIT = -1e4


def key_mask(mask, scale, keys, device):
    """The key mask `mask` [Tk] as the kernels take it, on `device`: bf16, added to the
    scores of its key, divided by the softmax scale. A bool mask keeps the keys that are
    True; a float mask is additive, clamped at -1e4. A CPU mask is converted on the host
    and uploaded; a tt mask is converted on the device."""
    if mask.dim() != 1 or mask.size(0) != keys:
        raise ValueError(
            f"kohakutt.sdpa_varlen: attn_mask must be [{keys}] "
            "(one value per packed key), "
            f"got {tuple(mask.shape)}"
        )
    if mask.dtype == torch.bool:
        additive = (mask.to(torch.bfloat16) - 1.0) * (-_MASKED_LOGIT / scale)
    else:
        additive = mask.to(torch.bfloat16).clamp_min(_MASKED_LOGIT) * (1.0 / scale)
    return additive.to(device)


class _Masked(torch.autograd.Function):
    """Packed variable-length attention with a key mask (key_mask's form): the forward
    and its dq, dk, dv; the mask gets no gradient."""

    @staticmethod
    def forward(ctx, q, k, v, mask, cu_q, cu_k, causal, scale):
        records, masks = plan(cu_q, cu_k, causal, scale, q.device)
        out, _ = torch.ops.ktt._sdpa_varlen(
            q, k, v, records, masks, list(cu_q), list(cu_k), causal, scale, True, mask
        )
        ctx.save_for_backward(q, k, v, mask, out)
        ctx.lengths = (cu_q, cu_k, causal, scale)
        return out

    @staticmethod
    def backward(ctx, grad):
        q, k, v, mask, out = ctx.saved_tensors
        cu_q, cu_k, causal, scale = ctx.lengths
        records, masks = plan(cu_q, cu_k, causal, scale, q.device)
        key_records, key_masks = plan(
            cu_q, cu_k, causal, scale, q.device, key_side=True
        )
        dq, dk, dv = torch.ops.ktt._sdpa_varlen_backward(
            grad,
            out,
            q,
            k,
            v,
            records,
            masks,
            key_records,
            key_masks,
            list(cu_q),
            list(cu_k),
            causal,
            scale,
            mask,
        )
        return dq, dk, dv, None, None, None, None, None


def masked(q, k, v, mask, cu_q, cu_k, causal, scale):
    """Packed variable-length attention of q, k, v (offsets `cu_q` / `cu_k` as tuples)
    with key mask `mask` (key_mask's result): the output [Tq, H, Dv]."""
    return _Masked.apply(q, k, v, mask, cu_q, cu_k, bool(causal), float(scale))


def install():
    """Registers the tt kernels of torch's varlen attention ops."""
    from torch.nn.attention import varlen

    varlen._varlen_attn.register_kernel("tt")(_forward)
    varlen._varlen_attn_backward.register_kernel("tt")(_backward)
