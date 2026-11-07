"""TT block-float weights: bf16 tt tensors stored as BFLOAT8_B ("bfp8") or BFLOAT4_B
("bfp4"), read as bf16; F.linear / F.scaled_mm use them as stored."""

import torch
from torch import nn

FORMATS = ("bfp8", "bfp4")


def block_float_(tensor, format="bfp8"):
    """Stores bf16 tt `tensor` (covering its whole storage) as block floats in place;
    returns it."""
    return torch.ops.ktt.block_float_(tensor, format)


def storage_format(tensor):
    """The storage format: "bfp8" / "bfp4" for block floats, else the dtype's name."""
    return torch.ops.ktt.storage_format(tensor)


def block_float_linears(module, format="bfp8", skip=(), kn=True):
    """Stores the bf16 weight of every nn.Linear in `module` as block floats (`format`
    None: kept bf16), except under a qualified-name prefix in `skip`; returns the
    converted names. With `kn` the storage is the (in, out) transpose and `weight` its
    transposed view (same values; F.linear then runs TT-NN's minimal matmul)."""
    done = []
    for name, child in module.named_modules():
        if not isinstance(child, nn.Linear) or name.startswith(tuple(skip)):
            continue
        weight = child.weight
        if weight.dtype != torch.bfloat16 or weight.device.type == "cpu":
            continue
        if kn:
            stored = weight.data.t().contiguous()
            if format is not None:
                block_float_(stored, format)
            child.weight = nn.Parameter(stored.t(), requires_grad=weight.requires_grad)
        elif format is not None:
            block_float_(weight.data, format)
        done.append(name)
    return done
