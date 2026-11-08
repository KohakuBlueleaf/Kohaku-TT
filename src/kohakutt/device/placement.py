"""Mesh placements of tt tensors: placement() reports one, shard() / replicate() force
one, memory_stats() reports memory over the mesh."""

import re

import torch

from kohakutt.device.session import live_index


def _mesh_shape():
    return list(torch.ops.ktt.mesh_shape())


def _largest_axis(mesh):
    return max(range(len(mesh)), key=lambda a: mesh[a])


def shard(tensor, dim):
    """A copy of tt tensor `tensor` sharded on `dim` over the largest mesh axis (other
    axes replicated)."""
    mesh = _mesh_shape()
    spec = ["R"] * len(mesh)
    spec[_largest_axis(mesh)] = f"S({dim})"
    return torch.ops.ktt.distribute(tensor, ",".join(spec))


def replicate(tensor):
    """A replicated copy of tt tensor `tensor`."""
    return torch.ops.ktt.distribute(tensor, ",".join(["R"] * len(_mesh_shape())))


def placement(tensor):
    """The placement of tt tensor `tensor` over the mesh axes of size > 1:
    "Replicated", "Sharded(dim=d)" (d a dim of `tensor`) or "Partial(sum|max|min)",
    comma-joined when several axes have size > 1."""
    mesh = _mesh_shape()
    axes = list(
        re.finditer(r"R|S\((\d+)\)|P\((\w+)\)", torch.ops.ktt.placement(tensor))
    )
    # Device dims of rank 0 / 1 tensors carry leading unit dims.
    lift = max(0, 2 - tensor.dim())
    names = []
    for axis, size in enumerate(mesh):
        if size == 1:
            continue
        item = axes[axis] if axis < len(axes) else None
        if item is None or item.group(0) == "R":
            names.append("Replicated")
        elif item.group(1) is not None:
            names.append(f"Sharded(dim={int(item.group(1)) - lift})")
        else:
            names.append(f"Partial({item.group(2)})")
    return ", ".join(names) or "Replicated"


def memory_stats(device=None, per_chip=False):
    """{allocated, peak, free, total} device bytes of mesh `device` (default the live
    mesh): live tt tensors (padded) and their peak, DRAM free and allocatable; summed
    over the chips (a replicated tensor counts once per chip), or one chip's with
    `per_chip`."""
    live_index(device)
    allocated, peak = torch.ops.ktt.memory_stats()
    free, total = torch.ops.ktt.mem_get_info()
    chips = max(1, torch.ops.ktt.mesh_size())
    if per_chip:
        return {
            "allocated": allocated,
            "peak": peak,
            "free": free // chips,
            "total": total // chips,
        }
    return {
        "allocated": allocated * chips,
        "peak": peak * chips,
        "free": free,
        "total": total,
    }
