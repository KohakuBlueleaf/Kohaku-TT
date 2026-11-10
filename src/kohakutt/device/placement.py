"""Mesh placements of tt tensors: placement() reports one, shard() / replicate() force
one, placement_log() lists the decisions made per op, memory_stats() reports memory
over the mesh."""

import contextlib
import re

import torch

from kohakutt.device.session import live_index


@torch._dynamo.assume_constant_result
def _mesh_shape():
    # A constant to torch.compile: the mesh is fixed while a graph runs.
    return list(torch.ops.ktt.mesh_shape())


def _largest_axis(mesh):
    return max(range(len(mesh)), key=lambda a: mesh[a])


def _check_marked(tensor, mark):
    """Reports (marks.conflict) an activation mark that asks a tensor marked in code or
    by a mesh_plan for another placement; the mark applies to the returned copy only.
    Under torch.compile the backend checks the graph instead."""
    if torch.compiler.is_compiling():
        return
    own = getattr(tensor, "_ktt_mark", None)
    if own is not None and own[1] in ("mark", "plan") and own[0] != mark:
        from kohakutt.device.marks import conflict

        conflict(
            f"kohakutt: a tensor marked {own[0]} ({own[1]}) is placed {mark} in"
            f" forward; the module's tensor keeps {own[0]}, the copy is {mark}"
        )


def shard(tensor, dim):
    """A copy of tt tensor `tensor` sharded on `dim` over the largest mesh axis (other
    axes replicated). Given an nn.Module, marks its tensors sharded on `dim` instead
    and returns the module (kohakutt.device.marks). Works under torch.compile."""
    if isinstance(tensor, torch.nn.Module):
        from kohakutt.device.marks import shard_module

        return shard_module(tensor, dim)
    _check_marked(tensor, f"S({dim})")
    mesh = _mesh_shape()
    spec = ["R"] * len(mesh)
    spec[_largest_axis(mesh)] = f"S({dim})"
    return torch.ops.ktt.distribute(tensor, ",".join(spec))


def replicate(tensor):
    """A replicated copy of tt tensor `tensor`. Given an nn.Module, marks its tensors
    replicated instead and returns the module (kohakutt.device.marks). Works under
    torch.compile."""
    if isinstance(tensor, torch.nn.Module):
        from kohakutt.device.marks import replicate_module

        return replicate_module(tensor)
    _check_marked(tensor, "R")
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


@contextlib.contextmanager
def record_placements():
    """Records the placement decision of every op call inside the block, for
    placement_log(); the earlier recording state returns on exit. KTT_MESH_EXPLAIN=1
    records from load."""
    before = torch.ops.ktt.mesh_record(True)
    try:
        yield
    finally:
        torch.ops.ktt.mesh_record(before)


def placement_log():
    """The recorded placement decisions, oldest first (the last 65536): per op call
    {"op", "inputs", "output", "conversions", "reason"}. Placements read "[R, S(0)]":
    one entry per mesh axis, over device dims. A conversion is {"input", "from", "to",
    "kind", "reused"}; `reused` is True when a kept copy served it."""
    out = []
    for line in torch.ops.ktt.mesh_placement_log():
        op, inputs, output, conversions, reason = line.split("\t")
        steps = []
        for item in filter(None, conversions.split("|")):
            index, source, target, kind, reused = item.split(">")
            steps.append(
                {
                    "input": int(index),
                    "from": source,
                    "to": target,
                    "kind": kind,
                    "reused": reused == "1",
                }
            )
        out.append(
            {
                "op": op,
                "inputs": inputs.split(";") if inputs else [],
                "output": output,
                "conversions": steps,
                "reason": reason,
            }
        )
    return out


def reset_placement_log():
    """Empties placement_log()."""
    torch.ops.ktt.reset_mesh_placement_log()


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
