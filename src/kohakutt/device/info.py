"""Facts about a live tt mesh (chip count, compute grid) and the native layer's
process-wide counters."""

import torch

from kohakutt.device.session import live_index


def mesh_size(device=None):
    """Number of chips in mesh `device` (a torch.device from mesh() / device();
    default the live mesh)."""
    live_index(device)
    return torch.ops.ktt.mesh_size()


def compute_grid(device=None):
    """(columns, rows) of one chip's Tensix compute grid in mesh `device`."""
    live_index(device)
    cols, rows = torch.ops.ktt.compute_grid()
    return cols, rows


def counters():
    """{uploads, downloads, ops, op_ns} done by the native layer since load, over all
    meshes; op_ns is host time inside TT-NN op calls."""
    uploads, downloads, ops, op_ns = torch.ops.ktt.counters()
    return {"uploads": uploads, "downloads": downloads, "ops": ops, "op_ns": op_ns}
