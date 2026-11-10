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


COUNTERS = (
    "uploads",
    "downloads",
    "ops",
    "op_ns",
    "programs",
    "desc_hits",
    "alloc_ns",
    "args_ns",
    "desc_ns",
    "enqueue_ns",
    "program_entries",
    "kernel_compiles",
)


def counters():
    """Counts of the native layer since load, over all meshes: uploads, downloads,
    ops, op_ns (host time inside TT-NN op calls); for Kohaku-TT's own programs:
    programs (launches), desc_hits (descriptor-cache hits) and host ns per phase,
    alloc_ns (outputs), args_ns (runtime arguments), desc_ns (the rest of the
    descriptor, or the cache lookup and patch) and enqueue_ns (generic_op); from
    TT-Metal: program_entries (program-cache entries after the last own launch) and
    kernel_compiles (kernel sources its JIT build compiled)."""
    return dict(zip(COUNTERS, torch.ops.ktt.counters(), strict=True))
