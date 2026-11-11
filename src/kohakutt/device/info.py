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
    "device_bytes",
    "device_bytes_peak",
    "attention_flash",
    "attention_own",
    "attention_ttnn",
    "attention_math",
    "attention_backward_flash",
    "attention_backward_ttnn",
    "attention_backward_math",
)


def counters():
    """Counts of the native layer since load, over all meshes, keyed by COUNTERS: host
    transfers, TT-NN ops and host ns, own-program launches, cache hits and host ns per
    phase, TT-Metal cache entries and kernel compiles, live device tile bytes per chip
    (and peak), attention calls per kernel (attention_*, attention_backward_*)."""
    return dict(zip(COUNTERS, torch.ops.ktt.counters(), strict=True))


def head_major_reads():
    """{op: reads} of head-major tensors in the tile form of their own shape, each a
    conversion, since load or the last reset_head_major_reads()."""
    pairs = (item.rsplit("=", 1) for item in torch.ops.ktt.head_major_reads())
    return {name: int(count) for name, count in pairs}


def reset_head_major_reads():
    """Clears head_major_reads()."""
    torch.ops.ktt.reset_head_major_reads()
