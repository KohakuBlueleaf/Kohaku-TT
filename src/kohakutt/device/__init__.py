"""The tt device for users: device(chip) / mesh(chips) as torch devices, close, and
facts about the live mesh."""

from kohakutt.device.info import (
    compute_grid,
    counters,
    head_major_reads,
    mesh_size,
    reset_head_major_reads,
)
from kohakutt.device.placement import (
    memory_stats,
    placement,
    placement_log,
    record_placements,
    replicate,
    reset_placement_log,
    shard,
)
from kohakutt.device.session import (
    clear_program_cache,
    close,
    device,
    mesh,
    open,
    synchronize,
)

__all__ = [
    "clear_program_cache",
    "close",
    "compute_grid",
    "counters",
    "device",
    "head_major_reads",
    "memory_stats",
    "mesh",
    "mesh_size",
    "open",
    "placement",
    "placement_log",
    "record_placements",
    "replicate",
    "reset_head_major_reads",
    "reset_placement_log",
    "shard",
    "synchronize",
]
