"""Kohaku-TT: PyTorch on Tenstorrent Blackhole as the native device "tt". Importing
registers the device (never opens it); kohakutt.device(0) or kohakutt.mesh([0, 1]) gives
a torch.device for .to(), then ordinary torch code runs on it. See examples/.
"""

from kohakutt import _env  # noqa: F401  (first: environment defaults)
from kohakutt.attention import sdpa, sdpa_varlen
from kohakutt.block_float import block_float_, block_float_linears, storage_format
from kohakutt.core.numerics import (
    Numerics,
    current_numerics,
    numerics,
    numerics_for,
    sdpa_kernel,
)
from kohakutt.core.stats import diff_stats, format_stats
from kohakutt.device import (
    BoardResetRequired,
    clear_program_cache,
    close,
    compute_grid,
    counters,
    device,
    head_major_reads,
    memory_stats,
    mesh,
    mesh_size,
    open,
    placement,
    placement_log,
    record_placements,
    replicate,
    reset_head_major_reads,
    reset_placement_log,
    shard,
    synchronize,
)
from kohakutt.device.marks import apply_marks, apply_plan, colwise, place, rowwise
from kohakutt.torch_backend import registration
from kohakutt.torch_backend.fallback import (
    UNSUPPORTED,
    KohakuTTUnsupportedOperator,
    fallback_counts,
    reset_fallback_counts,
)
from kohakutt.torch_backend.trace import Captured, capture

#: True when the native layer is built and the "tt" device is registered.
AVAILABLE = registration.install()

__all__ = [
    "AVAILABLE",
    "UNSUPPORTED",
    "BoardResetRequired",
    "Captured",
    "KohakuTTUnsupportedOperator",
    "Numerics",
    "apply_marks",
    "apply_plan",
    "block_float_",
    "block_float_linears",
    "capture",
    "clear_program_cache",
    "close",
    "colwise",
    "compute_grid",
    "counters",
    "current_numerics",
    "device",
    "diff_stats",
    "fallback_counts",
    "format_stats",
    "head_major_reads",
    "memory_stats",
    "mesh",
    "mesh_size",
    "numerics",
    "numerics_for",
    "open",
    "place",
    "placement",
    "placement_log",
    "record_placements",
    "replicate",
    "reset_fallback_counts",
    "reset_head_major_reads",
    "reset_placement_log",
    "rowwise",
    "sdpa",
    "sdpa_kernel",
    "sdpa_varlen",
    "shard",
    "storage_format",
    "synchronize",
]
__version__ = "0.1.0"
