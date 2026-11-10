"""Everything specific to a tinygrad release: internal imports, op names, the seam,
the PROGRAM handed back to the scheduler, launch arguments, device registration.
The rest of the backend imports tinygrad internals only from here (pinned to 0.14.x).
"""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass

import numpy as np
import tinygrad.engine.realize as realize
from tinygrad.device import Allocator, Compiled, Program, canonicalize_device
from tinygrad.dtype import AddrSpace, _to_np_dtype, to_dtype
from tinygrad.helpers import (
    TRAINING,
    Context,
    argfix,
    flat_to_grouped,
    resolve_pool_pads,
)
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import Invalid, KernelInfo, Ops, ProgramInfo, UOp

import tinygrad
from tinygrad import Tensor, TinyJit, Variable, dtypes, nn

SUPPORTED = "0.14"
DEVICE_NAME = "KTT"

__all__ = [
    "AddrSpace",
    "Allocator",
    "Compiled",
    "Context",
    "Invalid",
    "KernelInfo",
    "Ops",
    "Program",
    "ProgramInfo",
    "Renderer",
    "Tensor",
    "TinyJit",
    "UOp",
    "Variable",
    "argfix",
    "canonicalize_device",
    "dtypes",
    "flat_to_grouped",
    "nn",
    "np_dtype",
    "np_storage",
    "resolve_pool_pads",
    "to_dtype",
]


def check_version():
    """Raise unless tinygrad is the pinned release (KTT_TINYGRAD_ANY=1: warn only)."""
    version = getattr(tinygrad, "__version__", None)
    if version is None:
        from importlib.metadata import version as dist_version

        version = dist_version("tinygrad")
    if version.startswith(SUPPORTED):
        return version
    message = f"kohakutt.tinygrad targets tinygrad {SUPPORTED}.x, found {version}"
    if os.environ.get("KTT_TINYGRAD_ANY") == "1":
        print("warning:", message, file=sys.stderr)
        return version
    raise ImportError(message + " (set KTT_TINYGRAD_ANY=1 to try anyway)")


# ---- op names: one table for the names that differ between releases


def _ops(*names):
    return {getattr(Ops, n) for n in names if hasattr(Ops, n)}


INT_DIV = _ops("CDIV", "FLOORDIV", "TRUNCDIV", "DIV")
INT_MOD = _ops("CMOD", "FLOORMOD", "TRUNCMOD", "MOD")
RECIP = getattr(Ops, "RECIPROCAL", None) or Ops.RECIP
INDEX_DTYPES = (dtypes.weakint,)


def np_dtype(dt):
    """numpy dtype values of tinygrad dtype `dt` are computed in on the host (bfloat16:
    float32 holding bf16 values); index and bool types included."""
    if dt.fmt is not None:
        return _to_np_dtype(dt)
    if dt == dtypes.bool:
        return np.bool_
    if dt == dtypes.bfloat16:
        return np.float32
    return np.float64 if dtypes.is_float(dt) else np.int64


def np_storage(dt):
    """numpy dtype of a `dt` buffer's bytes (bfloat16: uint16 bits); None for dtypes
    the host evaluator does not model (the fp8 formats)."""
    if dt == dtypes.bfloat16:
        return np.uint16
    return _to_np_dtype(dt) if dt.fmt is not None else None


DTYPE_NAMES = {
    dtypes.float16: "float16",
    dtypes.float32: "float32",
    dtypes.float64: "float64",
    dtypes.bfloat16: "bfloat16",
    dtypes.int8: "int8",
    dtypes.int16: "int16",
    dtypes.int32: "int32",
    dtypes.int64: "int64",
    dtypes.uint8: "uint8",
    dtypes.uint16: "uint16",
    dtypes.uint32: "uint32",
    dtypes.uint64: "uint64",
    dtypes.bool: "bool",
}


def dtype_name(dt) -> str:
    """The dtype's numpy-style name (float16, not half)."""
    return DTYPE_NAMES.get(dt, getattr(dt, "name", str(dt)))


def is_index_dtype(dt) -> bool:
    return dt in INDEX_DTYPES


# ---- kernel parameters: buffers and variables of a kernel AST


@dataclass(frozen=True)
class Param:
    slot: int
    name: str | None
    dtype: object
    is_var: bool
    size: int | None  # elements, for buffers


def kernel_params(ast: UOp) -> list[Param]:
    """The PARAMs of a kernel AST by slot: buffers (is_var False), variables (True)."""
    seen = {}
    for u in ast.toposort():
        if u.op is Ops.PARAM:
            is_var = u.addrspace == AddrSpace.ALU
            size = (
                None
                if is_var
                else int(u.src[0].arg) if u.src and u.src[0].op is Ops.CONST else None
            )
            seen[(u.arg.slot, is_var)] = Param(
                u.arg.slot, u.arg.name, u.dtype, is_var, size
            )
    return sorted(seen.values(), key=lambda p: (p.is_var, p.slot))


def buffer_params(ast: UOp) -> list[Param]:
    return [p for p in kernel_params(ast) if not p.is_var]


def var_params(ast: UOp) -> list[Param]:
    return [p for p in kernel_params(ast) if p.is_var]


def is_buffer_param(u: UOp) -> bool:
    return u.op is Ops.PARAM and u.addrspace != AddrSpace.ALU


def param_slot(u: UOp) -> int:
    return u.arg.slot


def param_var_name(u: UOp) -> str:
    return u.arg.name


# ---- the seam


_seam_handlers: dict[str, object] = {}
_orig_to_program = realize.to_program
_installed = False


def program_uop(ast: UOp, renderer: Renderer, source: str) -> UOp:
    """The PROGRAM the scheduler expects from to_program: SOURCE names the plan, BINARY
    carries the AST key. Its SINK is a tagged copy of `ast`, since the original SINK
    must not occur inside the PROGRAM that replaces it."""
    info = ProgramInfo.from_sink(ast, renderer.target)
    params = tuple(
        sorted((u for u in ast.toposort() if is_buffer_param(u)), key=param_slot)
    )
    tagged = ast.replace(tag="ktt")
    return UOp(
        Ops.PROGRAM,
        src=(
            tagged,
            UOp(Ops.LINEAR, src=params),
            UOp(Ops.SOURCE, arg=source),
            UOp(Ops.BINARY, arg=ast.key),
        ),
        arg=info,
    )


def lib_program_uop(prg: UOp, renderer: Renderer) -> UOp:
    """A custom_kernel PROGRAM (source "lib ...") with its ProgramInfo and BINARY."""
    return prg.replace(
        src=prg.src[:3] + (UOp(Ops.BINARY, arg=prg.key),),
        arg=ProgramInfo.from_sink(prg.src[0], renderer.target),
    )


def is_lib_program(ast: UOp) -> bool:
    return (
        ast.op is Ops.PROGRAM
        and ast.src[2].op is Ops.SOURCE
        and str(ast.src[2].arg).startswith("lib ")
    )


def set_seam(handler):
    """Replace realize.to_program: `handler` serves KTT kernels, tinygrad the rest."""
    global _installed
    if _installed:
        return

    def to_program(ast: UOp, renderer: Renderer):
        if renderer.target.device == DEVICE_NAME:
            return handler(ast, renderer)
        return _orig_to_program(ast, renderer)

    realize.to_program = to_program
    _installed = True


def program_buffers(info: ProgramInfo) -> tuple[int, ...]:
    """Buffer slots of a launch, in the order Program.__call__ receives them."""
    return info.globals


def launch_args(args: tuple, vals: tuple, params: list[Param], var_names: list[str]):
    """(buffers by slot, variable values by name) of one launch; `args` are buffer
    handles in slot order, `vals` the values in the order of ProgramInfo.vars."""
    bufs = {p.slot: a for p, a in zip((q for q in params if not q.is_var), args)}
    return bufs, dict(zip(var_names, vals))


# ---- registration


def register_device(device_cls):
    """Make Device["KTT"] resolve to device_cls (as tinygrad.runtime.ops_ktt)."""
    mod = types.ModuleType("tinygrad.runtime.ops_ktt")
    mod.KTTDevice = device_cls
    sys.modules["tinygrad.runtime.ops_ktt"] = mod


def set_knobs(device_max_bufs: int | None = None):
    """Write the tinygrad settings the device needs (no parallel compile, no BEAM)."""
    from tinygrad import helpers

    for name, value in (
        ("PARALLEL", 0),
        ("BEAM", 0),
        ("SPLIT_REDUCEOP", 0),
        ("NO_MEMORY_PLANNER", 1),
    ):
        var = getattr(helpers, name, None)
        if var is not None and hasattr(var, "value"):
            var.value = value
        else:
            os.environ[name] = str(value)


def training() -> bool:
    """True inside Context(TRAINING=1)."""
    return bool(TRAINING.value)


def custom_kernel(out: Tensor, *inputs: Tensor, fxn, grad_fxn=None) -> Tensor:
    """Tensor.custom_kernel(...)[0]; the first argument is the output buffer."""
    return out.custom_kernel(*inputs, fxn=fxn, grad_fxn=grad_fxn)[0]


def kernel_sink(name: str, *args: UOp) -> UOp:
    return UOp.sink(*args, arg=KernelInfo(name=name))


def source_program(sink: UOp, source: str) -> UOp:
    return UOp(
        Ops.PROGRAM,
        src=(
            sink,
            UOp(Ops.LINEAR, src=tuple(sink.toposort())),
            UOp(Ops.SOURCE, arg=source),
        ),
    )
