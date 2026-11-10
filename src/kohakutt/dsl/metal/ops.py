"""The table of tile operations a compute kernel can name: per op its TT-Metal call,
its setup call, its header and its operands. One row adds an op to `c.op`.
"""

import struct
from dataclasses import dataclass

from kohakutt.dsl.metal.binding import Expr

API = "api/compute/"


@dataclass(frozen=True)
class TileOp:
    """Tile op: `call(slots..., values...)` after `init()`, declared in `header`.
    `slots` register operands (the last is the result), `values` fp32 scalars."""

    call: str
    init: str
    header: str
    slots: int = 1
    values: int = 0


def _unary(name, header="compute_kernel_api.h", call=None):
    call = call or f"{name}_tile"
    return TileOp(call, f"{call}_init", API + header)


def _scalar(name):
    header = API + "eltwise_unary/binop_with_scalar.h"
    return TileOp(f"{name}_unary_tile", "binop_with_scalar_tile_init", header, 1, 1)


def _binary(name, header="eltwise_binary_sfpu.h", call=None):
    call = call or f"{name}_binary_tile"
    return TileOp(call, f"{call}_init", API + header, 3)


UNARY = "eltwise_unary/"
OPS = {
    **{name: _unary(name) for name in ("silu", "sigmoid", "tanh", "log", "abs")},
    **{name: _unary(name) for name in ("sign", "square")},
    "exp": _unary("exp", UNARY + "exp.h"),
    "recip": _unary("recip", UNARY + "recip.h"),
    "relu": _unary("relu", UNARY + "relu.h"),
    "rsqrt": _unary("rsqrt", UNARY + "rsqrt.h"),
    "sqrt": _unary("sqrt", UNARY + "sqrt.h"),
    "neg": _unary("neg", UNARY + "negative.h", "negative_tile"),
    "gelu": _unary("gelu", UNARY + "gelu.h"),
    **{name: _unary(name, UNARY + "comp.h") for name in ("gtz", "ltz", "gez", "lez")},
    **{name: _unary(name, UNARY + "comp.h") for name in ("eqz", "nez")},
    **{f"{name}_scalar": _scalar(name) for name in ("add", "sub", "mul", "div")},
    "rsub_scalar": _scalar("rsub"),
    "fill": TileOp("fill_tile_bitcast", "fill_tile_init", API + UNARY + "fill.h", 1, 1),
    **{name: _binary(name) for name in ("add", "sub", "mul", "div", "rsub")},
    "pow": _binary("pow", call="power_binary_tile"),
    **{name: _binary(name) for name in ("eq", "ne", "lt", "gt", "le", "ge")},
    "max": _binary("max", "binary_max_min.h", "binary_max_tile"),
    "min": _binary("min", "binary_max_min.h", "binary_min_tile"),
}

# Setup calls that take a buffer: the unpacker's ways of filling a register slot.
UNPACK_INITS = {
    "copy": "copy_init({cb})",
    "transpose": "transpose_init({cb})",
    "bcast_row": "unary_bcast_init<BroadcastType::ROW>({cb})",
    "bcast_col": "unary_bcast_init<BroadcastType::COL>({cb})",
}
UNPACK_CALLS = {
    "copy": "copy_tile({cb}, {index}, {dst})",
    "transpose": "transpose_tile({cb}, {index}, {dst})",
    "bcast_row": "unary_bcast<BroadcastType::ROW>({cb}, {index}, {dst})",
    "bcast_col": "unary_bcast<BroadcastType::COL>({cb}, {index}, {dst})",
    "copy_block": "copy_block({cb}, {index}, {dst}, {count})",
}
UNPACK_HEADERS = {
    "copy": API + "tile_move_copy.h",
    "copy_block": API + "tile_move_copy.h",
    "transpose": API + "transpose.h",
    "bcast_row": API + "bcast.h",
    "bcast_col": API + "bcast.h",
}
# The setup a way of unpacking needs (a block copy runs under the copy setup).
UNPACK_SETUP = {**{name: name for name in UNPACK_INITS}, "copy_block": "copy"}


def bits(value):
    """The C++ text of fp32 scalar `value`: a Python float as its bit pattern, a
    kernel value (already the bits) as itself."""
    if isinstance(value, Expr):
        return value.cpp()
    return f"0x{struct.unpack('<I', struct.pack('<f', float(value)))[0]:08x}u"
