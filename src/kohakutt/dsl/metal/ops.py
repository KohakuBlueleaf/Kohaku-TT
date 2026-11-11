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
INT32 = "<DataFormat::Int32>"
# typecast_tile's (input, output) format codes: DataFormat Int32 is 8, Float32 0.
FORMATS, BACK = "8u, 0u", "0u, 8u"
TYPECAST = API + UNARY + "typecast.h"


def _int(name, header, init=None, templated_init=False):
    """Binary int32 op `name`: `{name}_tile<Int32>(a, b, out)`."""
    init = init or f"{name}_tile_init"
    init = init + (INT32 if templated_init else "")
    return TileOp(f"{name}_tile{INT32}", init, API + header, 3)


# Lanewise int32 ops (int32 registers): arithmetic, comparisons (1 or 0), bitwise,
# shifts; named `<op>_int32`.
INTEGER = {
    "add_int32": _int("add_int", "add_int_sfpu.h"),
    "sub_int32": _int("sub_int", "sub_int_sfpu.h"),
    "mul_int32": _int("mul_int", "mul_int_sfpu.h", templated_init=True),
    **{
        f"{cmp}_int32": _int(f"{cmp}_int", "binary_comp.h", templated_init=True)
        for cmp in ("eq", "ne", "lt", "gt", "le", "ge")
    },
    **{
        f"{op}_int32": TileOp(
            f"bitwise_{op}_binary_tile{INT32}",
            "binary_bitwise_tile_init",
            API + "binary_bitwise_sfpu.h",
            3,
        )
        for op in ("and", "or", "xor")
    },
    **{
        f"{name}_int32": TileOp(
            f"binary_{call}_tile{INT32}",
            "binary_shift_tile_init",
            API + "binary_shift.h",
            3,
        )
        for name, call in (("shl", "left_shift"), ("shr", "right_shift"))
    },
    # Lanewise conversions between int32 and fp32 registers.
    "int32_to_fp32": TileOp(
        f"typecast_tile<{FORMATS}>", f"typecast_tile_init<{FORMATS}>", TYPECAST
    ),
    "fp32_to_int32": TileOp(
        f"typecast_tile<{BACK}>", f"typecast_tile_init<{BACK}>", TYPECAST
    ),
}

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
    **INTEGER,
    "fill": TileOp("fill_tile_bitcast", "fill_tile_init", API + UNARY + "fill.h", 1, 1),
    **{name: _binary(name) for name in ("add", "sub", "mul", "div", "rsub")},
    "pow": _binary("pow", call="power_binary_tile"),
    **{name: _binary(name) for name in ("eq", "ne", "lt", "gt", "le", "ge")},
    "max": _binary("max", "binary_max_min.h", "binary_max_tile"),
    "min": _binary("min", "binary_max_min.h", "binary_min_tile"),
    # A slot's tile transposed in place (rows and columns swapped).
    "transpose_dest": TileOp(
        "transpose_dest<DST_ACCUM_MODE>",
        "transpose_dest_init<DST_ACCUM_MODE>",
        API + "transpose_dest.h",
    ),
    # A setup only: `c.init("stats")`, the one every `c.stats_*` call runs under.
    "stats": TileOp(None, "two_pass_stats_init_shifted", API + "welford.h", 0),
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
