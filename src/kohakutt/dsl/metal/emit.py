"""The C++ emitter: one TT-Metalium source per kernel of a verified program, written
under a name made from its content.
"""

import hashlib
import os
import textwrap
from pathlib import Path

from kohakutt.dsl.metal.binding import Expr
from kohakutt.dsl.metal.body import (
    Apply,
    Assign,
    Barrier,
    Binary,
    CBOp,
    Copy,
    ExitWhen,
    Format,
    Init,
    Inline,
    Let,
    Loop,
    Matmul,
    MatmulInit,
    Mcast,
    McastSet,
    Pack,
    PackAccumulate,
    PackFormat,
    Put,
    Read,
    Reduce,
    ReduceDone,
    ReduceInit,
    Registers,
    SemOp,
    Send,
    SfpuCall,
    SfpuFn,
    Slot,
    SrcaFormat,
    Startup,
    Take,
    Tiles,
    TilesInit,
    Unary,
    Waypoint,
    When,
    Write,
    Zone,
    walk,
)
from kohakutt.dsl.metal.ops import (
    OPS,
    UNPACK_CALLS,
    UNPACK_HEADERS,
    UNPACK_INITS,
    bits,
)
from kohakutt.dsl.metal.verify import _align, _loads, clobbers, sfpu_threads

VECTOR_MODES = {"RC": "VectorMode::RC", "none": "VectorMode::None"}

ROLE_TEXT = {"dm0": "reader", "dm1": "writer", "compute": "compute"}
CB_CALLS = {
    "reserve": "cb_reserve_back",
    "push": "cb_push_back",
    "wait": "cb_wait_front",
    "pop": "cb_pop_front",
}
BARRIERS = {
    "reads_done": "noc_async_read_barrier",
    "flushed": "noc_async_writes_flushed",
    "writes_done": "noc_async_write_barrier",
    "atomics_done": "noc_async_atomic_barrier",
}
WORD = "*reinterpret_cast<volatile tt_l1_ptr uint32_t*>({})"
REGISTERS = {
    "acquire": ["tile_regs_acquire();"],
    "commit": ["tile_regs_commit();", "tile_regs_wait();"],
    "release": ["tile_regs_release();"],
}
SFPU_HEADERS = ["ckernel.h", "ckernel_ops.h", "llk_math_eltwise_binary_sfpu_params.h"]
DEBUG_HEADERS = {
    Zone: "api/debug/kernel_profiler.hpp",
    Waypoint: "api/debug/waypoint.h",
}
RESTORE = (
    "// Puts back the vector unit's shared -1.0 constant (other kernels read it).",
    "inline void restore_constants() {",
    "    ckernel::sfpu::_restore_lconst_neg1_();",
    "    TTI_SFPNOP;",
    "}",
)


def _text(value):
    """The C++ text of a count, step or register slot: an int or a value."""
    return value.cpp() if isinstance(value, Expr) else str(int(value))


def _factor(value):
    """The C++ text of `value` as a factor of a product."""
    text = value.cpp()
    return f"({text})" if " " in text else text


def _startup(s):
    # The one-input form has no source order: the reversed one names the input twice.
    in1 = s.in0 if s.in1 is None and s.reverse else s.in1
    names = ", ".join(f"cb_{cb.name}" for cb in (s.in0, in1, s.out) if cb is not None)
    order = "<SrcOrder::Reverse>" if s.reverse else ""
    return f"compute_kernel_hw_startup{order}({names})"


def _init(s):
    if s.name in UNPACK_INITS:
        return UNPACK_INITS[s.name].format(cb=f"cb_{s.cb.name}")
    return f"{OPS[s.name].init}()"


def _format(s):
    if s.srca is None:
        return f"reconfig_data_format_srcb(cb_{s.srcb.name})"
    if s.srcb is None:
        return f"reconfig_data_format_srca(cb_{s.srca.name})"
    return f"reconfig_data_format(cb_{s.srca.name}, cb_{s.srcb.name})"


def _copy(s):
    return UNPACK_CALLS[s.kind].format(
        cb=f"cb_{s.cb.name}",
        index=s.index.cpp(),
        dst=_text(s.dst),
        count=_text(s.count),
    )


def _shape(init):
    """The trailing arguments of the matmul calls of setup `init`."""
    return ", ".join(_text(v) for v in (init.transpose, init.ct, init.rt, init.kt))


def _matmul(s):
    i = s.init
    at = f"{s.ia.cpp()}, {s.ib.cpp()}, {_text(s.dst)}"
    return f"matmul_block(cb_{i.a.name}, cb_{i.b.name}, {at}, {_shape(i)})"


def _sfpu_call(s):
    slots = f"{_text(s.a)}, {_text(s.b)}, {_text(s.out)}"
    # A generated function's scalar arguments come after its operand slots.
    scalars = getattr(s.fn, "scalars", 0)
    given = len(s.args) - scalars
    extra = "".join(
        f", {_text(a) if i < given else bits(a)}" for i, a in enumerate(s.args)
    )
    call = f"{s.fn.name}, {slots}, {VECTOR_MODES[s.mode]}{extra}"
    return f"MATH((_llk_math_eltwise_binary_sfpu_params_({call})))"


def _apply(s):
    args = [_text(slot) for slot in s.slots] + [bits(v) for v in s.values]
    return f"{OPS[s.name].call}({', '.join(args)})"


BCAST_NAMES = {"row": "rows", "col": "cols", "scalar": "scalar"}


def _tiles_init(s):
    cbs = f"cb_{s.a.name}, cb_{s.b.name}"
    if s.bcast is None:
        return f"{s.op}_init({cbs}{', true' if s.acc else ''})"
    return f"{s.op}_bcast_{BCAST_NAMES[s.bcast]}_init({cbs})"


def _tiles(s):
    i = s.init
    name = f"{i.op}_tiles" + (
        "" if i.bcast is None else f"_bcast_{BCAST_NAMES[i.bcast]}"
    )
    at = f"{s.ia.cpp()}, {s.ib.cpp()}, {_text(s.dst)}"
    return f"{name}(cb_{i.a.name}, cb_{i.b.name}, {at})"


def _reduce_kind(init):
    kind = {"sum": "SUM", "max": "MAX"}[init.kind]
    return f"<PoolType::{kind}, ReduceDim::REDUCE_{init.dim.upper()}>"


def _reduce(s):
    i = s.init
    at = f"{s.index.cpp()}, {s.scaler_index.cpp()}, {_text(s.dst)}"
    return f"reduce_tile{_reduce_kind(i)}(cb_{i.cb.name}, cb_{i.scaler.name}, {at})"


def _pack(s):
    if s.at is None:
        return f"pack_tile({_text(s.dst)}, cb_{s.cb.name})"
    return f"pack_tile<true>({_text(s.dst)}, cb_{s.cb.name}, {s.at.cpp()})"


# The C++ call of each one-line compute statement.
COMPUTE_CALLS = {
    Startup: _startup,
    Init: _init,
    SrcaFormat: lambda s: f"reconfig_data_format_srca(cb_{s.cb.name})",
    Format: _format,
    PackFormat: lambda s: f"pack_reconfig_data_format(cb_{s.cb.name})",
    PackAccumulate: lambda s: f"pack_reconfig_l1_acc({_text(s.on)})",
    Copy: _copy,
    MatmulInit: lambda s: (
        f"matmul_block_init(cb_{s.a.name}, cb_{s.b.name}, {_shape(s)})"
    ),
    Matmul: _matmul,
    TilesInit: _tiles_init,
    Tiles: _tiles,
    ReduceInit: lambda s: (
        f"reduce_init{_reduce_kind(s)}(cb_{s.cb.name}, cb_{s.scaler.name}, "
        f"cb_{s.out.name})"
    ),
    Reduce: _reduce,
    ReduceDone: lambda s: "reduce_uninit()",
    Unary: lambda s: f"{OPS[s.name].call}({_text(s.dst)})",
    Binary: lambda s: (
        f"{OPS[s.name].call}({_text(s.a)}, {_text(s.b)}, {_text(s.out)})"
    ),
    Apply: _apply,
    SfpuCall: _sfpu_call,
    Pack: _pack,
}


def cache_dir():
    """The directory generated sources are written to: KOHAKUTT_CACHE/km, else
    XDG_CACHE_HOME/kohakutt/km, else ~/.cache/kohakutt/km."""
    root = os.environ.get("KOHAKUTT_CACHE")
    if root:
        return Path(root) / "km"
    user = os.environ.get("XDG_CACHE_HOME")
    return (Path(user) if user else Path.home() / ".cache") / "kohakutt" / "km"


def _buffers(kernel):
    """The buffers `kernel` names, in order of first use."""
    out = []

    def add(cb):
        if cb is not None and cb not in out:
            out.append(cb)

    for stmt, _ in walk(kernel.body):
        if isinstance(stmt, Startup):
            for cb in (stmt.in0, stmt.in1, stmt.out):
                add(cb)
        elif isinstance(stmt, Format):
            add(stmt.srca)
            add(stmt.srcb)
        elif isinstance(stmt, (MatmulInit, Matmul, TilesInit, Tiles)):
            init = stmt if isinstance(stmt, (MatmulInit, TilesInit)) else stmt.init
            add(init.a)
            add(init.b)
        elif isinstance(stmt, (ReduceInit, Reduce)):
            init = stmt if isinstance(stmt, ReduceInit) else stmt.init
            for cb in (init.cb, init.scaler, init.out):
                add(cb)
        elif isinstance(stmt, Read):
            add(stmt.dst.slot.cb)
        elif isinstance(stmt, Inline):
            for use in stmt.uses:
                add(use.cb)
        else:
            add(getattr(stmt, "cb", None))
    return out


def _header(name):
    """The header of setup or op `name`."""
    return OPS[name].header if name in OPS else UNPACK_HEADERS[name]


def _includes(kernel):
    """The headers of `kernel`'s source."""
    stmts = [s for s, _ in walk(kernel.body)]
    debug = [
        header
        for kind, header in DEBUG_HEADERS.items()
        if any(isinstance(s, kind) for s in stmts)
    ]
    if kernel.role != "compute":
        return ["api/dataflow/dataflow_api.h", *debug, *kernel.includes]
    return _compute_includes(kernel, stmts, debug)


def _compute_includes(kernel, stmts, debug):
    """The headers of compute kernel `kernel` with statements `stmts`, the debug
    headers `debug` included."""
    out = ["api/compute/common.h", "api/compute/compute_kernel_api.h"]
    binary = "api/compute/eltwise_binary_sfpu.h"
    if any(
        isinstance(s, Binary) or (isinstance(s, Init) and _header(s.name) == binary)
        for s in stmts
    ):
        out.append(binary)
    if any(isinstance(s, SrcaFormat) for s in stmts):
        out.append("api/compute/reconfig_data_format.h")
    out.append("api/compute/tile_move_copy.h")
    more = []
    for s in stmts:
        if isinstance(s, Startup) and (s.reverse or s.in1 is None):
            more.append("api/compute/compute_kernel_hw_startup.h")
        elif isinstance(s, (Apply, Init)) and s.name in OPS:
            more.append(OPS[s.name].header)
        elif isinstance(s, Init):
            more.append(UNPACK_HEADERS[s.name])
        elif isinstance(s, Copy):
            more.append(UNPACK_HEADERS[s.kind])
        elif isinstance(s, (MatmulInit, Matmul)):
            more.append("api/compute/matmul.h")
        elif isinstance(s, (TilesInit, Tiles)):
            init = s if isinstance(s, TilesInit) else s.init
            more.append("api/compute/eltwise_binary.h")
            if init.bcast is not None:
                more.append("api/compute/bcast.h")
        elif isinstance(s, (ReduceInit, Reduce, ReduceDone)):
            more.append("api/compute/reduce.h")
        elif isinstance(s, (PackFormat, PackAccumulate)):
            more.append("api/compute/pack.h")
        elif isinstance(s, Format):
            more.append("api/compute/reconfig_data_format.h")
        elif isinstance(s, Take):
            more.append("api/compute/cb_api.h")
    out += sorted(set(more) - set(out)) + debug
    if any(isinstance(h, SfpuFn) for h in kernel.helpers) or clobbers(kernel):
        out += SFPU_HEADERS
    if clobbers(kernel):
        out.append("sfpu/ckernel_sfpu_load_config.h")
    return out + [h for h in kernel.includes if h not in out]


class _Source:
    """The lines of one kernel source; names are checked for clashes."""

    def __init__(self, program, kernel):
        self.program, self.kernel = program, kernel
        self.lines, self.names = [], set()
        self.used_slots = set()
        for stmt, _ in walk(kernel.body):
            if isinstance(stmt, Read):
                self.used_slots.add(id(stmt.dst.slot))
            if isinstance(stmt, Write) and stmt.slot is not None:
                self.used_slots.add(id(stmt.slot))
            if isinstance(stmt, Send):
                self.used_slots.add(id(stmt.src.slot))
            self.used_slots |= {id(load.slot) for load in _loads(stmt)}
        for stmt in kernel.inlines():
            self.used_slots |= {
                id(v) for v in stmt.args.values() if isinstance(v, Slot)
            }

    def name(self, text):
        if text in self.names:
            raise ValueError(
                f"kohakutt.dsl.metal: {self.program.name} {self.kernel.role} "
                f"declares {text} twice"
            )
        self.names.add(text)
        return text

    def add(self, depth, text):
        self.lines.append("    " * depth + text if text else "")

    def preamble(self, layout):
        add, previous = self.add, None
        for ref in layout.tensors:
            offset = f"{previous}.next_compile_time_args_offset()" if previous else "0"
            previous = self.name(f"{ref.name}_args")
            add(1, f"constexpr auto {previous} = TensorAccessorArgs<{offset}>();")
        buffers = [
            f"{self.name('cb_' + cb.name)} = {cb.index}" for cb in _buffers(self.kernel)
        ]
        if buffers:
            add(1, f"constexpr uint32_t {', '.join(buffers)};")
        sems = [sem for stmt in self.kernel.inlines() for sem in stmt.sems]
        sems += [s.sem for s, _ in walk(self.kernel.body) if isinstance(s, SemOp)]
        for sem in sems:
            if f"sem_{sem.name}" not in self.names:
                name = self.name(f"sem_{sem.name}")
                add(1, f"constexpr uint32_t {name} = {sem.id};")
        common = [f"{ref.name}_addr" for ref in layout.tensors] + layout.values
        common += ["cores"] if layout.core_count else []
        for i, text in enumerate(common):
            value = f"get_common_arg_val<uint32_t>({i})"
            add(1, f"const uint32_t {self.name(text)} = {value};")
        per_core = (["core"] if layout.core_index else []) + [
            a.cpp() for a in layout.per_core
        ]
        at, args = 0, ([None] if layout.core_index else []) + list(layout.per_core)
        for text, arg in zip(per_core, args):
            if hasattr(arg, "table"):
                add(1, f"const uint32_t {self.name(text)} = {at};")
                at += arg.size
                continue
            add(1, f"const uint32_t {self.name(text)} = get_arg_val<uint32_t>({at});")
            at += 1
        for var in layout.lets:
            add(1, f"const uint32_t {self.name(var.name)} = {var.definition.cpp()};")
        for ref in layout.tensors:
            args = f"{ref.name}_args, {ref.name}_addr, {ref.page_bytes}"
            add(1, f"const auto {self.name(ref.name)} = TensorAccessor({args});")

    def address(self, dst, align):
        """The C++ text of slot address `dst` for a read aligned to `align` bytes."""
        cb, parts = dst.slot.cb, [dst.slot.name]
        if dst.index.value() != 0:
            parts.append(f"{_factor(dst.index)} * {cb.page_bytes}")
        if dst.offset.value() != 0:
            parts.append(_factor(dst.offset))
        if dst.like is not None:
            parts.append(f"({_factor(dst.like)} & {align - 1})")
        return " + ".join(parts)

    def semaphore(self, s, depth):
        """The lines of semaphore statement `s`."""
        here = f"get_semaphore(sem_{s.sem.name})"
        local = f"reinterpret_cast<volatile tt_l1_ptr uint32_t*>({here})"
        if s.kind in ("inc", "fetch_add"):
            where = (
                f"get_noc_addr({here})"
                if s.at is None
                else f"get_noc_addr({s.at.x.cpp()}, {s.at.y.cpp()}, {here})"
            )
            self.add(depth, f"noc_semaphore_inc({where}, {s.value.cpp()});")
        if s.kind == "fetch_add":
            # The atomic's return lands at a fixed L1 word once it completes.
            self.add(depth, "noc_async_atomic_barrier();")
            self.add(depth, "invalidate_l1_cache();")
            word = WORD.format("MEM_NOC_ATOMIC_RET_VAL_ADDR")
            self.add(depth, f"const uint32_t {self.name(s.var.name)} = {word};")
        elif s.kind == "wait_min":
            self.add(depth, f"noc_semaphore_wait_min({local}, {s.value.cpp()});")
        elif s.kind == "wait_eq":
            self.add(depth, f"noc_semaphore_wait({local}, {s.value.cpp()});")
        elif s.kind == "set":
            self.add(depth, f"noc_semaphore_set({local}, {s.value.cpp()});")

    def body(self, stmts, depth):
        for stmt in stmts:
            self.statement(stmt, depth)

    def statement(self, s, depth):
        add = self.add
        if isinstance(s, Loop) and s.count is None:
            var = self.name(s.var.name)
            add(depth, f"for (uint32_t {var} = 0;; ++{var}) {{")
            self.body(s.body, depth + 1)
            add(depth, "}")
        elif isinstance(s, ExitWhen):
            add(depth, f"if ({s.condition.cpp()}) {{")
            add(depth + 1, "break;")
            add(depth, "}")
        elif isinstance(s, Put):
            cb = f"cb_{s.cb.name}"
            add(depth, f"cb_reserve_back({cb}, 1);")
            add(depth, f"{WORD.format(f'get_write_ptr({cb})')} = {s.value.cpp()};")
            add(depth, f"cb_push_back({cb}, 1);")
        elif isinstance(s, Take):
            cb, var = f"cb_{s.cb.name}", self.name(s.var.name)
            add(depth, f"cb_wait_front({cb}, 1);")
            if self.kernel.role == "compute":
                add(depth, f"const uint32_t {var} = read_tile_value({cb}, 0, 0);")
            else:
                add(depth, "invalidate_l1_cache();")
                word = WORD.format(f"get_read_ptr({cb})")
                add(depth, f"const uint32_t {var} = {word};")
            add(depth, f"cb_pop_front({cb}, 1);")
        elif isinstance(s, SemOp):
            self.semaphore(s, depth)
        elif isinstance(s, Send):
            address = self.address(s.src, 16)
            target = f"get_noc_addr({s.to.x.cpp()}, {s.to.y.cpp()}, {address})"
            add(depth, f"noc_async_write({address}, {target}, {s.size});")
        elif isinstance(s, Mcast):
            address, count = self.address(s.src, 16), s.to.count.cpp()
            target = _multicast(s.to, address)
            size = s.size.cpp()
            add(
                depth,
                f"noc_async_write_multicast({address}, {target}, {size}, {count});",
            )
        elif isinstance(s, McastSet):
            here = f"get_semaphore(sem_{s.sem.name})"
            target = _multicast(s.to, here)
            add(
                depth,
                f"noc_semaphore_set_multicast({here}, {target}, {s.to.count.cpp()});",
            )
        elif isinstance(s, Loop):
            var, step = self.name(s.var.name), _text(s.step)
            advance = f"++{var}" if step == "1" else f"{var} += {step}"
            start = "0" if s.start is None else s.start.cpp()
            head = f"uint32_t {var} = {start}; {var} < {s.count.cpp()}; {advance}"
            add(depth, f"for ({head}) {{")
            self.body(s.body, depth + 1)
            add(depth, "}")
        elif isinstance(s, When):
            add(depth, f"if ({s.condition.cpp()}) {{")
            self.body(s.body, depth + 1)
            if s.orelse is not None:
                add(depth, "} else {")
                self.body(s.orelse, depth + 1)
            add(depth, "}")
        elif isinstance(s, Let):
            add(depth, f"const uint32_t {self.name(s.var.name)} = {s.expr.cpp()};")
        elif isinstance(s, Assign):
            name = self.name(s.var.name) if s.declares else s.var.name
            add(depth, f"{'uint32_t ' if s.declares else ''}{name} = {s.expr.cpp()};")
        elif isinstance(s, CBOp):
            add(depth, f"{CB_CALLS[s.kind]}(cb_{s.cb.name}, {_text(s.pages)});")
            if s.slot is not None and id(s.slot) in self.used_slots:
                pointer = f"get_write_ptr(cb_{s.cb.name})"
                add(depth, f"const uint32_t {self.name(s.slot.name)} = {pointer};")
        elif isinstance(s, Read) and s.size is None:
            target = f"{s.ref.name}, {self.address(s.dst, _align(s.ref))}"
            add(depth, f"noc_async_read_page({s.page.cpp()}, {target});")
        elif isinstance(s, Read):
            source = f"{s.ref.name}.get_noc_addr({s.page.cpp()}, {s.offset.cpp()})"
            target = self.address(s.dst, _align(s.ref))
            add(depth, f"noc_async_read({source}, {target}, {s.size});")
        elif isinstance(s, Write):
            source = f"get_read_ptr(cb_{s.cb.name})" if s.slot is None else s.slot.name
            if s.index is not None and s.index.value() != 0:
                source += f" + {_factor(s.index)} * {s.cb.page_bytes}"
            add(depth, f"noc_async_write_page({s.page.cpp()}, {s.ref.name}, {source});")
        elif isinstance(s, Barrier):
            add(depth, f"{BARRIERS[s.kind]}();")
        elif isinstance(s, Registers):
            for line in REGISTERS[s.step]:
                add(depth, line)
        elif isinstance(s, Zone) and s.step == "begin":
            add(depth, "{")
            add(depth, f'DeviceZoneScopedN("{s.name}");')
        elif isinstance(s, Zone):
            add(depth, "}")
        elif isinstance(s, Waypoint):
            add(depth, f'WAYPOINT("{s.name}");')
        elif type(s) in COMPUTE_CALLS:
            add(depth, f"{COMPUTE_CALLS[type(s)](s)};")
        elif isinstance(s, Inline):
            bound = [
                (name, v.name if isinstance(v, Slot) else v.cpp())
                for name, v in s.args.items()
            ]
            bound = [(name, value) for name, value in bound if name != value]
            inner = depth + 1 if bound else depth
            if bound:
                add(depth, "{")
            for name, value in bound:
                add(inner, f"const uint32_t {name} = {value};")
            for line in textwrap.dedent(s.code).strip("\n").splitlines():
                add(inner, line.rstrip())
            if bound:
                add(depth, "}")
        else:
            raise TypeError(f"kohakutt.dsl.metal: cannot emit {s!r}")


def _multicast(rect, address):
    """The multicast NoC address of `address` on the cores of rectangle `rect`."""
    corners = ", ".join(a.cpp() for a in rect.args()[:4])
    return f"get_noc_multicast_addr({corners}, {address})"


def source(program, name):
    """The C++ source text of kernel `name` (its role, or a variant's name) of
    `program`."""
    kernel = program.kernels[name]
    role = kernel.role
    layout = kernel.layout()
    out = _Source(program, kernel)
    add = out.add
    what = ROLE_TEXT[role] if name == role else f"{ROLE_TEXT[role]} {name}"
    add(0, f"// {program.name} {what}, generated by kohakutt.dsl.metal.")
    add(0, "// Change the program's model, not this file.")
    add(0, "#include <cstdint>" if role == "compute" else "#include <stdint.h>")
    add(0, "")
    for header in _includes(kernel):
        add(0, f'#include "{header}"')
    math = sorted({h for fn in kernel.helpers for h in getattr(fn, "headers", ())})
    if math:
        # Headers of generated vector-unit functions: for the math thread only.
        add(0, "#ifdef TRISC_MATH")
        for header in math:
            add(0, f'#include "{header}"')
        add(0, "#endif")
    add(0, "")
    for name, value in kernel.defines.items():
        add(0, f"#define {name} {int(value) if not isinstance(value, str) else value}")
    helpers = [textwrap.dedent(fn.code).strip("\n") for fn in kernel.helpers]
    if clobbers(kernel):
        helpers.append("\n".join(RESTORE))
    if helpers:
        add(0, "namespace {")
        for helper in helpers:
            add(0, "")
            for line in helper.splitlines():
                add(1, line.rstrip())
        add(0, "")
        add(0, "}  // namespace")
        add(0, "")
    common = [f"{ref.name} address" for ref in layout.tensors] + layout.values
    common += ["core count"] if layout.core_count else []
    per_core = (["core index"] if layout.core_index else []) + [
        a.cpp() for a in layout.per_core
    ]
    if common:
        add(0, f"// Common arguments: {', '.join(common)}.")
    if per_core:
        add(0, f"// Per-core arguments: {', '.join(per_core)}.")
    add(0, "void kernel_main() {")
    out.preamble(layout)
    if kernel.active is not None:
        add(1, f"if (core >= {kernel.active.cpp()}) {{")
        add(2, "return;")
        add(1, "}")
    out.body(kernel.body, 1)
    if clobbers(kernel):
        # The restore follows the kernel's last vector-unit instruction.
        threads = sfpu_threads(kernel)
        if threads == {"math", "pack"}:
            add(1, "PACK((TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::PACK)));")
        thread = "PACK" if "pack" in threads else "MATH"
        add(1, f"{thread}((restore_constants()));")
    add(0, "}")
    return "\n".join(out.lines) + "\n"


def emit(program, directory=None):
    """Writes the sources of `program` under `directory` (default cache_dir());
    returns {role: path}. A file is named by its content and written once."""
    directory = Path(directory) if directory else cache_dir()
    paths = {}
    for role in program.kernels:
        text = source(program, role)
        digest = hashlib.sha1(text.encode()).hexdigest()[:16]
        path = directory / f"{program.name}_{role}_{digest}.cpp"
        if not path.exists():
            directory.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(f".{os.getpid()}.tmp")
            partial.write_text(text)
            partial.replace(path)
        paths[role] = path
    return paths
