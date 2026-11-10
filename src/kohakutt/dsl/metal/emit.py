"""The C++ emitter: one TT-Metalium source per kernel of a verified program, written
under a name made from its content.
"""

import hashlib
import os
import textwrap
from pathlib import Path

from kohakutt.dsl.metal.binding import Expr
from kohakutt.dsl.metal.body import (
    Assign,
    Barrier,
    Binary,
    CBOp,
    Copy,
    Init,
    Inline,
    Let,
    Loop,
    Pack,
    Read,
    Registers,
    SfpuCall,
    SfpuFn,
    Slot,
    SrcaFormat,
    Startup,
    Unary,
    When,
    Write,
    walk,
)
from kohakutt.dsl.metal.verify import _align, _loads, clobbers, sfpu_threads

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
}
UNARY = {"silu": "silu_tile"}
BINARY = {op: f"{op}_binary_tile" for op in ("add", "sub", "mul", "div")}
INITS = {
    "silu": "silu_tile_init",
    **{op: f"{call}_init" for op, call in BINARY.items()},
}
REGISTERS = {
    "acquire": ["tile_regs_acquire();"],
    "commit": ["tile_regs_commit();", "tile_regs_wait();"],
    "release": ["tile_regs_release();"],
}
SFPU_HEADERS = ["ckernel.h", "ckernel_ops.h", "llk_math_eltwise_binary_sfpu_params.h"]
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
        elif isinstance(stmt, Read):
            add(stmt.dst.slot.cb)
        elif isinstance(stmt, Inline):
            for use in stmt.uses:
                add(use.cb)
        else:
            add(getattr(stmt, "cb", None))
    return out


def _includes(kernel):
    """The headers of `kernel`'s source."""
    if kernel.role != "compute":
        return ["api/dataflow/dataflow_api.h", *kernel.includes]
    stmts = [s for s, _ in walk(kernel.body)]
    out = ["api/compute/common.h", "api/compute/compute_kernel_api.h"]
    if any(
        isinstance(s, Binary) or (isinstance(s, Init) and s.name in BINARY)
        for s in stmts
    ):
        out.append("api/compute/eltwise_binary_sfpu.h")
    if any(isinstance(s, SrcaFormat) for s in stmts):
        out.append("api/compute/reconfig_data_format.h")
    out.append("api/compute/tile_move_copy.h")
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
        for stmt in self.kernel.inlines():
            for sem in stmt.sems:
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
        for i, text in enumerate(per_core):
            add(1, f"const uint32_t {self.name(text)} = get_arg_val<uint32_t>({i});")
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

    def body(self, stmts, depth):
        for stmt in stmts:
            self.statement(stmt, depth)

    def statement(self, s, depth):
        add = self.add
        if isinstance(s, Loop):
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
        elif isinstance(s, Startup):
            names = ", ".join(f"cb_{cb.name}" for cb in (s.in0, s.in1, s.out))
            add(depth, f"compute_kernel_hw_startup({names});")
        elif isinstance(s, Init):
            call = (
                f"copy_init(cb_{s.cb.name})"
                if s.name == "copy"
                else f"{INITS[s.name]}()"
            )
            add(depth, f"{call};")
        elif isinstance(s, SrcaFormat):
            add(depth, f"reconfig_data_format_srca(cb_{s.cb.name});")
        elif isinstance(s, Registers):
            for line in REGISTERS[s.step]:
                add(depth, line)
        elif isinstance(s, Copy):
            add(depth, f"copy_tile(cb_{s.cb.name}, {s.index.cpp()}, {_text(s.dst)});")
        elif isinstance(s, Unary):
            add(depth, f"{UNARY[s.name]}({_text(s.dst)});")
        elif isinstance(s, Binary):
            add(depth, f"{BINARY[s.name]}({_text(s.a)}, {_text(s.b)}, {_text(s.out)});")
        elif isinstance(s, SfpuCall):
            slots = f"{_text(s.a)}, {_text(s.b)}, {_text(s.out)}"
            call = f"{s.fn.name}, {slots}, VectorMode::RC"
            add(depth, f"MATH((_llk_math_eltwise_binary_sfpu_params_({call})));")
        elif isinstance(s, Pack):
            add(depth, f"pack_tile({_text(s.dst)}, cb_{s.cb.name});")
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


def source(program, role):
    """The C++ source text of kernel `role` of `program`."""
    kernel = program.kernels[role]
    layout = kernel.layout()
    out = _Source(program, kernel)
    add = out.add
    add(0, f"// {program.name} {ROLE_TEXT[role]}, generated by kohakutt.dsl.metal.")
    add(0, "// Change the program's model, not this file.")
    add(0, "#include <cstdint>" if role == "compute" else "#include <stdint.h>")
    add(0, "")
    for header in _includes(kernel):
        add(0, f'#include "{header}"')
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
