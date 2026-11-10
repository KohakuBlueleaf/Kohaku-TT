"""The text form of a program's model (`dump`) and the hash that names its kernel
class (`fingerprint`): two programs with equal fingerprints build the same kernels.
"""

import dataclasses
import hashlib
import textwrap

from kohakutt.dsl.metal.binding import Bounded, Expr
from kohakutt.dsl.metal.body import CppFn, Inline, Loop, Slot, SlotAt, When
from kohakutt.dsl.metal.emit import source


def _show(value):
    """The text of one field of a statement or declaration."""
    if isinstance(value, str):
        return value
    if isinstance(value, Bounded):
        return f"{value.cpp()} in [{value.low.cpp()}, {value.high.cpp()})"
    if isinstance(value, Expr):
        return value.cpp()
    if isinstance(value, SlotAt):
        text = f"{value.slot.name}[{value.index.cpp()}] + {value.offset.cpp()}"
        return text if value.like is None else f"{text} + like({value.like.cpp()})"
    if isinstance(value, (Slot, CppFn)):
        return value.name
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_show(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_show(v)}" for k, v in value.items()) + "}"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _fields(value)
    name = getattr(value, "name", None)
    return name if isinstance(name, str) else repr(value)


def _fields(stmt, skip=()):
    """`Kind(field=value, ...)` of a dataclass instance, unset fields left out."""
    shown = [
        f"{f.name}={_show(getattr(stmt, f.name))}"
        for f in dataclasses.fields(stmt)
        if f.name not in skip and getattr(stmt, f.name) not in (None, (), {})
    ]
    return f"{type(stmt).__name__}({', '.join(shown)})"


def _body(stmts, depth, out):
    pad = "    " * depth
    for stmt in stmts:
        if isinstance(stmt, (Loop, When)):
            out.append(f"{pad}{_fields(stmt, skip=('body',))}:")
            _body(stmt.body, depth + 1, out)
        elif isinstance(stmt, Inline):
            out.append(f"{pad}{_fields(stmt, skip=('code',))}:")
            code = textwrap.dedent(stmt.code).strip("\n")
            out += [f"{pad}    | {line.rstrip()}" for line in code.splitlines()]
        else:
            out.append(f"{pad}{_fields(stmt)}")


def _core_set(cores):
    return "all" if cores.first is None else f"rows {cores.first}..{cores.last}"


def _per_core(program, name, values, count):
    if callable(values):
        values = [values(core) for core in range(program.machine.cores)]
    return f"  per-core {name} x{count}: {_show(values)}"


def dump(program):
    """The model of `program` as text: its declarations, then every kernel's
    statements as a tree. For reading and comparing programs, not for parsing."""
    m = program.machine
    out = [f"program {program.name} on {m.cols} x {m.rows} cores"]
    out.append(f"  size classes: {_show(list(program.size_classes))}")
    for name, value in program.statics.items():
        out.append(f"  static {name} = {value!r}")
    for ref in program.tensors:
        role = "output" if ref.is_output else "input"
        out.append(f"  {role} {ref.name}: {ref.dtype}, {ref.kind}")
    for name in program.values:
        sized = program.sized_values.get(name)
        if sized is not None:
            how = "itself, up to" if sized.exact else "rounded up to one of"
            limit = sized.choices[-1] if sized.exact else _show(list(sized.choices))
            out.append(f"  sized {name} = {sized.of.cpp()}: {how} {limit}")
            continue
        notes = [
            f"{label} {given[name]}"
            for label, given in (
                ("one of", program.choices),
                ("multiple of", program.multiples),
            )
            if name in given
        ]
        out.append(f"  runtime {name}" + (": " + ", ".join(notes) if notes else ""))
    if program.split_of is not None:
        how = "dealt in turn" if program.strided else "split"
        total, least = program.split_of
        out.append(f"  {how} over the cores: {total}, at least {least} per core")
    if program.layout_of is not None:
        out.append(f"  layout: {program.layout_of.cpp()} cores have work")
    for name, values, count in program.per_core_args:
        out.append(_per_core(program, name, values, count))
    out += [
        f"  cb {cb.index} {cb.name}: {_show(cb.pages)} pages of {cb.dtype}"
        for cb in program.cbs
    ]
    for s in program.scratches:
        text = f"{_show(s.bytes)} bytes, align {s.align}"
        out.append(f"  scratch {s.index} {s.name}: {text}")
    for sem in program.semaphores:
        on = _core_set(sem.cores)
        out.append(
            f"  semaphore {sem.id} {sem.name}: initial {_show(sem.initial)}, {on}"
        )
    for role, kernel in program.kernels.items():
        noc = "" if kernel.noc is None else f", NoC {kernel.noc}"
        out.append(f"kernel {role} on {_core_set(kernel.cores)}{noc}")
        for name, value in kernel.options.items():
            out.append(f"  option {name} = {_show(value)}")
        for name, value in kernel.defines.items():
            out.append(f"  define {name} = {_show(value)}")
        out += [f"  include {header}" for header in kernel.includes]
        out += [f"  helper {fn.name}" for fn in kernel.helpers]
        _body(kernel.body, 1, out)
    return "\n".join(out) + "\n"


def fingerprint(program):
    """A hash of everything the kernels and the description of `program` are built
    from: its model text and its generated sources."""
    text = dump(program) + "".join(source(program, role) for role in program.kernels)
    return hashlib.sha1(text.encode()).hexdigest()[:16]
