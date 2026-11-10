"""The verifier: rules a program must pass before any source is emitted. Each rule
comes from a hazard hit on the hardware or documented by TT-Metal; RULES lists them.
"""

import math
import re
from dataclasses import dataclass

from kohakutt.dsl.metal.binding import (
    CoreCount,
    Expr,
    Load,
    Value,
    Var,
    VerifyError,
    _upper_and,
    as_expr,
)
from kohakutt.dsl.metal.body import (
    Barrier,
    Binary,
    CBOp,
    Copy,
    Inline,
    Loop,
    Pack,
    Read,
    Registers,
    SfpuCall,
    Slot,
    Unary,
    When,
    Write,
    conditional,
    exprs,
    loops,
    walk,
)

# Runtime arguments per core and in common (TT-Metal's max_runtime_args).
MAX_ARGS = 341
# Circular buffer indices of a core.
MAX_BUFFERS = 64
# The NoC a reader (dm0) and a writer (dm1) use unless pinned.
DEFAULT_NOC = {"dm0": 0, "dm1": 1}
# The shared vector-unit constant register an SFPCONFIG write may change.
SHARED_CONSTANT = "lreg11"
KINDS = ("reserve", "push", "wait", "pop")


def _fail(rule, message):
    raise VerifyError(rule, message)


def _int(value, what):
    """`value` as a static int, or a BIND-1 error naming `what`."""
    if isinstance(value, bool) or not isinstance(value, (int, Expr)):
        _fail("BIND-1", f"{what} is {value!r}, not a static integer")
    return as_expr(value)._static(what)


def _counts(value, what):
    """The values a page count, loop step or register slot can take: a static's one
    value or a bounded runtime value's choices; anything else is BIND-1."""
    choices = getattr(value, "choices", None)
    return tuple(choices) if choices else (_int(value, what),)


def _trips(path, times=1):
    """How often a statement under loops `path` runs: (static factor, symbolic
    factors). A loop over a blocked loop's remainder counts the items once."""
    number, factors = 1, []
    path = loops(path)
    blocked = {
        id(loop.count.block_of)
        for loop in path
        if isinstance(loop.count, Var) and loop.count.block_of in path
    }
    for loop in path:
        if id(loop) in blocked:
            continue
        if loop.start is not None:
            keys = (loop.count.key(), loop.start.key(), as_expr(loop.step).key())
            factors.append(("from", *keys))
            continue
        count, steps = loop.count, _counts(loop.step, "a loop step")
        step = steps[0]
        if isinstance(count, Var) and count.block_of in path:
            factors.append(("each", count.block_of.count.key()))
        elif len(steps) > 1:
            factors.append(("blocks", count.key(), loop.step.key()))
        elif count.value() is not None:
            number *= math.ceil(count.value() / step)
        elif step == 1:
            factors.append(("each", count.key()))
        else:
            factors.append(("blocks", count.key(), step))
    if isinstance(times, Expr) and times.value() is None:
        factors.append(("each", times.key()))
    else:
        number *= int(times)
    return number, tuple(sorted(factors))


def cb_ops(kernel):
    """(kind, buffer, pages, loops, runs) of every buffer call of `kernel`, inline
    stages' declared ones included; `pages` is an int or a bounded runtime value."""
    for stmt, path in walk(kernel.body):
        if isinstance(stmt, CBOp):
            _counts(stmt.pages, "a page count")
            yield stmt.kind, stmt.cb, stmt.pages, path, 1
        elif isinstance(stmt, Inline):
            for use in stmt.uses:
                for kind in KINDS:
                    if getattr(use, kind):
                        yield kind, use.cb, getattr(use, kind), path, stmt.times


def _totals(kernel, kind):
    """{buffer name: {symbolic factors: pages}} of `kind` calls over one launch."""
    out = {}
    for k, cb, pages, path, times in cb_ops(kernel):
        if k == kind:
            if conditional(path):
                _fail(
                    "CB-2",
                    f"a {kind} on buffer {cb.name} sits inside a condition; its count "
                    "per launch cannot be proved",
                )
            number, factors = _trips(path, times)
            counts = _counts(pages, "a page count")
            if len(counts) > 1:
                factors = (*factors, ("pages", pages.key()))
            else:
                number *= counts[0]
            per_cb = out.setdefault(cb.name, {})
            per_cb[factors] = per_cb.get(factors, 0) + number
    return out


def _nodes(expr, seen):
    """Every value `expr` is computed from, itself included, once each."""
    if id(expr) in seen:
        return
    seen.add(id(expr))
    yield expr
    for part in expr.parts():
        yield from _nodes(part, seen)


def _loads(stmt):
    """The L1 loads the values of statement `stmt` are computed from."""
    seen, found = set(), list(exprs(stmt))
    if isinstance(stmt, Inline):
        found = [v for v in found if not isinstance(v, Slot)]
    return [n for e in found for n in _nodes(e, seen) if isinstance(n, Load)]


def _users(program, kinds):
    """{buffer name: roles of the kernels making a call of `kinds` on it}."""
    out = {}
    for role, kernel in program.kernels.items():
        for kind, cb, *_ in cb_ops(kernel):
            if kind in kinds:
                out.setdefault(cb.name, set()).add(role)
    return out


def bind_1(program, _):
    for cb in program.cbs:
        if _int(cb.pages, f"the page count of buffer {cb.name}") < 1:
            _fail("BIND-1", f"buffer {cb.name} has no pages")
    for s in program.scratches:
        _int(s.bytes, f"the size of scratch {s.name}")
    for sem in program.semaphores:
        _int(sem.initial, f"the initial value of semaphore {sem.name}")
    for role, kernel in program.kernels.items():
        for name, value in kernel.defines.items():
            if isinstance(value, Expr):
                value._static(f"define {name} of {role}")
        for stmt, _path in walk(kernel.body):
            strided = isinstance(stmt, Loop) and stmt.start is not None
            if strided and not isinstance(stmt.step, CoreCount):
                _counts(stmt.step, f"a loop step of {role}")
            if isinstance(stmt, Loop) and not strided:
                _counts(stmt.step, f"a loop step of {role}")
            if isinstance(stmt, CBOp):
                _counts(stmt.pages, f"a {stmt.kind} count on buffer {stmt.cb.name}")


def bind_2(program, _):
    for name, values, _count in program.per_core_args:
        if values is not None and not callable(values):
            _fail(
                "BIND-2",
                f"per-core argument {name} takes its value from {values!r}; a per-core "
                "value is a function of the core index, or derived on the device",
            )


def arg_1(program, _):
    for name, _values, count in program.per_core_args:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            _fail(
                "ARG-1",
                f"per-core argument {name} has {count!r} entries; argument counts "
                "are fixed per kernel class",
            )
    for role, kernel in program.kernels.items():
        layout = kernel.layout()
        if layout.common > MAX_ARGS or layout.per_core_count > MAX_ARGS:
            _fail("ARG-1", f"{role} takes more than {MAX_ARGS} runtime arguments")


def cb_4(program, _):
    seen = {}
    for b in (*program.cbs, *program.scratches):
        if not 0 <= b.index < MAX_BUFFERS:
            _fail("CB-4", f"buffer {b.name} has index {b.index}")
        if b.index in seen:
            _fail("CB-4", f"buffers {seen[b.index]} and {b.name} share index {b.index}")
        seen[b.index] = b.name
    producers = _users(program, ("reserve", "push"))
    consumers = _users(program, ("wait", "pop"))
    for what, users in (("filled", producers), ("drained", consumers)):
        for name, roles in users.items():
            if len(roles) > 1:
                _fail("CB-4", f"buffer {name} is {what} by {sorted(roles)}")
    for role, kernel in program.kernels.items():
        for stmt, _path in walk(kernel.body):
            name = getattr(getattr(stmt, "cb", None), "name", None)
            held = isinstance(stmt, Write) and stmt.slot is not None
            if held and role not in producers.get(name, ()):
                _fail("CB-4", f"{role} writes from buffer {name} unreserved")
            waited = isinstance(stmt, (Copy, Write)) and not held
            if waited and role not in consumers.get(name, ()):
                _fail("CB-4", f"{role} reads buffer {name} without a wait")
            if isinstance(stmt, Pack) and role not in producers.get(name, ()):
                _fail("CB-4", f"{role} packs into buffer {name} unreserved")


def cb_1(program, _):
    for role, kernel in program.kernels.items():
        for kind, cb, pages, path, times in cb_ops(kernel):
            total = _int(cb.pages, "pages")
            recurring = bool(loops(path)) or not (isinstance(times, int) and times == 1)
            for n in _counts(pages, "a page count"):
                if n < 1 or n > total or (recurring and total % n):
                    _fail(
                        "CB-1",
                        f"{role} makes a {kind} of {n} on buffer {cb.name} of "
                        f"{total} pages; a recurring count must divide the buffer",
                    )


def _align(ref):
    """The byte alignment NoC reads of `ref` keep: 64 from DRAM, 16 from L1."""
    return 64 if ref.kind == "dram" else 16


def _held(slot):
    """Bytes of the smallest reserve `slot` can be."""
    return _counts(slot.pages, "pages")[0] * slot.cb.page_bytes


def _slot_end(role, dst, size, align):
    """Checks that `size` bytes written at slot address `dst` stay in its reserve."""
    cb = dst.slot.cb
    pages = _counts(dst.slot.pages, "pages")
    held = _held(dst.slot)
    top = dst.index.upper()
    # A loop variable counting up to the reserve's own runtime count stays inside it.
    if len(pages) > 1 and getattr(dst.index, "count", None) is dst.slot.pages:
        top = 0
    offset = dst.offset.upper()
    if dst.like is not None and offset is not None:
        offset += _upper_and(dst.like, dst.like.upper(), align - 1)
    if top is None or offset is None:
        _fail("CB-3", f"{role} writes buffer {cb.name} at an unbounded address")
    end = top * cb.page_bytes + offset + size
    if end > held:
        _fail(
            "CB-3",
            f"{role} writes up to byte {end} of a {held}-byte reserve on buffer "
            f"{cb.name}",
        )


def cb_3(program, _):
    for role, kernel in program.kernels.items():
        for stmt, _path in walk(kernel.body):
            if isinstance(stmt, Read):
                size = stmt.size or stmt.ref.page_bytes
                _slot_end(role, stmt.dst, size, _align(stmt.ref))
            held = isinstance(stmt, Write) and stmt.slot is not None
            if held and stmt.ref.page_bytes > _held(stmt.slot):
                _fail(
                    "CB-3",
                    f"{role} writes a {stmt.ref.page_bytes}-byte page from a "
                    f"{_held(stmt.slot)}-byte reserve on buffer {stmt.cb.name}",
                )
            for load in _loads(stmt):
                top = load.word.upper()
                if top is None or (top + 1) * 4 > _held(load.slot):
                    _fail(
                        "CB-3",
                        f"{role} loads a word of buffer {load.slot.cb.name} past its "
                        "reserve, or at an unbounded index",
                    )
            if isinstance(stmt, (Copy, Write)) and stmt.index is not None:
                top = stmt.index.upper()
                if top is None or top >= _int(stmt.cb.pages, "pages"):
                    _fail("CB-3", f"{role} reads past buffer {stmt.cb.name}")
            if isinstance(stmt, Inline):
                for use in stmt.uses:
                    held = (use.reserve or _int(use.cb.pages, "pages")) * (
                        use.cb.page_bytes
                    )
                    if use.written > held:
                        _fail(
                            "CB-3",
                            f"inline code of {role} writes {use.written} bytes into "
                            f"{held} reserved on buffer {use.cb.name}",
                        )


def cb_2(program, _):
    pushes, pops = {}, {}
    for kernel in program.kernels.values():
        pushes.update(_totals(kernel, "push"))
        pops.update(_totals(kernel, "pop"))
    for name in pushes.keys() | pops.keys():
        if pushes.get(name) != pops.get(name):
            _fail(
                "CB-2",
                f"buffer {name}: pushes {pushes.get(name)} and pops {pops.get(name)} "
                "per launch differ",
            )


def cb_5(program, _):
    kernel = program.kernels.get("compute")
    if kernel is None:
        return
    packed = {s.cb.name for s, _ in walk(kernel.body) if isinstance(s, Pack)}
    for stmt, _path in walk(kernel.body):
        if isinstance(stmt, Copy) and stmt.cb.name in packed:
            _fail(
                "CB-5",
                f"compute packs into buffer {stmt.cb.name} and copies from it; the "
                "unpacker may read the slot before the packer wrote it",
            )


def noc_1(program, _):
    dm0, dm1 = program.kernels.get("dm0"), program.kernels.get("dm1")
    if dm0 is None or dm1 is None or not dm0.cores.overlaps(dm1.cores):
        return
    nocs = [
        DEFAULT_NOC[k.role] if k.noc is None else _int(k.noc, "a NoC")
        for k in (dm0, dm1)
    ]
    if nocs[0] == nocs[1] or not set(nocs) <= {0, 1}:
        _fail("NOC-1", f"dm0 and dm1 of a core use NoC {nocs[0]} and NoC {nocs[1]}")


def noc_2(program, _):
    for role, kernel in program.kernels.items():
        for stmt, _path in walk(kernel.body):
            if not isinstance(stmt, Read):
                continue
            align = _align(stmt.ref)
            cb, src, dst = stmt.dst.slot.cb, stmt.offset, stmt.dst
            aligned = cb.page_bytes % align == 0 and stmt.ref.page_bytes % align == 0
            if dst.like is not None:
                # The destination was built from the source offset modulo `align`.
                congruent = (
                    dst.like.key() == src.key() and dst.offset.multiple() % align == 0
                )
            elif src.value() is not None and dst.offset.value() is not None:
                congruent = (src.value() - dst.offset.value()) % align == 0
            else:
                congruent = (
                    src.multiple() % align == 0 and dst.offset.multiple() % align == 0
                )
            if not aligned or not congruent:
                _fail(
                    "NOC-2",
                    f"{role} reads {stmt.ref.name} at byte {src.cpp()} into byte "
                    f"{dst.offset.cpp()} of buffer {cb.name}; source and destination "
                    f"must be congruent modulo {align}",
                )


class _Flight:
    """NoC work a kernel has issued and not waited for: buffers with reads landing
    (`reading`), waited pages being written (`writing`), reserved slots being
    written (`sending`), and whether a write is unacknowledged."""

    def __init__(self):
        self.reading, self.writing, self.sending = set(), set(), set()
        self.unacked = False

    def copy(self):
        out = _Flight()
        out.reading, out.writing = set(self.reading), set(self.writing)
        out.sending, out.unacked = set(self.sending), self.unacked
        return out

    def join(self, other):
        """What is in flight when either this or `other` happened."""
        self.reading |= other.reading
        self.writing |= other.writing
        self.sending |= other.sending
        self.unacked = self.unacked or other.unacked


def _noc_3(role, stmts, f):
    """Checks statements `stmts` of `role` in order, updating flight state `f`."""
    for stmt in stmts:
        for load in _loads(stmt):
            if load.slot.cb.name in f.reading:
                _fail(
                    "NOC-3",
                    f"{role} loads from buffer {load.slot.cb.name} before its reads "
                    "land",
                )
        if isinstance(stmt, Loop):
            # Twice: the second pass meets what the first left in flight.
            _noc_3(role, stmt.body, f)
            _noc_3(role, stmt.body, f)
        elif isinstance(stmt, When):
            skipped = f.copy()
            _noc_3(role, stmt.body, f)
            f.join(skipped)
        elif isinstance(stmt, Read):
            name = stmt.dst.slot.cb.name
            if name in f.sending:
                _fail("NOC-3", f"{role} reads into buffer {name} while it is written")
            f.reading.add(name)
        elif isinstance(stmt, Write):
            name = stmt.cb.name
            if stmt.slot is not None and name in f.reading:
                _fail("NOC-3", f"{role} writes buffer {name} before its reads land")
            (f.writing if stmt.slot is None else f.sending).add(name)
            f.unacked = True
        elif isinstance(stmt, Barrier):
            if stmt.kind == "reads_done":
                f.reading.clear()
            else:
                f.writing.clear()
                f.sending.clear()
                f.unacked = f.unacked and stmt.kind != "writes_done"
        elif isinstance(stmt, CBOp):
            name = stmt.cb.name
            if stmt.kind == "push" and name in f.reading:
                _fail("NOC-3", f"{role} pushes buffer {name} before its reads land")
            if stmt.kind == "pop" and name in f.writing:
                _fail("NOC-3", f"{role} pops buffer {name} before its write left")
        elif isinstance(stmt, Inline):
            for use in stmt.uses:
                name = use.cb.name
                if name in f.reading or (use.written and name in f.sending):
                    _fail(
                        "NOC-3",
                        f"inline code of {role} touches buffer {name} with NoC work "
                        "on it in flight",
                    )


def noc_3(program, _):
    for role, kernel in program.kernels.items():
        flight = _Flight()
        _noc_3(role, kernel.body, flight)
        if flight.reading:
            _fail("NOC-3", f"{role} ends with reads not waited for")
        if flight.unacked:
            _fail("NOC-3", f"{role} ends without waiting for its writes")


def data_1(program, _):
    def refuse(role, value, what):
        if isinstance(value, Expr) and value.raw_data():
            _fail(
                "DATA-1",
                f"{role} uses {value.cpp()}, loaded from L1, as {what}; declare its "
                "range with .bounded(low, high) where it only selects what is read",
            )

    for role, kernel in program.kernels.items():
        for stmt, _path in walk(kernel.body):
            for load in _loads(stmt):
                refuse(role, load.word, "the index of a load")
            if isinstance(stmt, Loop):
                for value in (stmt.count, stmt.step, stmt.start):
                    refuse(role, value, "a loop bound")
            elif isinstance(stmt, CBOp):
                refuse(role, stmt.pages, "a page count")
            elif isinstance(stmt, Read):
                dst = stmt.dst
                for value in (stmt.page, stmt.offset, dst.index, dst.offset, dst.like):
                    refuse(role, value, "a NoC page or address")
            elif isinstance(stmt, Write):
                refuse(role, stmt.page, "a NoC page")
                refuse(role, stmt.index, "a buffer page index")
            elif isinstance(stmt, Inline):
                refuse(role, stmt.times, "a run count")
            elif isinstance(stmt, Copy):
                refuse(role, stmt.index, "a buffer page index")
            for slot in _slots(stmt):
                refuse(role, slot, "a register slot")


def loop_1(program, _):
    for role, kernel in program.kernels.items():
        for stmt, _path in walk(kernel.body):
            plain = isinstance(stmt, Loop) and stmt.start is None
            if not plain or stmt.remainder is not None:
                continue
            step, count = stmt.step, stmt.count
            if isinstance(step, Value) and step.choices:
                whole = isinstance(count, Value) and count.multiple_of == step.name
                steps = step.choices
            else:
                whole, steps = False, (_int(step, "a loop step"),)
            if not whole and any(count.multiple() % s for s in steps):
                _fail(
                    "LOOP-1",
                    f"{role} loops over {count.cpp()} in steps of {_text(step)}; "
                    "declare the count a multiple of the step (p.runtime(name, "
                    "multiple_of=...)) or take the remainder with k.blocks",
                )


def _text(value):
    return value.cpp() if isinstance(value, Expr) else str(value)


def sem_1(program, _):
    sems = program.semaphores
    for i, a in enumerate(sems):
        _int(a.id, f"the id of semaphore {a.name}")
        for b in sems[i + 1 :]:
            if a.id == b.id and a.cores.overlaps(b.cores):
                _fail("SEM-1", f"semaphores {a.name} and {b.name} share id {a.id}")
    for role, kernel in program.kernels.items():
        for stmt in kernel.inlines():
            for sem in stmt.sems:
                if sem not in sems or not sem.cores.covers(kernel.cores):
                    _fail(
                        "SEM-1",
                        f"{role} uses semaphore {sem.name}, which does not exist on "
                        "every core the kernel runs on",
                    )


def _inline_code(kernel):
    """(code, declared clobbers) of every inline stage and helper of `kernel`."""
    parts = [(s.code, s.clobbers) for s in kernel.inlines()]
    return parts + [(fn.code, fn.clobbers) for fn in kernel.helpers]


def clobbers(kernel):
    """The shared vector-unit constants the inline code of `kernel` declares."""
    return sorted({c for _, declared in _inline_code(kernel) for c in declared})


def sfpu_threads(kernel):
    """The compute threads ("math", "pack") that issue vector-unit work in `kernel`:
    typed vector-unit calls run on the math thread, inline stages where declared."""
    typed = (Unary, Binary, SfpuCall)
    threads = (
        {"math"} if any(isinstance(s, typed) for s, _ in walk(kernel.body)) else set()
    )
    return threads | {s.sfpu for s in kernel.inlines() if s.sfpu}


def sfpu_1(program, _):
    for role, kernel in program.kernels.items():
        for stmt in kernel.inlines():
            if stmt.sfpu not in (None, "math", "pack") or (
                stmt.sfpu and role != "compute"
            ):
                _fail("SFPU-1", f"{role} declares vector-unit work on {stmt.sfpu!r}")
        for code, declared in _inline_code(kernel):
            if "SFPCONFIG" in code and SHARED_CONSTANT not in declared:
                _fail(
                    "SFPU-1",
                    f"inline code of {role} writes a vector-unit constant; declare "
                    f'clobbers=["{SHARED_CONSTANT}"] so it is restored at exit',
                )
            if declared and role != "compute":
                _fail("SFPU-1", f"{role} declares vector-unit constants; it has none")
            unknown = set(declared) - {SHARED_CONSTANT}
            if unknown:
                _fail("SFPU-1", f"{role} declares unknown constants {sorted(unknown)}")


def _slots(stmt):
    """The register slots statement `stmt` names."""
    if isinstance(stmt, (Copy, Unary, Pack)):
        return [stmt.dst]
    if isinstance(stmt, (Binary, SfpuCall)):
        return [stmt.a, stmt.b, stmt.out]
    return []


def dst_1(program, _):
    kernel = program.kernels.get("compute")
    if kernel is None:
        return
    capacity = 4 if kernel.options.get("fp32_dest", True) else 8
    for stmt, _path in walk(kernel.body):
        for slot in _slots(stmt):
            top = slot.upper() if isinstance(slot, Expr) else slot
            if isinstance(top, bool) or not isinstance(top, int):
                _fail(
                    "DST-1", f"compute names register slot {slot!r}, which is unbounded"
                )
            if not 0 <= top < capacity:
                slot = slot.cpp() if isinstance(slot, Expr) else slot
                _fail(
                    "DST-1",
                    f"compute names register slot {slot} (up to {top}); the registers "
                    f"hold {capacity} tiles in this mode",
                )


def reg_1(program, _):
    kernel = program.kernels.get("compute")
    if kernel is None:
        return
    state, scope = None, None
    for stmt, path in walk(kernel.body):
        if isinstance(stmt, Registers):
            wanted = {"acquire": None, "commit": "math", "release": "pack"}[stmt.step]
            if state != wanted:
                _fail("REG-1", f"compute: register {stmt.step} out of order")
            loops = [id(loop) for loop in path]
            if stmt.step != "acquire" and loops != scope:
                _fail(
                    "REG-1", f"compute: register {stmt.step} inside a loop of its scope"
                )
            scope = loops
            state = {"acquire": "math", "commit": "pack", "release": None}[stmt.step]
        elif isinstance(stmt, (Copy, Unary, Binary, SfpuCall)) and state != "math":
            _fail("REG-1", "compute: a math call outside acquire .. commit")
        elif isinstance(stmt, Pack) and state != "pack":
            _fail("REG-1", "compute: a pack outside commit .. release")
    if state is not None:
        _fail("REG-1", "compute ends inside a register scope")


_BUFFER_CALL = re.compile(
    r"\b(?:cb_\w+|get_(?:write|read)_ptr|get_tile_size)\(\s*(\w+)"
)
_SEMAPHORE_CALL = re.compile(r"\bget_semaphore\(\s*(\w+)")


def inl_1(program, _):
    for role, kernel in program.kernels.items():
        for stmt in kernel.inlines():
            names = {f"cb_{u.cb.name}" for u in stmt.uses}
            names |= {str(u.cb.index) for u in stmt.uses}
            for found in _BUFFER_CALL.findall(stmt.code):
                if found not in names:
                    _fail(
                        "INL-1",
                        f"inline code of {role} uses buffer {found}, which it does "
                        "not declare",
                    )
            sems = {f"sem_{s.name}" for s in stmt.sems} | {str(s.id) for s in stmt.sems}
            for found in _SEMAPHORE_CALL.findall(stmt.code):
                if found not in sems:
                    _fail(
                        "INL-1",
                        f"inline code of {role} uses semaphore {found}, which it does "
                        "not declare",
                    )


def l1_1(program, l1_budget):
    if l1_budget is not None and program.footprint() > l1_budget:
        _fail(
            "L1-1",
            f"buffers and scratch take {program.footprint()} bytes per core, the "
            f"budget is {l1_budget}",
        )


@dataclass(frozen=True)
class Rule:
    """A verifier rule: what it checks, the hazard it comes from, and its check
    (None: a reserved slot, not checked yet)."""

    id: str
    checks: str
    origin: str
    check: object = None


RULES = (
    Rule(
        "BIND-1",
        "no runtime value in a compile-time slot (pages, defines, steps, Python "
        "control flow)",
        "a new shape never compiles",
        bind_1,
    ),
    Rule(
        "BIND-2",
        "a per-core argument is a function of the core index",
        "the launch patches addresses and common values only",
        bind_2,
    ),
    Rule(
        "DATA-1",
        "a value loaded from L1 is no count, loop bound or register slot, and selects "
        "a NoC page or address only through a declared range; the range is the "
        "launch's contract, not checked on the device",
        "an index tensor's values are known on the device only; an index outside "
        "its table reads another tensor's memory",
        data_1,
    ),
    Rule(
        "ARG-1",
        "runtime argument counts are fixed per kernel class and within the limit",
        "programs differing in per-core argument length share a cache entry",
        arg_1,
    ),
    Rule(
        "CB-4",
        "one filling and one draining kernel per buffer; reads follow a wait, packs "
        "a reserve",
        "only the unpacker waits and pops, only the packer reserves and pushes",
        cb_4,
    ),
    Rule(
        "CB-1",
        "every recurring reserve / push / wait / pop count divides the buffer",
        "a non-dividing push wraps differently on the two sides: wrong data",
        cb_1,
    ),
    Rule(
        "CB-3",
        "a kernel writes no further than the pages it reserved",
        "a buffer smaller than what its kernel wrote overran L1 and hung the card",
        cb_3,
    ),
    Rule(
        "CB-2",
        "pushes equal pops on every buffer over a launch",
        "unpaired pushes and pops stall the consumer or the producer",
        cb_2,
    ),
    Rule(
        "LOOP-1",
        "a loop in steps of n runs over a count that is a multiple of n",
        "the last step of a stepped loop reads and pushes whole steps past the count",
        loop_1,
    ),
    Rule(
        "CB-5",
        "a page the packer wrote is not copied back without a pushed token",
        "nothing orders the packer and the unpacker on a reserved slot",
        cb_5,
    ),
    Rule(
        "NOC-1",
        "the reader and the writer of a core use different NoCs",
        "two processors on one NoC share its command buffers and counters: a hang",
        noc_1,
    ),
    Rule(
        "NOC-2",
        "a DRAM read keeps source and destination congruent modulo 64 bytes (L1: 16)",
        "misaligned sub-page reads return another row's data",
        noc_2,
    ),
    Rule(
        "NOC-3",
        "a push, a load or a write follows the reads into its pages, a pop or a new "
        "read the flush of its write, and a writing kernel ends with its writes "
        "acknowledged",
        "a kernel ending with pending NoC transactions races the next one",
        noc_3,
    ),
    Rule(
        "SEM-1",
        "a semaphore exists on every core that uses it; ids are distinct per core",
        "a missing semaphore reads another program's word",
        sem_1,
    ),
    Rule(
        "SFPU-1",
        "a kernel that writes the shared vector-unit constant declares it (and it is "
        "restored at exit)",
        "a changed -1.0 constant made later kernels on the core return infinities",
        sfpu_1,
    ),
    Rule(
        "SFPU-2",
        "inline vector-unit code: a multiply-add result is not read by the next "
        "instruction through a form the stall logic misses",
        "an unspaced multiply-add fed a stale value into a clamp",
    ),
    Rule(
        "DST-1",
        "register slots stay within the capacity of the mode (4 fp32 or 8 tiles)",
        "the destination register holds 8 tiles at 16 bits, 4 at fp32",
        dst_1,
    ),
    Rule(
        "REG-1",
        "math calls sit between acquire and commit, packs between commit and release",
        "the math and pack threads hand the registers over at commit",
        reg_1,
    ),
    Rule(
        "INL-1",
        "inline code declares every buffer and semaphore its text names",
        "the verifier sees inline code only through its declaration",
        inl_1,
    ),
    Rule(
        "L1-1",
        "buffers and scratch fit the L1 budget given",
        "a program's buffers must not clash with L1 tensors",
        l1_1,
    ),
)


def verify(program, l1_budget=None):
    """Raises VerifyError for the first rule of RULES `program` breaks."""
    for rule in RULES:
        if rule.check is None:
            continue
        rule.check(program, l1_budget)
