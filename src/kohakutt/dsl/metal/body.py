"""Typed kernel bodies: the statements a data-movement or compute kernel is made of,
and the builders (`DM`, `Compute`) that record them while the program is traced.
"""

import contextlib
from dataclasses import dataclass, field

from kohakutt.dsl.metal.binding import (
    COMPARISONS,
    Expr,
    Load,
    Mutable,
    Op,
    Var,
    as_expr,
    minimum,
)


@dataclass
class Loop:
    """`for (var = start; var < count; var += step)`; `remainder` is a blocked loop's
    n; `start` is None for 0."""

    var: Var
    count: Expr
    step: object
    body: list = field(default_factory=list)
    remainder: Var | None = None
    start: Expr | None = None


@dataclass
class When:
    """`if (condition) { body }`: the body may or may not run."""

    condition: Expr
    body: list = field(default_factory=list)


@dataclass
class Let:
    """`const uint32_t var = expr;`"""

    var: Var
    expr: Expr


@dataclass
class Assign:
    """`var = expr;` on a mutable variable; `declares`: `uint32_t var = expr;`."""

    var: Mutable
    expr: Expr
    declares: bool = False


@dataclass
class CBOp:
    """reserve / push / wait / pop of `pages` on `cb`; `slot` is a reserve's pages."""

    kind: str
    cb: object
    pages: object
    slot: object = None


@dataclass
class Read:
    """NoC read of a page (or of `size` bytes at byte `offset` in it) into `dst`."""

    ref: object
    page: Expr
    dst: object
    offset: Expr = None
    size: int | None = None


@dataclass
class Write:
    """NoC write to page `page` of `ref` of waited page `index` of `cb`, or of the
    first page reserved as `slot`."""

    cb: object
    ref: object
    page: Expr
    index: Expr = None
    slot: object = None


@dataclass
class Barrier:
    """Completion of NoC work: "reads_done", "flushed" or "writes_done"."""

    kind: str


@dataclass
class Startup:
    """Unpacker / packer setup for the kernel's first formats."""

    in0: object
    in1: object
    out: object


@dataclass
class Init:
    """Setup call of op `name` ("copy" takes its buffer)."""

    name: str
    cb: object = None


@dataclass
class SrcaFormat:
    """Switches the first source register to the format of `cb`."""

    cb: object


@dataclass
class Registers:
    """Tile register scope step: "acquire", "commit" (and wait) or "release"."""

    step: str


@dataclass
class Copy:
    """Tile `index` of `cb` into register slot `dst`."""

    cb: object
    index: Expr
    dst: int


@dataclass
class Unary:
    """Vector-unit op `name` on register slot `dst` in place."""

    name: str
    dst: int


@dataclass
class Binary:
    """Vector-unit op `name`: slot `out` = slot `a` op slot `b`."""

    name: str
    a: int
    b: int
    out: int


@dataclass
class SfpuCall:
    """Inline vector-unit function `fn` over slots `a`, `b` into `out`, every face."""

    fn: object
    a: int
    b: int
    out: int


@dataclass
class Pack:
    """Register slot `dst` into the reserved page of `cb`."""

    dst: int
    cb: object


@dataclass
class CBUse:
    """What one run of an inline stage does to `cb`: pages per call and the bytes it
    writes from the reserved pointer."""

    cb: object
    reserve: int = 0
    push: int = 0
    wait: int = 0
    pop: int = 0
    written: int = 0


@dataclass
class Inline:
    """Inline C++ with its declared effects; `times` runs of them per execution.
    `sfpu` names the compute thread ("math" or "pack") its vector-unit work runs on."""

    code: str
    uses: tuple = ()
    sems: tuple = ()
    clobbers: tuple = ()
    times: object = 1
    sfpu: str | None = None
    args: dict = field(default_factory=dict)


class CppFn:
    """A C++ helper `name` (its whole definition in `code`) emitted above the kernel
    for inline stages to call."""

    clobbers = ()

    def __init__(self, name, code):
        self.name, self.code = name, code


class SfpuFn(CppFn):
    """An inline vector-unit function `void name(uint32_t a, uint32_t b, uint32_t out)`
    run per face; `clobbers` names the shared constant registers it writes."""

    def __init__(self, name, code, clobbers=()):
        super().__init__(name, code)
        self.clobbers = tuple(clobbers)


class Slot:
    """The pages a reserve returned; `slot[i]` is page i, `slot.at(b)` byte b."""

    def __init__(self, cb, pages, name):
        self.cb, self.pages, self.name = cb, pages, name

    def __getitem__(self, index):
        return SlotAt(self, as_expr(index), as_expr(0))

    def at(self, offset, like=None):
        """The address `offset` bytes into the first page; with `like` (the source
        offset of a read), moved up by `like` modulo the read's alignment so that
        source and destination are congruent."""
        like = None if like is None else as_expr(like)
        return SlotAt(self, as_expr(0), as_expr(offset), like)

    def load32(self, word):
        """The 32-bit word `word` of the first page, read from L1 on the device: a
        data value."""
        return Load(self, word)


@dataclass
class SlotAt:
    """Byte `offset` (plus `like` modulo the alignment) of page `index` of a slot."""

    slot: Slot
    index: Expr
    offset: Expr
    like: Expr | None = None


class Builder:
    """Records the statements of one kernel."""

    def __init__(self, program, kernel):
        self.program, self.kernel = program, kernel
        self._stack, self._names = [kernel.body], {}

    def _add(self, stmt):
        self._stack[-1].append(stmt)
        return stmt

    def _fresh(self, base):
        count = self._names.get(base, 0) + 1
        self._names[base] = count
        return base if count == 1 else f"{base}{count}"

    def _loop(self, loop):
        self._add(loop)
        self._stack.append(loop.body)
        try:
            yield
        finally:
            self._stack.pop()

    def range(self, count, step=1, name="i", start=None):
        """A device loop over `start` (default 0) .. count - 1 in steps of `step`;
        yields its variable."""
        count = as_expr(count)
        top = count.upper()
        var = Var(self._fresh(name), upper=None if top is None else top - 1)
        var.count = count
        if start is None:
            var.step = step if isinstance(step, int) else None
        else:
            start = as_expr(start)
        for _ in self._loop(Loop(var, count, step, start=start)):
            yield var

    def var(self, name, initial):
        """A 32-bit device variable starting at `initial`; `k.set(v, value)` assigns
        it."""
        var = Mutable(self._fresh(name), initial)
        self._add(Assign(var, var.assigned[0], declares=True))
        return var

    def set(self, var, value):
        """Assigns `value` to variable `var` of `k.var`."""
        if not isinstance(var, Mutable):
            raise TypeError("kohakutt.dsl.metal: set takes a variable of k.var")
        value = as_expr(value)
        var.assigned.append(value)
        self._add(Assign(var, value))

    @contextlib.contextmanager
    def when(self, condition):
        """Runs the statements inside only where `condition` (a comparison: `a < b`,
        `a.ne(b)`, `a.eq(b)`) holds on the device."""
        if not (isinstance(condition, Op) and condition.op in COMPARISONS):
            raise TypeError("kohakutt.dsl.metal: when takes a comparison of values")
        stmt = self._add(When(condition))
        self._stack.append(stmt.body)
        try:
            yield
        finally:
            self._stack.pop()

    def blocks(self, count, block, names=("done", "n")):
        """A device loop over `count` items `block` at a time; yields (items done, items
        of this block)."""
        count = as_expr(count)
        top = count.upper()
        done = Var(self._fresh(names[0]), upper=None if top is None else top - 1)
        loop = Loop(done, count, block)
        for _ in self._loop(loop):
            n = Var(self._fresh(names[1]), upper=as_expr(block).upper())
            n.block_of = loop
            loop.remainder = n
            self._add(Let(n, minimum(count - done, block)))
            yield done, n

    def min(self, a, b):
        """The smaller of `a` and `b`."""
        return minimum(a, b)

    def let(self, name, expr):
        """A named device variable holding `expr`."""
        expr = as_expr(expr)
        var = Var(self._fresh(name), upper=expr.upper())
        var.source = expr
        self._add(Let(var, expr))
        return var

    def inline(
        self, code, uses=(), sems=(), clobbers=(), times=1, sfpu=None, args=None
    ):
        """Inline C++ at this point. Declared: the buffers (`uses`) and semaphores it
        touches, the vector-unit constants it writes (`clobbers`) and its thread
        (`sfpu`), and `args` {C++ name: value or reserved slot} it reads."""
        sfpu = "math" if clobbers and sfpu is None else sfpu
        stmt = Inline(code, tuple(uses), tuple(sems), tuple(clobbers), times, sfpu)
        stmt.args = {
            name: v if isinstance(v, Slot) else as_expr(v)
            for name, v in (args or {}).items()
        }
        self._add(stmt)

    def helper(self, fn):
        """Emits C++ helper `fn` (a CppFn) above this kernel, once."""
        if fn not in self.kernel.helpers:
            self.kernel.helpers.append(fn)

    def define(self, name, value):
        """Preprocessor define `name` of this kernel."""
        self.kernel.defines[name] = value

    def include(self, header):
        """An extra `#include` of this kernel."""
        self.kernel.includes.append(header)

    def _cb(self, kind, cb, pages):
        slot = None
        if kind == "reserve":
            slot = Slot(cb, pages, self._fresh(f"{cb.name}_ptr"))
        self._add(CBOp(kind, cb, pages, slot))
        return slot


class DM(Builder):
    """Builder of a data-movement kernel (dm0: reader, dm1: writer)."""

    def read(self, source, dst):
        """Reads `source` (`ref.page(i)` or `ref.span(...)`) into `dst` (`slot[i]`)."""
        if isinstance(dst, Slot):
            dst = dst[0]
        offset = as_expr(source.offset)
        self._add(Read(source.ref, source.index, dst, offset, source.size))

    def write(self, cb, target, index=0):
        """Writes to `target` (`ref.page(i)`) waited page `index` of buffer `cb` (the
        front page by default), or the first page of a reserved slot given as `cb`."""
        if isinstance(cb, Slot):
            self._add(Write(cb.cb, target.ref, target.index, None, cb))
        else:
            self._add(Write(cb, target.ref, target.index, as_expr(index)))

    def reads_done(self):
        """Waits until every read issued so far has landed."""
        self._add(Barrier("reads_done"))

    def flushed(self):
        """Waits until every write issued so far has left this core."""
        self._add(Barrier("flushed"))

    def writes_done(self):
        """Waits until every write issued so far has been acknowledged."""
        self._add(Barrier("writes_done"))


class Compute(Builder):
    """Builder of the compute kernel."""

    def __init__(self, program, kernel):
        super().__init__(program, kernel)
        self._scope = None

    def startup(self, in0, in1, out):
        """Hardware setup for inputs `in0`, `in1` and output `out`."""
        self._add(Startup(in0, in1, out))

    def init(self, name, cb=None):
        """The setup call of op `name`: "copy" (with its buffer), "silu", or a binary
        op ("add", "sub", "mul", "div")."""
        self._add(Init(name, cb))

    def commit(self):
        """Hands the registers from the math thread to the packer; needed before a
        loop of packs (a reserve or pack directly in the scope commits by itself)."""
        self._commit()

    def binary(self, name, a, b, out):
        """Register slot `out` = slot `a` `name` slot `b` on the vector unit; `name`
        is "add", "sub", "mul" or "div"."""
        self._add(Binary(name, a, b, out))

    def srca_format(self, cb):
        """Switches the unpacker's first source to the format of `cb`."""
        self._add(SrcaFormat(cb))

    @contextlib.contextmanager
    def registers(self):
        """A tile register scope: math first; the first reserve or pack inside commits
        the registers to the packer."""
        self._add(Registers("acquire"))
        self._scope = "math"
        yield
        if self._scope == "math":
            self._add(Registers("commit"))
        self._add(Registers("release"))
        self._scope = None

    def _commit(self):
        if self._scope == "math":
            self._add(Registers("commit"))
            self._scope = "pack"

    def _cb(self, kind, cb, pages):
        if kind == "reserve":
            self._commit()
        return super()._cb(kind, cb, pages)

    def copy(self, cb, index, dst):
        """Copies tile `index` of the waited pages of `cb` into register slot `dst`."""
        self._add(Copy(cb, as_expr(index), dst))

    def silu(self, dst):
        """Register slot `dst` = silu(slot `dst`)."""
        self._add(Unary("silu", dst))

    def mul(self, a, b, out):
        """Register slot `out` = slot `a` * slot `b`, on the vector unit."""
        self._add(Binary("mul", a, b, out))

    def sfpu_binary(self, fn, a, b, out):
        """Runs inline vector-unit function `fn` on slots `a`, `b` into `out`."""
        self.helper(fn)
        self._add(SfpuCall(fn, a, b, out))

    def pack(self, dst, cb):
        """Packs register slot `dst` into the reserved page of `cb`."""
        self._commit()
        self._add(Pack(dst, cb))


def walk(body, path=()):
    """Every statement of `body` with the loops around it, in program order."""
    for stmt in body:
        yield stmt, path
        if isinstance(stmt, (Loop, When)):
            yield from walk(stmt.body, (*path, stmt))


def loops(path):
    """The loops of `path` (the conditions around a statement left out)."""
    return tuple(p for p in path if isinstance(p, Loop))


def conditional(path):
    """True when a statement under `path` sits inside a `when`."""
    return any(isinstance(p, When) for p in path)


def exprs(stmt):
    """The device values statement `stmt` reads."""
    if isinstance(stmt, Loop):
        found = [stmt.count, stmt.step, stmt.start]
    elif isinstance(stmt, When):
        found = [stmt.condition]
    elif isinstance(stmt, (Let, Assign)):
        found = [stmt.expr]
    elif isinstance(stmt, CBOp):
        found = [stmt.pages]
    elif isinstance(stmt, Read):
        found = [stmt.page, stmt.offset, stmt.dst.index, stmt.dst.offset, stmt.dst.like]
    elif isinstance(stmt, Write):
        found = [stmt.page, stmt.index]
    elif isinstance(stmt, Copy):
        found = [stmt.index, stmt.dst]
    elif isinstance(stmt, (Unary, Pack)):
        found = [stmt.dst]
    elif isinstance(stmt, (Binary, SfpuCall)):
        found = [stmt.a, stmt.b, stmt.out]
    elif isinstance(stmt, Inline):
        found = [stmt.times, *stmt.args.values()]
    else:
        found = []
    return [e for e in found if isinstance(e, Expr)]
