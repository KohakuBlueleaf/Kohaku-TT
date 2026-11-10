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
from kohakutt.dsl.metal.ops import OPS, UNPACK_INITS

# How often a vector-unit function runs per call: per face, or once.
SFPU_MODES = ("RC", "none")


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
    """`if (condition) { body } else { orelse }`; `orelse` is None without an else."""

    condition: Expr
    body: list = field(default_factory=list)
    orelse: list | None = None

    def __post_init__(self):
        # One entry per when, so that paths through one else branch compare equal.
        self.else_entry = Else(self)


class Else:
    """The else branch of `when` as an entry of a statement's path."""

    def __init__(self, when):
        self.when = when


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
    """Unpacker / packer setup for the kernel's first formats; `in1` None: one
    input; `reverse`: the matrix unit's source order."""

    in0: object
    in1: object
    out: object
    reverse: bool = False


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
class Format:
    """Switches the unpacker's sources to the formats of `srca` and `srcb` (None:
    left as it is)."""

    srca: object = None
    srcb: object = None


@dataclass
class PackFormat:
    """Switches the packer to the format of `cb`."""

    cb: object


@dataclass
class PackAccumulate:
    """Packer accumulation into the L1 pages of `cb` on (1) or off (0): a pack adds
    to what the page holds."""

    on: object
    cb: object = None


@dataclass
class Copy:
    """Tile `index` of `cb` into register slot `dst`, unpacked as `kind`: "copy",
    "transpose", "bcast_row", "bcast_col", or "copy_block" of `count` tiles."""

    cb: object
    index: Expr
    dst: int
    kind: str = "copy"
    count: object = 1


@dataclass
class MatmulInit:
    """Matrix-unit setup for blocks of `rt` x `ct` tiles of `a` @ `b` over `kt`
    inner tiles; `transpose`: `b` transposed within its tiles."""

    a: object
    b: object
    transpose: object = 0
    ct: object = 1
    rt: object = 1
    kt: object = 1


@dataclass
class Matmul:
    """Adds tile `ia` of `a` times `ct` tiles of `b` from `ib` into register slots
    `dst` .. `dst + ct * rt - 1` (the shape of the setup `init`)."""

    init: MatmulInit
    ia: Expr
    ib: Expr
    dst: object = 0


@dataclass
class Apply:
    """Tile op `name` of the op table on register `slots` (the last is the result)
    with fp32 scalar `values` (floats, or kernel values holding the bits)."""

    name: str
    slots: tuple
    values: tuple = ()


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
    """Inline vector-unit function `fn` over slots `a`, `b` into `out`: once per
    face ("RC"), or once for the registers ("none"), with extra arguments `args`."""

    fn: object
    a: int
    b: int
    out: int
    mode: str = "RC"
    args: tuple = ()


@dataclass
class Pack:
    """Register slot `dst` into the reserved pages of `cb`: the next page, or page
    `at` of the reserve."""

    dst: int
    cb: object
    at: Expr | None = None


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
        """A device loop over `start` (default 0) .. count - 1 in steps of `step`,
        written `for i in k.range(...)`; yields its variable."""
        with self.loop(count, step, name, start) as var:
            yield var

    def blocks(self, count, block, names=("done", "n")):
        """A device loop over `count` items `block` at a time, written `for done, n
        in k.blocks(...)`; yields (items done, items of this block)."""
        with self.block_loop(count, block, names) as pair:
            yield pair

    @contextlib.contextmanager
    def loop(self, count, step=1, name="i", start=None):
        """`range` as a context manager: `with k.loop(count) as i:`."""
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

    @contextlib.contextmanager
    def otherwise(self):
        """The else branch of the `when` just closed: `with k.otherwise():`."""
        last = self._stack[-1][-1] if self._stack[-1] else None
        if not isinstance(last, When) or last.orelse is not None:
            raise TypeError("kohakutt.dsl.metal: otherwise follows a when block")
        last.orelse = []
        self._stack.append(last.orelse)
        try:
            yield
        finally:
            self._stack.pop()

    @contextlib.contextmanager
    def block_loop(self, count, block, names=("done", "n")):
        """`blocks` as a context manager: `with k.block_loop(n, 4) as (done, m):`."""
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

    def macro(self, name, value):
        """Preprocessor define `name` of this kernel (a value of the whole program is
        named with `p.define`)."""
        self.kernel.defines[name] = value

    def include(self, header):
        """An extra `#include` of this kernel."""
        self.kernel.includes.append(header)

    def reserve(self, cb, pages):
        """Reserves `pages` at the back of `cb`; returns the slot to fill."""
        return self._cb("reserve", cb, pages)

    def push(self, cb, pages):
        """Hands `pages` reserved pages of `cb` to the consumer."""
        self._cb("push", cb, pages)

    def wait(self, cb, pages):
        """Waits until `pages` pages are at the front of `cb`."""
        self._cb("wait", cb, pages)

    def pop(self, cb, pages):
        """Frees `pages` pages at the front of `cb`."""
        self._cb("pop", cb, pages)

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

    def startup(self, in0, in1, out, reverse=False):
        """Hardware setup for inputs `in0`, `in1` (None: one input) and output `out`;
        `reverse` takes the matrix unit's source order."""
        self._add(Startup(in0, in1, out, reverse))

    def init(self, name, cb=None):
        """The setup call of op `name`: a way of unpacking with its buffer ("copy",
        "transpose", "bcast_row", "bcast_col") or an op of the op table ("silu",
        "add", "rsqrt", "mul_scalar", ...)."""
        if name not in OPS and name not in UNPACK_INITS:
            raise ValueError(f"kohakutt.dsl.metal: no setup call named {name!r}")
        if (name in UNPACK_INITS) != (cb is not None):
            raise TypeError(f"kohakutt.dsl.metal: init {name!r}: buffer given wrongly")
        self._add(Init(name, cb))

    def op(self, name, *slots, value=None, values=()):
        """Tile op `name` of the op table on register `slots` (operands, then the
        result; one slot: in place), e.g. `c.op("rsqrt", 0)`, `c.op("max", 0, 1, 0)`,
        `c.op("add_scalar", 0, value=1e-6)`. Its setup is `c.init(name)`."""
        given = (*values, *(() if value is None else (value,)))
        entry = OPS.get(name)
        if entry is None:
            raise ValueError(f"kohakutt.dsl.metal: no tile op named {name!r}")
        if len(slots) != entry.slots or len(given) != entry.values:
            raise TypeError(
                f"kohakutt.dsl.metal: {name} takes {entry.slots} register slots and "
                f"{entry.values} scalar values"
            )
        self._add(Apply(name, tuple(slots), given))

    def format(self, srca=None, srcb=None):
        """Switches the unpacker's sources to the formats of buffers `srca`, `srcb`."""
        self._add(Format(srca, srcb))

    def pack_format(self, cb):
        """Switches the packer to the format of buffer `cb`."""
        self._add(PackFormat(cb))

    def pack_accumulate(self, on, cb=None):
        """Packer accumulation on or off: with it on, a pack into buffer `cb` adds
        to what its page holds in L1."""
        self._add(PackAccumulate(on, cb))

    def matmul_init(self, a, b, transpose=0, ct=1, rt=1, kt=1):
        """Matrix-unit setup: blocks of `rt` x `ct` output tiles of `a` @ `b` over
        `kt` inner tiles (`transpose`: b transposed within its tiles). Returns the
        setup that `c.matmul` takes."""
        return self._add(MatmulInit(a, b, transpose, ct, rt, kt))

    def matmul(self, init, ia, ib, dst=0):
        """Adds tile `ia` of a times the `ct` tiles of b from `ib` into register slots
        `dst` .. (the registers sum over calls), in the shape of setup `init`."""
        self._add(Matmul(init, as_expr(ia), as_expr(ib), dst))

    def bcast(self, cb, index, dst, along):
        """Unpacks tile `index` of `cb` into slot `dst` with its first row ("row")
        or first column ("col") repeated over the tile."""
        if along not in ("row", "col"):
            raise ValueError("kohakutt.dsl.metal: bcast is along 'row' or 'col'")
        self._add(Copy(cb, as_expr(index), dst, f"bcast_{along}"))

    def transpose(self, cb, index, dst):
        """Unpacks tile `index` of `cb` into slot `dst`, transposed."""
        self._add(Copy(cb, as_expr(index), dst, "transpose"))

    def copy_block(self, cb, index, dst, count):
        """Copies `count` tiles of `cb` from `index` into slots `dst` .."""
        self._add(Copy(cb, as_expr(index), dst, "copy_block", count))

    def sfpu(self, fn, a, b, out, mode="RC", args=()):
        """Runs vector-unit function `fn` on slots `a`, `b` into `out`: once per face
        ("RC") or once over the whole registers ("none"), with extra `args`."""
        if mode not in SFPU_MODES:
            raise ValueError(f"kohakutt.dsl.metal: sfpu mode is one of {SFPU_MODES}")
        self.helper(fn)
        self._add(SfpuCall(fn, a, b, out, mode, tuple(args)))

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
        self.acquire()
        yield
        self.release()

    def acquire(self):
        """Opens a tile register scope (`registers` written out); `release` ends it."""
        self._add(Registers("acquire"))
        self._scope = "math"

    def release(self):
        """Ends the register scope, committing first when nothing has."""
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

    def pack(self, dst, cb, at=None):
        """Packs register slot `dst` into the reserved pages of `cb`: the next page,
        or page `at` of the reserve."""
        self._commit()
        self._add(Pack(dst, cb, None if at is None else as_expr(at)))


def walk(body, path=()):
    """Every statement of `body` with the loops and conditions around it, in
    program order (an else branch's entry is an Else)."""
    for stmt in body:
        yield stmt, path
        if isinstance(stmt, (Loop, When)):
            yield from walk(stmt.body, (*path, stmt))
        if isinstance(stmt, When) and stmt.orelse is not None:
            yield from walk(stmt.orelse, (*path, stmt.else_entry))


def loops(path):
    """The loops of `path` (the conditions around a statement left out)."""
    return tuple(p for p in path if isinstance(p, Loop))


def conditional(path):
    """True when a statement under `path` sits inside a `when` or its else."""
    return any(isinstance(p, (When, Else)) for p in path)


def conditions(path):
    """(condition, branch taken) of every `when` around a statement under `path`."""
    found = [(p.condition, True) for p in path if isinstance(p, When)]
    return found + [(p.when.condition, False) for p in path if isinstance(p, Else)]


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
        found = [stmt.index, stmt.dst, stmt.count]
    elif isinstance(stmt, Unary):
        found = [stmt.dst]
    elif isinstance(stmt, Pack):
        found = [stmt.dst, stmt.at]
    elif isinstance(stmt, Binary):
        found = [stmt.a, stmt.b, stmt.out]
    elif isinstance(stmt, SfpuCall):
        found = [stmt.a, stmt.b, stmt.out, *stmt.args]
    elif isinstance(stmt, Apply):
        found = [*stmt.slots, *stmt.values]
    elif isinstance(stmt, MatmulInit):
        found = [stmt.transpose, stmt.ct, stmt.rt, stmt.kt]
    elif isinstance(stmt, Matmul):
        found = [stmt.ia, stmt.ib, stmt.dst, *exprs(stmt.init)]
    elif isinstance(stmt, PackAccumulate):
        found = [stmt.on]
    elif isinstance(stmt, Inline):
        found = [stmt.times, *stmt.args.values()]
    else:
        found = []
    return [e for e in found if isinstance(e, Expr)]
