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
    Word,
    as_expr,
    minimum,
)
from kohakutt.dsl.metal.ops import OPS, UNPACK_INITS

# How often a vector-unit function runs per call: per face, or once.
SFPU_MODES = ("RC", "none")
# Whole-tile matrix-unit ops, their broadcasts, and the reductions.
TILE_OPS = ("add", "sub", "mul")
TILE_BCASTS = (None, "row", "col", "scalar")
REDUCE_KINDS = ("sum", "max")
REDUCE_DIMS = ("row", "col", "scalar")
# The calls on the two-pass statistics registers.
STATS_KINDS = ("clear", "sum", "mean", "m2", "save", "restore", "finalize")


@dataclass
class Loop:
    """`for (var = start; var < count; var += step)`; `remainder` is a blocked loop's
    n; `start` is None for 0; `count` None: endless, left by its ExitWhen."""

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
class ExitWhen:
    """`if (condition) break;` of the endless loop around it."""

    condition: Expr


@dataclass
class Put:
    """One control word `value` into buffer `cb`: reserve, write, push."""

    cb: object
    value: Expr


@dataclass
class Hold:
    """Marks buffer `cb` held by the compute kernel: its pages, reserved once and
    never pushed, are packed into (or onto) and copied from in place."""

    cb: object


@dataclass
class Fence:
    """The packs before this point have landed in L1 before the unpacker reads on
    (a token through buffer `cb`, pushed by the pack thread, waited by the unpack)."""

    cb: object


@dataclass
class Take:
    """The next control word of buffer `cb` into `var`: wait, read, pop."""

    var: object
    cb: object


@dataclass
class ReadWord:
    """Word `word` (32-bit, face order) of waited page `index` of buffer `cb` into
    `var`, the same on the compute kernel's three threads."""

    var: object
    cb: object
    index: object
    word: int


@dataclass
class SemOp:
    """A semaphore operation: "inc" (by `value`, on core `at`, None: this core),
    "fetch_add" (one on core `at`; `var` gets the old value), "wait_min" (until at
    least `value`) or "set" (to `value`, this core)."""

    kind: str
    sem: object
    value: Expr
    at: object = None
    var: object = None
    lane: object = None


@dataclass
class Mcast:
    """NoC multicast of `size` bytes (a value: the whole reserve) at slot address
    `src` to the same L1 address on the cores of rectangle `to`, in lane `lane`."""

    src: object
    to: object
    size: object
    lane: object


@dataclass
class McastSet:
    """Sets semaphore `sem` on the cores of rectangle `to` to this core's value of
    it, as part of lane `lane`."""

    sem: object
    to: object
    lane: object


@dataclass
class Send:
    """NoC write of `size` bytes at slot address `src` to the same L1 address on
    core `to`."""

    src: object
    to: object
    size: int


@dataclass
class Zone:
    """Start ("begin") or end ("end") of device-profiler zone `name`: the statements
    between them run in one zone, a C++ block of their own."""

    name: str
    step: str


@dataclass
class Waypoint:
    """Watcher waypoint `name` (1 to 4 characters), written when it is reached."""

    name: str


@dataclass
class Sync:
    """Barrier `point` of the program's cores: this core's writes land, it arrives
    at core 0, then waits until every core running the program has arrived."""

    point: object


@dataclass
class Fetch:
    """NoC read of `size` bytes from core `at` at the L1 address of this core's slot
    `src` (the same pages there) into slot address `dst`."""

    src: object
    at: object
    dst: object
    size: int


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
    to what the page holds; `after` (a loop variable): off on its first iteration,
    on after it."""

    on: object
    cb: object = None
    after: object = None


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
class TilesInit:
    """Matrix-unit setup of `op` ("add", "sub", "mul") on tiles of buffers `a` and
    `b`; `bcast` ("row", "col", "scalar", None) repeats b's first row, first column
    or first element over the tile; `acc`: the result adds onto its register slot."""

    op: str
    a: object
    b: object
    bcast: str | None = None
    acc: bool = False


@dataclass
class Tiles:
    """Register slot `dst` = tile `ia` of a `op` tile `ib` of b, under setup `init`."""

    init: TilesInit
    ia: Expr
    ib: Expr
    dst: object = 0


@dataclass
class ReduceInit:
    """Matrix-unit reduction setup: `kind` ("sum", "max") along `dim` ("row": each row
    into its first column, "col": each column into its first row, "scalar") of tiles
    of `cb`, scaled by a tile of `scaler`; packs into `out` keep only the result."""

    kind: str
    dim: str
    cb: object
    scaler: object
    out: object


@dataclass
class Reduce:
    """Reduces tile `index` of the setup's buffer, scaled by tile `scaler_index` of its
    scaler, into register slot `dst`; calls on one slot sum (or take the max)."""

    init: ReduceInit
    index: Expr
    scaler_index: Expr
    dst: object = 0


@dataclass
class ReduceDone:
    """Restores the packer's default masks after a reduction (`reduce_uninit`)."""


@dataclass
class Tilize:
    """Outside a register scope, 32 rows of `block` tiles' width: "tilize" reads them
    as 32 one-row pages of `icb` and writes `block` tiles to `ocb`; "untilize" reads
    `block` tiles of `icb` and writes 32 one-row pages to `ocb` (static `block`)."""

    kind: str
    icb: object
    ocb: object
    block: int


@dataclass
class Stats:
    """A call on the vector unit's two-pass statistics registers: `kind` is one of
    STATS_KINDS (see `Compute.stats_*`) on register `slot`, rows `first` .. `first +
    count` of it, fp32 `value` (a reciprocal); `anchor`, `mean` and `dual` are static.
    """

    kind: str
    slot: object = None
    first: object = 0
    count: object = 32
    value: object = None
    anchor: bool = False
    mean: bool = True
    dual: bool = False


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
    face ("RC"), or once for the registers ("none"), with extra arguments `args`,
    issued by compute thread `thread` ("math" or "pack")."""

    fn: object
    a: int
    b: int
    out: int
    mode: str = "RC"
    args: tuple = ()
    thread: str = "math"


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
    """The pages a reserve (or, `front`, a wait) returned; `slot[i]` is page i,
    `slot.at(b)` byte b."""

    def __init__(self, cb, pages, name, front=False):
        self.cb, self.pages, self.name, self.front = cb, pages, name, front

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
        self._partials = False

    def _add(self, stmt):
        self._stack[-1].append(stmt)
        return stmt

    def _fresh(self, base):
        while True:
            count = self._names.get(base, 0) + 1
            self._names[base] = count
            name = base if count == 1 else f"{base}{count}"
            if name not in self.program.values:
                return name

    def _loop(self, loop):
        self._add(loop)
        self._stack.append(loop.body)
        try:
            yield
        finally:
            self._stack.pop()

    @contextlib.contextmanager
    def forever(self, name="it"):
        """An endless device loop, `with k.forever() as it:`, left by its one
        `k.exit_when(condition)` on a control word."""
        var = Var(self._fresh(name))
        for _ in self._loop(Loop(var, None, 1)):
            yield var

    def exit_when(self, condition):
        """Leaves the endless loop around it where `condition` holds."""
        if not (isinstance(condition, Op) and condition.op in COMPARISONS):
            raise TypeError("kohakutt.dsl.metal: exit_when takes a comparison")
        self._add(ExitWhen(condition))

    def take(self, cb, name="word"):
        """The next control word of buffer `cb` (a page of `p.words`)."""
        var = Word(self._fresh(name), f"word({cb.name})")
        self._add(Take(var, cb))
        return var

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

    @contextlib.contextmanager
    def zone(self, name):
        """The statements inside run as device-profiler zone `name` (recorded only in
        a profiler build of TT-Metal); values declared inside stay inside."""
        self._add(Zone(name, "begin"))
        yield
        self._add(Zone(name, "end"))

    def waypoint(self, name):
        """Watcher waypoint `name` (1 to 4 letters or digits): a hung core shows the
        last one it reached (with TT_METAL_WATCHER set)."""
        self._add(Waypoint(name))

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
        if any(getattr(u.cb, "partials", False) for u in uses):
            raise ValueError(
                "kohakutt.dsl.metal: inline code does not use a partials buffer; only "
                "km.matmul_passes does"
            )
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

    def _cb(self, kind, cb, pages, front=False):
        if cb.partials and not self._partials:
            raise ValueError(
                f"kohakutt.dsl.metal: buffer {cb.name} holds partial sums; only "
                "km.matmul_passes uses it"
            )
        slot = None
        if kind == "reserve" or front:
            slot = Slot(cb, pages, self._fresh(f"{cb.name}_ptr"), front)
        self._add(CBOp(kind, cb, pages, slot))
        return slot


class DM(Builder):
    """Builder of a data-movement kernel (dm0: reader, dm1: writer)."""

    def front(self, cb, pages):
        """Waits for `pages` pages at the front of `cb`; returns them as a slot to
        write, send or fetch from (not into) until the next pop of `cb`."""
        return self._cb("wait", cb, pages, front=True)

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

    def put(self, cb, value):
        """One control word `value` into buffer `cb` (a page of `p.words`)."""
        self._add(Put(cb, as_expr(value)))

    def sem_inc(self, sem, at=None, by=1):
        """Adds `by` to semaphore `sem` on core `at` (`p.core_ref`; None: this core)."""
        self._add(SemOp("inc", sem, as_expr(by), at))

    def fetch_add(self, sem, at, name="fetched"):
        """Adds one to semaphore `sem` on core `at` and returns its value before."""
        var = Word(self._fresh(name), f"fetch({sem.name})")
        self._add(SemOp("fetch_add", sem, as_expr(1), at, var))
        return var

    def sem_wait_min(self, sem, value):
        """Waits until this core's semaphore `sem` is at least `value`."""
        self._add(SemOp("wait_min", sem, as_expr(value)))

    def sem_set(self, sem, value):
        """Sets this core's semaphore `sem` to `value`."""
        self._add(SemOp("set", sem, as_expr(value)))

    def atomics_done(self):
        """Waits until every semaphore increment issued so far has landed."""
        self._add(Barrier("atomics_done"))

    def _lane(self, stmt):
        """Adds statement `stmt` of a lane (km.Lane builds them)."""
        self._add(stmt)

    def sync(self, point):
        """Barrier `point` (`p.sync_point()`): returns when every core running the
        program has reached it as often as this one; writes sent before it have
        landed. One kernel role holds a point's syncs."""
        self._add(Sync(point))

    def fetch(self, src, at, dst, size):
        """Reads `size` bytes from core `at` (`p.core_ref`) at the address of this
        core's slot `src` (its own reserve of the same buffer) into slot `dst`."""
        src = src[0] if isinstance(src, Slot) else src
        dst = dst[0] if isinstance(dst, Slot) else dst
        self._add(Fetch(src, at, dst, int(size)))

    def send(self, src, to, size):
        """Writes `size` bytes at slot address `src` (`slot[i]`, `slot.at(b)`) to the
        same address on core `to` (`p.core_ref`)."""
        if isinstance(src, Slot):
            src = src[0]
        self._add(Send(src, to, int(size)))


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
        if entry is None or entry.call is None:
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

    def pack_accumulate(self, on, cb=None, after=None):
        """Packer accumulation on or off: with it on, a pack into buffer `cb` adds
        to what its page holds in L1. `on="after_first"` with loop variable `after`:
        off on that loop's first iteration (the pack writes), on after (it adds)."""
        if (on == "after_first") != (after is not None):
            raise ValueError(
                'kohakutt.dsl.metal: pack_accumulate("after_first", cb, after=i) '
                "takes the loop variable i"
            )
        if after is not None:
            self._add(PackAccumulate(1, cb, after))
        else:
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

    def tiles_init(self, op, a, b, bcast=None, acc=False):
        """Matrix-unit setup of whole-tile `op` ("add", "sub", "mul") of buffers `a`
        and `b`, b broadcast along `bcast` ("row", "col", "scalar") when given; `acc`
        (no broadcast): the result adds onto its slot. Returns what `c.tiles` takes."""
        if op not in TILE_OPS:
            raise ValueError(f"kohakutt.dsl.metal: tiles op is one of {TILE_OPS}")
        if bcast not in TILE_BCASTS:
            raise ValueError(f"kohakutt.dsl.metal: bcast is one of {TILE_BCASTS}")
        if acc and bcast is not None:
            raise ValueError("kohakutt.dsl.metal: acc=True takes no broadcast")
        return self._add(TilesInit(op, a, b, bcast, bool(acc)))

    def tiles(self, init, ia, ib, dst=0):
        """Register slot `dst` = tile `ia` of a op tile `ib` of b (setup `init`)."""
        self._add(Tiles(init, as_expr(ia), as_expr(ib), dst))

    def reduce_init(self, kind, dim, cb, scaler, out):
        """Matrix-unit reduction setup: `kind` ("sum", "max") along `dim` ("row",
        "col", "scalar") of tiles of `cb` times a tile of `scaler` (ones for a plain
        sum); packs into `out` keep only the result until `c.reduce_done()`."""
        if kind not in REDUCE_KINDS or dim not in REDUCE_DIMS:
            raise ValueError(
                f"kohakutt.dsl.metal: reduce kind is one of {REDUCE_KINDS}, dim one "
                f"of {REDUCE_DIMS}"
            )
        return self._add(ReduceInit(kind, dim, cb, scaler, out))

    def reduce(self, init, index, scaler_index, dst=0):
        """Reduces tile `index` of `init`'s buffer, scaled by tile `scaler_index` of
        its scaler, into slot `dst`; calls on one slot sum (or take the max)."""
        self._add(Reduce(init, as_expr(index), as_expr(scaler_index), dst))

    def reduce_done(self):
        """Restores the packer's default masks after reductions."""
        self._add(ReduceDone())

    def tilize(self, icb, ocb, block):
        """32 waited one-row pages of `icb` (rows of `block` tiles' width) into
        `block` reserved tiles of `ocb`; outside a register scope."""
        self._add(Tilize("tilize", icb, ocb, int(block)))

    def untilize(self, icb, ocb, block):
        """`block` waited tiles of `icb` into 32 reserved one-row pages of `ocb`;
        outside a register scope."""
        self._add(Tilize("untilize", icb, ocb, int(block)))

    def stats_clear(self):
        """Zeroes the statistics registers (anchor or mean, sums, M2)."""
        self._add(Stats("clear"))

    def stats_sum(self, slot, first=0, count=32, anchor=False, dual=False):
        """Adds rows `first` .. `first + count` of transposed slot `slot` (each lane a
        row of the original tile) to the shifted sums; `anchor`: the first call,
        which takes the shift anchor; `dual`: two accumulators (no save / restore)."""
        self._add(
            Stats("sum", slot, as_expr(first), as_expr(count), None, anchor, True, dual)
        )

    def stats_mean(self, recip, dual=False):
        """Turns the shifted sums into the mean (times `recip`, the fp32 1 / n:
        a float or a kernel value holding its bits) and clears the sums."""
        self._add(Stats("mean", value=recip, dual=dual))

    def stats_m2(self, slot, first=0, count=32, dual=False):
        """Adds the squares of rows `first` .. `first + count` of transposed slot
        `slot` minus the mean (0 after a clear) to M2."""
        self._add(Stats("m2", slot, as_expr(first), as_expr(count), dual=dual))

    def stats_save(self, slot, dual=False):
        """Spills the statistics to register slots `slot` and `slot + 1`, so other
        vector-unit work may run until `c.stats_restore(slot)`."""
        self._add(Stats("save", slot, dual=dual))

    def stats_restore(self, slot):
        """Loads the statistics `c.stats_save(slot)` spilled (one accumulator)."""
        self._add(Stats("restore", slot))

    def stats_finalize(self, slot, recip, mean=True, dual=False):
        """Writes the mean (when `mean`) into row 0 of slot `slot` and M2 * `recip`
        (the variance) into row 0 of slot `slot + 1`; the statistics end here."""
        self._add(Stats("finalize", slot, value=recip, mean=bool(mean), dual=dual))

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

    def run(self, fn, *slots, out, values=(), thread="math", tiles=None):
        """Runs generated vector-unit function `fn` (`km.sfpu_fn`) on its operand
        `slots` and fp32 scalar `values` into slot `out`, once per face, on the math
        thread or (`thread="pack"`, after the commit) the pack thread; `tiles=n`: over
        n output tiles in one call (`fn.tiles`)."""
        if thread not in ("math", "pack"):
            raise ValueError("kohakutt.dsl.metal: a function runs on math or pack")
        if tiles is None and any(getattr(fn, "along", {}).values()):
            # Broadcast operands are read at fixed places of their slots.
            tiles = 1
        if (tiles is not None or getattr(fn, "whole", False)) and thread != "math":
            raise ValueError("kohakutt.dsl.metal: tiles= runs on the math thread")
        if len(slots) != fn.operands or len(values) != fn.scalars:
            raise ValueError(
                f"kohakutt.dsl.metal: {fn.name} takes {fn.operands} operand slots "
                f"and {fn.scalars} scalar values"
            )
        if fn.fp32 and not self.kernel.options.get("fp32_dest", True):
            raise ValueError(f"kohakutt.dsl.metal: {fn.name} needs fp32 registers")
        b = slots[1] if len(slots) > 1 else slots[0]
        args = (*slots[2:], *values)
        if tiles is not None:
            fn = fn.tiles(tiles)
        self.helper(fn)
        whole = tiles is not None or getattr(fn, "whole", False)
        mode = "none" if whole else "RC"
        self._add(SfpuCall(fn, slots[0], b, out, mode, args, thread))

    def hold(self, cb):
        """Holds all pages of `cb` for the rest of the kernel (once, at the top): packs
        write (or with `pack_accumulate`, add onto) page `at`, copies read it, in
        place; `c.fence` orders a pack before a later copy."""
        self._add(CBOp("reserve", cb, cb.pages, Slot(cb, cb.pages, f"{cb.name}_held")))
        self._add(Hold(cb))

    def fence(self, token):
        """The packs so far land in L1 before the unpacker goes on: one page of
        buffer `token` (`p.words`) pushed by the pack thread and taken by the unpack
        thread. Comes after the register scope of those packs."""
        self._add(CBOp("reserve", token, 1))
        self._add(CBOp("push", token, 1))
        self._add(CBOp("wait", token, 1))
        self._add(CBOp("pop", token, 1))
        self._add(Fence(token))

    def read_word(self, cb, word=0, index=0, name="flag"):
        """The 32-bit word `word` (in the tile's face order: row 0 column 0 is 0, row
        0 column 16 is 256) of waited page `index` of `cb`, a kernel value every
        compute thread reads alike (`c.when(flag.ne(0))`)."""
        if not isinstance(word, int) or not 0 <= word < cb.page_bytes // 4:
            raise ValueError(f"kohakutt.dsl.metal: {cb.name} has no word {word}")
        var = Word(self._fresh(name), f"word({cb.name},{word})")
        self._add(ReadWord(var, cb, as_expr(index), word))
        return var

    def row_reduce(self, op, first, n, into, fresh=False):
        """Slot `into` = the row max or sum (`op` "max" / "sum") of slots `first` ..
        `first + n - 1`, with `into`'s previous value unless `fresh`, as a column
        broadcast, on the vector unit (`c.init("mul")` before)."""
        from kohakutt.dsl.metal.sfpu import row_reduce

        fn = row_reduce(op, n, fresh)
        self.run(fn, *((first,) if fresh else (first, into)), out=into)

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
    elif isinstance(stmt, Tiles):
        found = [stmt.ia, stmt.ib, stmt.dst]
    elif isinstance(stmt, Reduce):
        found = [stmt.index, stmt.scaler_index, stmt.dst]
    elif isinstance(stmt, Stats):
        found = [stmt.slot, stmt.first, stmt.count, stmt.value]
    elif isinstance(stmt, PackAccumulate):
        found = [stmt.on, stmt.after]
    elif isinstance(stmt, Inline):
        found = [stmt.times, *stmt.args.values()]
    elif isinstance(stmt, ExitWhen):
        found = [stmt.condition]
    elif isinstance(stmt, Put):
        found = [stmt.value]
    elif isinstance(stmt, SemOp):
        found = [stmt.value, *(() if stmt.at is None else (stmt.at.x, stmt.at.y))]
    elif isinstance(stmt, Send):
        found = [stmt.src.index, stmt.src.offset, stmt.to.x, stmt.to.y]
    elif isinstance(stmt, Fetch):
        found = [stmt.src.index, stmt.src.offset, stmt.at.x, stmt.at.y]
        found += [stmt.dst.index, stmt.dst.offset]
    elif isinstance(stmt, Sync):
        point = stmt.point
        found = [point.first.x, point.first.y, point.cores()]
    elif isinstance(stmt, Mcast):
        found = [stmt.src.index, stmt.src.offset, stmt.size, *stmt.to.args()]
    elif isinstance(stmt, McastSet):
        found = list(stmt.to.args())
    else:
        found = []
    return [e for e in found if isinstance(e, Expr)]
