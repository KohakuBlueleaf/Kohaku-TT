"""The metal-level program model: cores, circular buffers, scratch, semaphores, tensor
references, the argument schema and the dm0 / dm1 / compute kernels.
"""

import bisect
import inspect
import itertools
import math
from dataclasses import dataclass, field

from kohakutt.dsl.metal.binding import (
    PER_CORE,
    Auto,
    CoreCount,
    CoreIndex,
    Expr,
    NocCoord,
    Select,
    Sized,
    Static,
    Value,
    Var,
    VerifyError,
    as_expr,
    minimum,
)
from kohakutt.dsl.metal.body import DM, Compute, Inline, exprs, walk
from kohakutt.dsl.metal.machine import SIZE_CLASSES, CoreSet

# Bytes of one 32x32 tile per storage format.
TILE_BYTES = {
    "bfloat16": 2048,
    "float32": 4096,
    "int32": 4096,
    "uint32": 4096,
    "bfloat8_b": 1088,
}
# Bytes of one element of the formats a tensor by rows may have.
ELEM_BYTES = {"bfloat16": 2, "float32": 4, "int32": 4, "uint32": 4}
# Bytes of a control-word page (`p.words`).
WORD_PAGE = 16
ROLES = ("dm0", "dm1", "compute")
# Values a sized value is rounded up to unless it names its own: steps of at most
# 1.5x, like the core counts.
PAGE_CLASSES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
# Combinations of sized values the default footprint may walk.
MAX_FOOTPRINTS = 50000


@dataclass
class Page:
    """Page `index` of `ref`, or `size` bytes at `offset` inside it."""

    ref: object
    index: Expr
    offset: object = 0
    size: int | None = None


class TensorRef:
    """A device tensor of the program: input `name` in storage format `dtype`,
    interleaved in DRAM ("dram") or L1 ("l1"), by tiles or (`rows`: the static last
    dim) by rows of its last dim. Its address is a runtime value."""

    is_output = False

    def __init__(self, name, dtype, kind="dram", rows=None):
        if dtype not in TILE_BYTES:
            raise ValueError(f"kohakutt.dsl.metal: unknown dtype {dtype!r}")
        if rows is not None and (
            dtype not in ELEM_BYTES or not isinstance(rows, int) or rows < 1
        ):
            raise ValueError(
                "kohakutt.dsl.metal: a tensor by rows has a static width (an int) "
                f"and a format of {sorted(ELEM_BYTES)}"
            )
        self.name, self.dtype, self.kind, self.rows = name, dtype, kind, rows

    @property
    def page_bytes(self):
        """Bytes of one page: a tile, or one row."""
        if self.rows is not None:
            return self.rows * ELEM_BYTES[self.dtype]
        return TILE_BYTES[self.dtype]

    def page(self, index):
        """Page `index` as a NoC source or target."""
        return Page(self, as_expr(index))

    def span(self, page, offset, size):
        """`size` bytes at byte `offset` of page `page`, as a NoC source."""
        return Page(self, as_expr(page), as_expr(offset), int(size))


class Out(TensorRef):
    """An output tensor of the program."""

    is_output = True


class CB:
    """Circular buffer `index` of `pages` pages in format `dtype`, on every core;
    pages of `page` bytes when given, else tiles."""

    def __init__(self, program, name, index, pages, dtype, page=None):
        self.program, self.name, self.index = program, name, index
        self.pages, self.dtype, self.page = pages, dtype, page
        # Partial sums of km.matmul_passes: only that function calls on it.
        self.partials = False

    @property
    def page_bytes(self):
        """Bytes of one page."""
        return self.page or TILE_BYTES[self.dtype]

    def reserve(self, pages):
        """Reserves `pages` at the back; returns the slot to fill."""
        return self.program.builder()._cb("reserve", self, pages)

    def push(self, pages):
        """Hands `pages` reserved pages to the consumer."""
        self.program.builder()._cb("push", self, pages)

    def wait(self, pages):
        """Waits until `pages` pages are at the front."""
        self.program.builder()._cb("wait", self, pages)

    def pop(self, pages):
        """Frees `pages` pages at the front."""
        self.program.builder()._cb("pop", self, pages)


@dataclass
class Scratch:
    """`bytes` of raw L1 on every core, aligned to `align`, held as buffer `index`."""

    name: str
    index: int
    bytes: object
    align: int = 16


def _member(x, y, group, i, cols, cores):
    """(x, y) of member `i` of the group of `group` consecutive cores (row-major) the
    core at (x, y) belongs to; (x, y) itself past the grid's `cores` (no such group
    has work)."""
    index = y * cols + x
    member = index - index % group + i
    return (x, y) if member >= cores else (member % cols, member // cols)


@dataclass
class SyncPoint:
    """Barrier `name` (`p.sync_point`): semaphore `sem` counting arrivals on core
    `first` (core 0, or the first of each `group`), scratch `word` polling it."""

    name: str
    sem: object
    word: object
    first: object
    program: object
    group: int | None = None
    members: dict = field(default_factory=dict)

    def cores(self):
        """The cores that arrive: the group, else the layout's count, else the
        launch's."""
        if self.group is not None:
            return as_expr(self.group)
        layout = self.program.layout_of
        return layout if layout is not None else CoreCount()

    def member(self, i):
        """Member `i` (0 .. group - 1) of this core's group, as a NoC target."""
        if self.group is None or not 0 <= i < self.group:
            raise ValueError(f"kohakutt.dsl.metal: {self.name} has no member {i}")
        if i not in self.members:
            machine, group = self.program.machine, self.group
            ref = self.program.core_ref(
                lambda x, y: _member(x, y, group, i, machine.cols, machine.cores),
                name=f"{self.name}_member{i}",
            )
            ref.group = group
            self.members[i] = ref
        return self.members[i]


@dataclass
class Semaphore:
    """Semaphore `id` with value `initial` at launch on the cores of `cores`."""

    name: str
    id: int
    initial: object
    cores: CoreSet


@dataclass
class CoreRef:
    """Another core of the program: logical core `core` ((x, y)), and its NoC
    coordinates `x`, `y` as common arguments; or, with `place` ((x, y) of the running
    core -> (x, y)), a core relative to each core, its coordinates per-core."""

    name: str
    core: tuple
    x: object
    y: object
    place: object = None


class PerCore(Expr):
    """Per-core argument `name`. `values` gives each core's value: a function of the
    core index, or a value (one per entry) of the core index and core count."""

    binding = PER_CORE

    def __init__(self, name, values, count=1, position=0):
        self.name, self.values, self.count = name, values, count
        self.position = position

    def at(self, core, cores):
        """The argument's value on core `core` of `cores`."""
        if callable(self.values):
            value = self.values(core)
            return int(value[self.position] if self.count != 1 else value)
        value = self.values[self.position] if self.count != 1 else self.values
        return as_expr(value).eval({"core": core, "cores": cores})

    def cpp(self):
        return self.name if self.count == 1 else f"{self.name}_{self.position}"

    def leaves(self):
        return [self]


class CoreArg(PerCore):
    """Per-core argument `name` the host derives from the running core's logical
    place: `fn(x, y, cores, noc)` gives an int, or a logical core and axis for
    its NoC coordinate."""

    def __init__(self, name, fn):
        super().__init__(name, None)
        self.fn = fn

    def item(self, xy, cores, noc):
        """The description item of this argument on the core at `xy`."""
        value = self.fn(*xy, cores, noc)
        return {"noc": list(value)} if isinstance(value, tuple) else int(value)


class Rect:
    """The cores of rows `rows` (a CoreSet) in the running core's column that run,
    as a multicast target: corners `sx`, `sy`, `ex`, `ey` (NoC coordinates, ordered
    for the kernel's NoC) and their number `count`, all per-core arguments."""

    def __init__(self, name, rows, machine):
        self.name, self.rows, self.machine = name, rows, machine

        def members(x, cores):
            return [y for (cx, y) in cores if cx == x and rows.first <= y <= rows.last]

        def corner(axis, end):
            def fn(x, y, cores, noc):
                ys = members(x, machine.core_list(cores))
                if not ys:
                    raise ValueError(f"kohakutt.dsl.metal: rectangle {name} is empty")
                # A NoC 1 rectangle runs from the high corner to the low one.
                low = (end == 1) == (noc == 1)
                return (x, min(ys) if low else max(ys), axis)

            return CoreArg(f"{name}_{'se'[end]}{'xy'[axis]}", fn)

        self.sx, self.sy = corner(0, 0), corner(1, 0)
        self.ex, self.ey = corner(0, 1), corner(1, 1)
        self.count = CoreArg(
            f"{name}_n",
            lambda x, y, cores, noc: len(members(x, machine.core_list(cores))),
        )

    def args(self):
        """The per-core arguments of the rectangle."""
        return (self.sx, self.sy, self.ex, self.ey, self.count)


class TableArg(PerCore):
    """The per-core argument block of table `table`: its used row count, then `rows`
    rows of `width` words."""

    def __init__(self, table):
        super().__init__(table.name, None)
        self.table, self.size = table, 1 + table.rows * table.width

    def cpp(self):
        return f"{self.name}_at"


class TableWord(Expr):
    """Word `col` of row `row` of table `table` on the running core (col None: the
    number of rows used this launch)."""

    binding = PER_CORE

    def __init__(self, table, row=None, col=None):
        self.table, self.row, self.col = table, row, col

    def cpp(self):
        base = f"{self.table.name}_at"
        if self.col is None:
            return f"get_arg_val<uint32_t>({base})"
        at = self.row * self.table.width + (1 + self.col)
        return f"get_arg_val<uint32_t>({base} + {at.cpp()})"

    def key(self):
        if self.col is None:
            return f"table({self.table.name})[used]"
        return f"table({self.table.name})[{self.row.key()},{self.col}]"

    def upper(self):
        if self.col is None:
            return self.table.rows
        return self.table.most.get(self.col)

    def leaves(self):
        return [self.table.arg, *([] if self.row is None else self.row.leaves())]

    def parts(self):
        return [] if self.row is None else [self.row]


class Table:
    """A per-core table of `rows` rows of `width` words, filled by the host at every
    launch (`Launcher.run(..., tables={name: rows per core})`): `t.used` rows this
    launch, `t[j, c]` word `c` of row `j`; `most` {column: largest value}."""

    def __init__(self, name, width, rows, most, pages=None):
        self.name, self.width, self.rows, self.most = name, width, rows, dict(most)
        self.pages = dict(pages or {})
        self.used = TableWord(self)
        self.arg = TableArg(self)

    def __getitem__(self, at):
        row, col = at
        if (
            isinstance(col, bool)
            or not isinstance(col, int)
            or not 0 <= col < self.width
        ):
            raise ValueError(
                f"kohakutt.dsl.metal: table {self.name} has columns 0 .. "
                f"{self.width - 1}, not {col!r}"
            )
        return TableWord(self, as_expr(row), col)

    def words(self, rows, page_counts=None):
        """The argument block of one core for its `rows` (lists of `width` ints);
        `page_counts` {tensor name: pages} bounds the columns declared in `pages`."""
        try:
            rows = [list(row) for row in rows]
        except TypeError:
            raise TypeError(
                f"kohakutt.dsl.metal: table {self.name} takes, per core, a list of "
                "rows of ints"
            ) from None
        if len(rows) > self.rows:
            raise ValueError(
                f"kohakutt.dsl.metal: table {self.name} holds {self.rows} rows, not "
                f"{len(rows)}"
            )
        out = [len(rows)]
        for row in rows:
            if len(row) != self.width:
                raise ValueError(
                    f"kohakutt.dsl.metal: a row of table {self.name} has {self.width} "
                    "words"
                )
            for col, word in enumerate(row):
                top = self.most.get(col, 2**32 - 1)
                if isinstance(word, bool) or not isinstance(word, int):
                    raise TypeError(
                        f"kohakutt.dsl.metal: table words are ints: {word!r}"
                    )
                if not 0 <= word <= top:
                    raise ValueError(
                        f"kohakutt.dsl.metal: table {self.name} column {col} takes "
                        f"0 .. {top}, not {word}"
                    )
                out.append(word)
            for col, (ref, count) in self.pages.items():
                have = (page_counts or {}).get(ref.name)
                if have is None or row[col] + row[count] > have:
                    raise ValueError(
                        f"kohakutt.dsl.metal: table {self.name} reads pages "
                        f"{row[col]} .. {row[col] + row[count] - 1} of {ref.name}, "
                        f"which has {have} pages"
                    )
        return out + [0] * (self.size_words - len(out))

    @property
    def size_words(self):
        return self.arg.size


@dataclass
class Layout:
    """The arguments one kernel reads: tensors (accessor compile args and address
    common args, in this order), then value common args, then the core count when
    used; per core, the core index first when used, then the declared arguments."""

    tensors: list = field(default_factory=list)
    values: list = field(default_factory=list)
    core_count: bool = False
    core_index: bool = False
    per_core: list = field(default_factory=list)
    lets: list = field(default_factory=list)

    @property
    def common(self):
        """Number of common runtime arguments."""
        return len(self.tensors) + len(self.values) + int(self.core_count)

    @property
    def per_core_count(self):
        """Number of per-core runtime arguments."""
        return int(self.core_index) + sum(getattr(a, "size", 1) for a in self.per_core)


class Kernel:
    """One RISC program: its role, cores, NoC pin, body, defines and (for compute)
    numerics options."""

    def __init__(self, role, cores, noc=None, options=None):
        self.role, self.cores, self.noc = role, cores, noc
        self.name = role
        self.options = dict(options or {})
        self.body, self.defines, self.includes, self.helpers = [], {}, [], []
        # The number of cores with work (a program layout's): cores past it return.
        self.active = None

    def layout(self):
        """The Layout of the arguments the body reads."""
        out = Layout()

        def visit(expr):
            for leaf in expr.leaves():
                if isinstance(leaf, Value) and leaf.name not in out.values:
                    out.values.append(leaf.name)
                elif isinstance(leaf, CoreIndex):
                    out.core_index = True
                elif isinstance(leaf, CoreCount):
                    out.core_count = True
                elif isinstance(leaf, PerCore) and leaf not in out.per_core:
                    out.per_core.append(leaf)
                elif isinstance(leaf, Var) and leaf.definition is not None:
                    visit(leaf.definition)
                    if leaf not in out.lets:
                        out.lets.append(leaf)

        for stmt, _ in walk(self.body):
            ref = getattr(stmt, "ref", None)
            if ref is not None and ref not in out.tensors:
                out.tensors.append(ref)
            for expr in exprs(stmt):
                visit(expr)
        if self.active is not None:
            out.core_index = True
            visit(self.active)
        return out

    def inlines(self):
        """The inline stages of the body."""
        return [s for s, _ in walk(self.body) if isinstance(s, Inline)]


class Program:
    """A traced program on `machine`: build it through `@program` and `trace`."""

    def __init__(self, name, machine, size_classes=SIZE_CLASSES):
        self.name, self.machine, self.size_classes = name, machine, size_classes
        self.cores = CoreSet()
        self.inputs, self.outputs = [], []
        self.cbs, self.scratches, self.semaphores = [], [], []
        self.kernels, self.values, self.statics = {}, [], {}
        self.choices, self.multiples = {}, {}
        self.per_core_args, self.split_of, self.strided = [], None, False
        self.sized_values, self.layout_of = {}, None
        self._builder, self._classes = None, None
        self.core_refs, self.lanes, self._names = [], [], {}
        self.sync_points = []
        self.joints, self.joint_names, self.limits = {}, set(), []
        self.declared, self.tables = {}, []

    @property
    def tensors(self):
        """Inputs, then outputs: the order tensor indices of the launch refer to."""
        return [*self.inputs, *self.outputs]

    @property
    def core(self):
        """The index of the running core among the program's cores."""
        return CoreIndex()

    @property
    def core_count_value(self):
        """The number of cores the program runs on, as a kernel value."""
        return CoreCount()

    def builder(self):
        """The builder of the kernel being traced."""
        if self._builder is None:
            raise RuntimeError(
                "kohakutt.dsl.metal: buffer calls belong inside a kernel body"
            )
        return self._builder

    def _free_index(self):
        used = {b.index for b in (*self.cbs, *self.scratches)}
        return next(i for i in itertools.count() if i not in used)

    def cb(self, index=None, like=None, dtype=None, pages=2, name=None, page=None):
        """A circular buffer of `pages` pages in the format of tensor `like` or of
        `dtype`, each a tile or `page` bytes (a static int: one row, say); `index` is
        chosen when not given."""
        if (like is None) == (dtype is None):
            raise ValueError("kohakutt.dsl.metal: cb takes exactly one of like, dtype")
        if page is not None and (not isinstance(page, int) or page % 16):
            raise ValueError("kohakutt.dsl.metal: a page is a static multiple of 16")
        index = self._free_index() if index is None else index
        name = name or (like.name if like is not None else f"cb{index}")
        dtype = dtype or like.dtype
        cb = CB(self, self._claim("buffer", name), index, pages, dtype, page)
        self.cbs.append(cb)
        return cb

    def partials(self, index, pages, name=None):
        """An fp32 buffer of `pages` (an exact sized value: one output block) for
        the partial sums of km.matmul_passes, the only code that may use it."""
        cb = self.cb(index, dtype="float32", pages=pages, name=name)
        cb.partials = True
        return cb

    def limit(self, value, low=None, high=None):
        """Runtime value `value` lies in `low` .. `high` (ints, or `high` another
        runtime value), checked at every launch; returns it."""
        name = getattr(high, "name", high)
        self.limits.append((value.name, low, name))
        value.at_least, value.at_most = low, name
        return value

    def table(self, name, width, rows, most=None, pages=None):
        """Per-core table `name` of `rows` rows of `width` words, written by the host
        at every launch. `most` {column: largest value} bounds a column used as a
        count; `pages` {column: (tensor, count column)} a column used as a page index,
        checked at launch against the tensor's pages. Read-only on the device."""
        for what, n in (("rows", rows), ("width", width)):
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise TypeError(f"kohakutt.dsl.metal: a table's {what} is a static int")

        def column(col):
            if (
                isinstance(col, bool)
                or not isinstance(col, int)
                or not 0 <= col < width
            ):
                raise ValueError(
                    f"kohakutt.dsl.metal: table {name} has columns 0 .. {width - 1}, "
                    f"not {col!r}"
                )

        most, pages = dict(most or {}), dict(pages or {})
        for col, top in most.items():
            column(col)
            if (
                isinstance(top, bool)
                or not isinstance(top, int)
                or not 0 <= top < 2**32
            ):
                raise ValueError(
                    f"kohakutt.dsl.metal: table {name} column {col}: a most is an int "
                    f"in 0 .. 2**32 - 1, not {top!r}"
                )
        for col, given in pages.items():
            column(col)
            ref, count = (
                given if isinstance(given, tuple) and len(given) == 2 else (0, 0)
            )
            if not isinstance(ref, TensorRef) or ref not in self.tensors:
                raise TypeError(
                    f"kohakutt.dsl.metal: table {name} pages[{col}] is (a tensor of "
                    "the program, the column counting its pages)"
                )
            column(count)
            if count not in most or count == col:
                raise ValueError(
                    f"kohakutt.dsl.metal: table {name} counts the pages of column "
                    f"{col} with column {count}, which needs a most"
                )
        out = Table(self._claim("table", name), width, rows, most, pages)
        self.tables.append(out)
        return out

    def choose(self, names, table):
        """Runtime values `names` chosen together: each launch gives one row of
        `table` (tuples of ints >= 1). Returns the values."""
        table = tuple(tuple(row) for row in table)
        if any(name in self.values for name in names):
            raise ValueError(f"kohakutt.dsl.metal: values {names} are declared twice")
        out = []
        for i, name in enumerate(names):
            value = self.runtime(name, choices=sorted({row[i] for row in table}))
            value.joint = (tuple(names), table)
            self.joints[tuple(names)] = table
            out.append(value)
        self.joint_names.update(names)
        return out

    def words(self, index=None, pages=4, name=None):
        """A buffer of `pages` 16-byte pages for control words (`k.put`, `k.take`)."""
        index = self._free_index() if index is None else index
        name = self._claim("buffer", name or f"cb{index}")
        cb = CB(self, name, index, pages, "uint32", WORD_PAGE)
        self.cbs.append(cb)
        return cb

    def _claim(self, kind, name):
        """Reserves C++ name `name` of a `kind` (one per program and kind)."""
        taken = self._names.setdefault(kind, set())
        if name in taken:
            raise ValueError(f"kohakutt.dsl.metal: two {kind}s named {name}")
        taken.add(name)
        return name

    def core_ref(self, index, name=None):
        """Core `index` of the program (row-major) as a NoC target: `k.sem_inc(sem,
        at=ref)`, `k.fetch_add(sem, at=ref)`, `k.send(slot, to=ref, size)`. A
        function `index(x, y) -> (x, y)` names a core relative to each core."""
        if callable(index):
            name = self._claim("core", name or f"core{len(self.core_refs)}")

            def coord(axis):
                return CoreArg(
                    f"{name}_{'xy'[axis]}",
                    lambda x, y, cores, noc: (*index(x, y), axis),
                )

            ref = CoreRef(name, None, coord(0), coord(1), index)
            self.core_refs.append(ref)
            return ref
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise TypeError("kohakutt.dsl.metal: core_ref takes a static core index")
        if index >= self.machine.cores:
            raise ValueError(f"kohakutt.dsl.metal: no core {index} on the grid")
        name = self._claim("core", name or f"core{index}")
        core = (index % self.machine.cols, index // self.machine.cols)
        x, y = NocCoord(f"{name}_x", core, 0), NocCoord(f"{name}_y", core, 1)
        ref = CoreRef(name, core, x, y)
        self.core_refs.append(ref)
        return ref

    def lane(self, senders, receivers, name="lane"):
        """A lane from the cores of row set `senders` (one row) to those of
        `receivers` in the same column (km.Lane): `lane.open(k)`, `lane.send(k,
        slot, size)` on the senders, `lane.receive(k)` on the receivers."""
        from kohakutt.dsl.metal.lane import Lane

        if senders.first is None or senders.first != senders.last:
            raise ValueError("kohakutt.dsl.metal: a lane's senders are one row")
        if receivers.first is None:
            raise ValueError("kohakutt.dsl.metal: a lane's receivers name their rows")
        name = self._claim("lane", name)
        rect = Rect(self._claim("core", f"{name}_to"), receivers, self.machine)
        head = self.core_ref(lambda x, y: (x, senders.first), f"{name}_from")
        ready = self.semaphore(0, name=f"{name}_ready")
        flag = self.semaphore(0, name=f"{name}_valid")
        out = Lane(name, senders, receivers, rect, head, ready, flag)
        self.lanes.append(out)
        return out

    def sync_point(self, name="sync", group=None):
        """A barrier for `k.sync` of the program's cores, or (`group`, a static size)
        of each run of `group` consecutive cores: a semaphore the first core counts
        arrivals on, a word each core polls it into, and the first core's reference."""
        # The word takes the highest free buffer index, away from chosen ones.
        used = {b.index for b in (*self.cbs, *self.scratches)}
        index = max(i for i in range(32) if i not in used)
        word = Scratch(self._claim("buffer", f"{name}_word"), index, 16)
        self.scratches.append(word)
        if group is not None and (not isinstance(group, int) or group < 1):
            raise ValueError("kohakutt.dsl.metal: a sync group is a static size")
        cols, cores = self.machine.cols, self.machine.cores
        first = (
            self.core_ref(0, name=f"{name}_first")
            if group is None
            else self.core_ref(
                lambda x, y: _member(x, y, group, 0, cols, cores),
                name=f"{name}_first",
            )
        )
        first.group = group
        point = SyncPoint(name, self.semaphore(0, name=name), word, first, self, group)
        self.sync_points.append(point)
        return point

    def scratch(self, nbytes, align=16, name=None):
        """`nbytes` of raw L1 per core."""
        index = self._free_index()
        name = self._claim("buffer", name or f"scratch{index}")
        out = Scratch(name, index, nbytes, align)
        self.scratches.append(out)
        return out

    def semaphore(self, initial=0, cores=None, name=None, id=None):
        """A semaphore on `cores` (default: every core of the program)."""
        ident = len(self.semaphores) if id is None else id
        name = self._claim("semaphore", name or f"sem{ident}")
        out = Semaphore(name, ident, initial, cores or self.cores)
        self.semaphores.append(out)
        return out

    def sized(self, name, of, classes=PAGE_CLASSES, exact=False, limit=None):
        """Sized value `name`, computed per launch from runtime values `of` and
        rounded up to one of `classes`; it may be a buffer's page count, and a new
        value builds a program without compiling. `exact=True` takes `of` itself, up
        to `limit` (a build per distinct value)."""
        if name in self.values:
            raise ValueError(f"kohakutt.dsl.metal: two values named {name}")
        if exact and (isinstance(limit, bool) or not isinstance(limit, int)):
            raise TypeError("kohakutt.dsl.metal: an exact sized value states its limit")
        choices = tuple(range(1, limit + 1)) if exact else tuple(classes)
        if limit is not None and not exact:
            choices = tuple(c for c in choices if c <= limit)
        value = Sized(name, of, choices, exact)
        self.values.append(name)
        self.sized_values[name] = value
        return value

    def define(self, name, value):
        """Names `value` (computed from argument values and the core's place) as
        `name`: each kernel that reads it computes it once, at its start."""
        return Var(name, as_expr(value), as_expr(value).upper())

    def sizes(self, runtime):
        """{sized value name: its value} for the runtime values of one launch."""
        env = dict(runtime)
        for name, value in self.sized_values.items():
            env[name] = value.fit(value.of.eval(env))
        return {name: env[name] for name in self.sized_values}

    def layout(self, cores):
        """Declares how many cores have work: `cores`, a value of the launch. The
        program runs on the smallest size class not below it; a core derives its
        place from `p.core`, and cores at or past `cores` return at once."""
        if self.split_of is not None or self.layout_of is not None:
            raise ValueError("kohakutt.dsl.metal: one split or layout per program")
        self.layout_of = as_expr(cores)
        for kernel in self.kernels.values():
            kernel.active = self.layout_of

    def runtime(self, name, choices=None, multiple_of=None):
        """Runtime value `name`: one common argument, the same on every core. With
        `choices`, the only values a launch may give it (such a value may be a page
        count, a loop step or a register slot); with `multiple_of` (an int, a runtime
        value or its name), every launch value is a multiple of it."""
        if name in ("core", "cores"):
            raise ValueError(
                f"kohakutt.dsl.metal: {name!r} names the core index or count; "
                "a runtime value takes another name"
            )
        given = (tuple(sorted(choices)) if choices else None, multiple_of)
        if name in self.joint_names or self.declared.setdefault(name, given) != given:
            raise ValueError(
                f"kohakutt.dsl.metal: value {name} is declared twice, differently"
            )
        if name not in self.values:
            self.values.append(name)
        if choices:
            self.choices[name] = tuple(choices)
        if multiple_of is not None:
            of = multiple_of.name if isinstance(multiple_of, Value) else multiple_of
            if isinstance(of, bool) or not isinstance(of, (int, str)):
                raise TypeError(
                    "kohakutt.dsl.metal: multiple_of takes an int or a runtime value"
                )
            self.multiples[name] = of
        return Value(name, self.choices.get(name), self.multiples.get(name))

    def per_core(self, name, values=None, count=1):
        """Per-core argument `name` (or `count` of them), fixed for the program's core
        count: `values(core index)` gives a core's value(s); or `values` is a value
        (a list of `count`) computed from `p.core` and `p.core_count_value`."""
        self.per_core_args.append((name, values, count))
        args = [PerCore(name, values, count, i) for i in range(_length(count))]
        return args[0] if len(args) == 1 else args

    def split(self, total, min_per_core=1):
        """(first item, item count) of the running core when runtime value `total`
        is split evenly over the cores; the first `total % cores` take one more.
        `min_per_core` (an int or a runtime value) bounds the core count from the
        total."""
        if not isinstance(total, Value):
            raise TypeError("kohakutt.dsl.metal: split takes a runtime value")
        if self.split_of is not None or self.layout_of is not None:
            raise ValueError("kohakutt.dsl.metal: one split per program")
        if isinstance(min_per_core, Value):
            min_per_core = min_per_core.name
        self.split_of = (total.name, min_per_core)
        base = self.runtime(f"{total.name}_base")
        extra = self.runtime(f"{total.name}_extra")
        core = CoreIndex()
        start = Var("start", core * base + minimum(core, extra))
        count = Var("count", base + Select(core < extra, 1, 0))
        return start, count

    def stride(self, total):
        """(first item, step) of the running core when the items of runtime value
        `total` are dealt out in turn: core c takes c, c + cores, ... (loop with
        `k.range(total, start=first, step=step)`)."""
        if not isinstance(total, Value):
            raise TypeError("kohakutt.dsl.metal: stride takes a runtime value")
        if self.split_of is not None or self.layout_of is not None:
            raise ValueError("kohakutt.dsl.metal: one split per program")
        self.split_of = (total.name, 1)
        self.strided = True
        return CoreIndex(), CoreCount()

    def input(self, name, dtype, kind="dram", rows=None):
        """Declares input tensor `name` in storage format `dtype` (by rows of `rows`
        elements when given: page i is row i); returns its TensorRef. Inputs keep
        their order of declaration."""
        return self._tensor(TensorRef(name, dtype, kind, rows), self.inputs)

    def output(self, name, dtype, kind="dram", rows=None):
        """Declares output tensor `name` in storage format `dtype` (by rows of `rows`
        elements when given); returns its Out."""
        return self._tensor(Out(name, dtype, kind, rows), self.outputs)

    def _tensor(self, ref, into):
        if any(t.name == ref.name for t in self.tensors):
            raise ValueError(f"kohakutt.dsl.metal: two tensors named {ref.name}")
        into.append(ref)
        return ref

    def kernel(self, role, *, cores=None, noc=None, name=None, **options):
        """Opens kernel `role` ("dm0", "dm1" or "compute") and returns its builder;
        statements are added by calling the builder (`k.reserve(cb, n)`, `with
        k.loop(n) as i:`). `noc` pins a reader's or writer's NoC; `options` are
        those of `compute`. A data-movement role may have variants on row sets."""
        if role not in ROLES:
            raise ValueError(f"kohakutt.dsl.metal: unknown kernel role {role!r}")
        name = name or role
        if (role == "compute" and name != role) or (name in ROLES and name != role):
            raise ValueError(
                f"kohakutt.dsl.metal: a {role} kernel is not named {name}; compute is "
                "one kernel named compute"
            )
        variants = [k for k in self.kernels.values() if k.role == role]
        if name in self.kernels or (variants and role == "compute"):
            raise ValueError(f"kohakutt.dsl.metal: two {role} kernels named {name}")
        if variants and (cores is None or any(k.cores.first is None for k in variants)):
            raise ValueError(
                f"kohakutt.dsl.metal: variants of {role} each name their rows (cores=)"
            )
        if role == "compute":
            unknown = set(options) - COMPUTE_OPTIONS.keys()
            if unknown or noc is not None:
                raise TypeError(
                    f"kohakutt.dsl.metal: unknown compute options {unknown or 'noc'}"
                )
            options = {**COMPUTE_OPTIONS, **options}
        elif options:
            raise TypeError(f"kohakutt.dsl.metal: {role} takes no options {options}")
        kernel = Kernel(role, cores or self.cores, noc, options or None)
        kernel.name, kernel.active = name, self.layout_of
        self.kernels[name] = kernel
        return (Compute if role == "compute" else DM)(self, kernel)

    def _kernel(self, role, fn, cores, noc, options, name=None):
        def trace(body):
            builder = self.kernel(role, cores=cores, noc=noc, name=name, **options)
            self._builder = builder
            try:
                body(builder)
            finally:
                self._builder = None
            return builder.kernel

        return trace if fn is None else trace(fn)

    def dm0(self, fn=None, *, noc=None, cores=None, name=None):
        """Decorates the reader kernel's body `fn(k)`; `noc` pins its NoC (0 or 1);
        `cores` and `name` make it one variant of several on row sets."""
        return self._kernel("dm0", fn, cores, noc, {}, name)

    def dm1(self, fn=None, *, noc=None, cores=None, name=None):
        """Decorates the writer kernel's body `fn(k)`, as dm0."""
        return self._kernel("dm1", fn, cores, noc, {}, name)

    def compute(self, fn=None, *, cores=None, **options):
        """Decorates the compute kernel's body `fn(c)`. Options: fp32_dest,
        unpack_fp32 (buffers), fidelity, approx, min_fidelity, max_fidelity, exact."""
        unknown = set(options) - COMPUTE_OPTIONS.keys()
        if unknown:
            raise TypeError(f"kohakutt.dsl.metal: unknown compute options {unknown}")
        return self._kernel("compute", fn, cores, None, options)

    def footprint(self, sizes=None):
        """Bytes of L1 per core the buffers and scratch areas take with sized values
        `sizes` ({name: value}; default: the most over every combination they can
        take, since a size may shrink as a value grows; past MAX_FOOTPRINTS of them,
        the sum of each buffer's most, a bound above it)."""
        terms = [(cb.pages, cb.page_bytes) for cb in self.cbs]
        terms += [(s.bytes, 1) for s in self.scratches]
        if sizes is not None:
            return sum(as_expr(n).eval(sizes) * unit for n, unit in terms)
        if self._combinations(list(self.sized_values)) <= MAX_FOOTPRINTS:
            return self._most(list(self.sized_values), terms)
        return sum(self._most(self._sized_in(n), [(n, unit)]) for n, unit in terms)

    def _sized_in(self, expr):
        """The names of the sized values `expr` reads."""
        names = {getattr(leaf, "name", None) for leaf in as_expr(expr).leaves()}
        return [name for name in self.sized_values if name in names]

    def _combinations(self, names):
        return math.prod(len(self.sized_values[n].choices) for n in names)

    def _most(self, names, terms):
        """The most bytes `terms` [(count, unit)] take over the sized values `names`."""
        if self._combinations(names) > MAX_FOOTPRINTS:
            raise ValueError(
                f"kohakutt.dsl.metal: {self.name}: too many combinations of sized "
                "values to bound the footprint; pass sizes"
            )
        choices = [self.sized_values[name].choices for name in names]
        return max(
            sum(as_expr(n).eval(dict(zip(names, picked))) * unit for n, unit in terms)
            for picked in itertools.product(*choices)
        )

    def core_count(self, runtime):
        """The core count for runtime values `runtime`: the smallest size class not
        below the cores with work (the layout's count, or what the split's total
        fills; cores past it take no work), or the largest class without either."""
        if self._classes is None:
            self._classes = self.machine.classes(self.size_classes)
        if self.layout_of is not None:
            wanted = self.layout_of.eval({**runtime, **self.sizes(runtime)})
            if not 1 <= wanted <= self._classes[-1]:
                raise ValueError(
                    f"kohakutt.dsl.metal: {self.name}: a layout of {wanted} cores on "
                    f"a grid of {self._classes[-1]}"
                )
        elif self.split_of is None:
            return self._classes[-1]
        else:
            name, least = self.split_of
            wanted = runtime[name] // max(
                1, runtime[least] if isinstance(least, str) else least
            )
        at = bisect.bisect_left(self._classes, wanted)
        return self._classes[min(at, len(self._classes) - 1)]

    def launch_values(self, runtime, cores):
        """{name: value} of one launch on `cores` cores: the runtime values, the sized
        values and a split's per-core share."""
        full = {**runtime, **self.sizes(runtime)}
        if self.split_of is not None and not self.strided:
            name = self.split_of[0]
            full[f"{name}_base"], full[f"{name}_extra"] = divmod(runtime[name], cores)
        return full

    def value_list(self, runtime, cores):
        """The values of `self.values`, in order, for one launch on `cores` cores."""
        full = self.launch_values(runtime, cores)
        for name, allowed in self.choices.items():
            if full[name] not in allowed:
                raise ValueError(
                    f"kohakutt.dsl.metal: {self.name}: {name} = {full[name]} is not "
                    f"one of {allowed}"
                )
        for name, low, high in self.limits:
            top = full[high] if isinstance(high, str) else high
            if (low is not None and full[name] < low) or (
                top is not None and full[name] > top
            ):
                raise ValueError(
                    f"kohakutt.dsl.metal: {self.name}: {name} = {full[name]} is "
                    f"outside {low} .. {high}"
                )
        for names, table in self.joints.items():
            row = tuple(full[name] for name in names)
            if row not in table:
                raise ValueError(
                    f"kohakutt.dsl.metal: {self.name}: {names} = {row} is not one of "
                    f"{table}"
                )
        for name, of in self.multiples.items():
            unit = full[of] if isinstance(of, str) else of
            if full[name] % unit:
                raise ValueError(
                    f"kohakutt.dsl.metal: {self.name}: {name} = {full[name]} is not "
                    f"a multiple of {of if isinstance(of, int) else f'{of} = {unit}'}"
                )
        return [int(full[name]) for name in self.values]


COMPUTE_OPTIONS = {
    "fp32_dest": True,
    "unpack_fp32": (),
    "fidelity": None,
    "approx": None,
    "min_fidelity": None,
    "max_fidelity": None,
    "exact": False,
}


def _length(count):
    """`count` as a list length; a count that is not a static int gives 1."""
    return count if isinstance(count, int) and not isinstance(count, bool) else 1


class Template:
    """A program function `fn(p, tensors..., statics...)`; `trace` builds its Program
    for one kernel class."""

    def __init__(self, name, fn, size_classes=SIZE_CLASSES):
        self.name, self.fn, self.size_classes = name, fn, size_classes
        self.tensors, self.statics = {}, {}
        params = list(inspect.signature(fn).parameters.values())[1:]
        for p in params:
            kind = _annotation(p.annotation)
            if kind in (TensorRef, Out):
                self.tensors[p.name] = kind
            else:
                self.statics[p.name] = p.default

    def _build(self, machine, dtypes, kinds, statics):
        program = Program(self.name, machine, self.size_classes)
        program.statics = dict(statics)
        args = {}
        for name, cls in self.tensors.items():
            ref = cls(name, dtypes[name], kinds.get(name, "dram"))
            (program.outputs if cls is Out else program.inputs).append(ref)
            args[name] = ref
        self.fn(program, **args, **statics)
        return program

    def trace(self, machine, dtypes, kinds=None, l1_budget=None, **static):
        """The verified Program for tensor formats `dtypes` ({name: dtype}) and static
        values `static`; `auto` defaults take their first candidate that verifies."""
        from kohakutt.dsl.metal.verify import verify

        unknown = set(static) - self.statics.keys()
        if unknown:
            raise TypeError(f"kohakutt.dsl.metal: {self.name} has no static {unknown}")
        chosen = {**self.statics, **static}
        missing = [n for n, v in chosen.items() if v is inspect.Parameter.empty]
        if missing:
            raise TypeError(f"kohakutt.dsl.metal: {self.name} needs static {missing}")
        autos = [n for n, v in chosen.items() if isinstance(v, Auto)]
        error = None
        for pick in itertools.product(*(chosen[n].candidates for n in autos)):
            statics = {**chosen, **dict(zip(autos, pick))}
            try:
                program = self._build(machine, dtypes, kinds or {}, statics)
                verify(program, l1_budget=l1_budget)
                return program
            except VerifyError as e:
                error = error or e
        raise error


def _annotation(annotation):
    """TensorRef, Out or Static for a parameter annotation (a class or its name)."""
    if isinstance(annotation, str):
        tail = annotation.split("[")[0].rsplit(".", 1)[-1]
        return {"TensorRef": TensorRef, "Out": Out}.get(tail, Static)
    return annotation if annotation in (TensorRef, Out) else Static


def program(name, size_classes=SIZE_CLASSES):
    """Decorates a program function `fn(p, tensors..., statics...)` as a Template
    named `name` that runs on the core counts of `size_classes`."""

    def wrap(fn):
        return Template(name, fn, size_classes)

    return wrap
