"""The metal-level program model: cores, circular buffers, scratch, semaphores, tensor
references, the argument schema and the dm0 / dm1 / compute kernels.
"""

import bisect
import inspect
import itertools
from dataclasses import dataclass, field

from kohakutt.dsl.metal.binding import (
    PER_CORE,
    Auto,
    CoreCount,
    CoreIndex,
    Expr,
    Select,
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
TILE_BYTES = {"bfloat16": 2048, "float32": 4096, "int32": 4096, "bfloat8_b": 1088}
ROLES = ("dm0", "dm1", "compute")


@dataclass
class Page:
    """Page `index` of `ref`, or `size` bytes at `offset` inside it."""

    ref: object
    index: Expr
    offset: object = 0
    size: int | None = None


class TensorRef:
    """A device tensor of the program: input `name` in storage format `dtype`,
    interleaved in DRAM ("dram") or L1 ("l1"). Its address is a runtime value."""

    is_output = False

    def __init__(self, name, dtype, kind="dram"):
        if dtype not in TILE_BYTES:
            raise ValueError(f"kohakutt.dsl.metal: unknown dtype {dtype!r}")
        self.name, self.dtype, self.kind = name, dtype, kind

    @property
    def page_bytes(self):
        """Bytes of one page (a tile)."""
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
    """Circular buffer `index` of `pages` pages in format `dtype`, on every core."""

    def __init__(self, program, name, index, pages, dtype):
        self.program, self.name, self.index = program, name, index
        self.pages, self.dtype = pages, dtype

    @property
    def page_bytes(self):
        """Bytes of one page."""
        return TILE_BYTES[self.dtype]

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


@dataclass
class Semaphore:
    """Semaphore `id` with value `initial` at launch on the cores of `cores`."""

    name: str
    id: int
    initial: object
    cores: CoreSet


class PerCore(Expr):
    """Per-core argument `name`; `values(core index)` gives each core's value."""

    binding = PER_CORE

    def __init__(self, name, values, count=1, position=0):
        self.name, self.values, self.count = name, values, count
        self.position = position

    def cpp(self):
        return self.name if self.count == 1 else f"{self.name}_{self.position}"

    def leaves(self):
        return [self]


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
        return int(self.core_index) + len(self.per_core)


class Kernel:
    """One RISC program: its role, cores, NoC pin, body, defines and (for compute)
    numerics options."""

    def __init__(self, role, cores, noc=None, options=None):
        self.role, self.cores, self.noc = role, cores, noc
        self.options = dict(options or {})
        self.body, self.defines, self.includes, self.helpers = [], {}, [], []

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
        self._builder, self._classes = None, None

    @property
    def tensors(self):
        """Inputs, then outputs: the order tensor indices of the launch refer to."""
        return [*self.inputs, *self.outputs]

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

    def cb(self, index=None, like=None, dtype=None, pages=2, name=None):
        """A circular buffer of `pages` pages in the format of tensor `like` or of
        `dtype`; `index` is chosen when not given."""
        if (like is None) == (dtype is None):
            raise ValueError("kohakutt.dsl.metal: cb takes exactly one of like, dtype")
        index = self._free_index() if index is None else index
        name = name or (like.name if like is not None else f"cb{index}")
        cb = CB(self, name, index, pages, dtype or like.dtype)
        self.cbs.append(cb)
        return cb

    def scratch(self, nbytes, align=16, name=None):
        """`nbytes` of raw L1 per core."""
        index = self._free_index()
        out = Scratch(name or f"scratch{index}", index, nbytes, align)
        self.scratches.append(out)
        return out

    def semaphore(self, initial=0, cores=None, name=None, id=None):
        """A semaphore on `cores` (default: every core of the program)."""
        ident = len(self.semaphores) if id is None else id
        out = Semaphore(name or f"sem{ident}", ident, initial, cores or self.cores)
        self.semaphores.append(out)
        return out

    def runtime(self, name, choices=None, multiple_of=None):
        """Runtime value `name`: one common argument, the same on every core. With
        `choices`, the only values a launch may give it (such a value may be a page
        count, a loop step or a register slot); with `multiple_of` (an int, a runtime
        value or its name), every launch value is a multiple of it."""
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
        """Per-core argument `name` (or `count` of them): `values(core index)` gives
        a core's value(s), fixed for the program's core count."""
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
        if self.split_of is not None:
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
        if self.split_of is not None:
            raise ValueError("kohakutt.dsl.metal: one split per program")
        self.split_of = (total.name, 1)
        self.strided = True
        return CoreIndex(), CoreCount()

    def _kernel(self, role, builder, fn, cores, noc, options):
        def trace(body):
            if role in self.kernels:
                raise ValueError(f"kohakutt.dsl.metal: two {role} kernels")
            kernel = Kernel(role, cores or self.cores, noc, options)
            self.kernels[role] = kernel
            self._builder = builder(self, kernel)
            try:
                body(self._builder)
            finally:
                self._builder = None
            return kernel

        return trace if fn is None else trace(fn)

    def dm0(self, fn=None, *, noc=None, cores=None):
        """Decorates the reader kernel's body `fn(k)`; `noc` pins its NoC (0 or 1)."""
        return self._kernel("dm0", DM, fn, cores, noc, None)

    def dm1(self, fn=None, *, noc=None, cores=None):
        """Decorates the writer kernel's body `fn(k)`; `noc` pins its NoC (0 or 1)."""
        return self._kernel("dm1", DM, fn, cores, noc, None)

    def compute(self, fn=None, *, cores=None, **options):
        """Decorates the compute kernel's body `fn(c)`. Options: fp32_dest,
        unpack_fp32 (buffers), fidelity, approx, min_fidelity, max_fidelity, exact."""
        unknown = set(options) - COMPUTE_OPTIONS.keys()
        if unknown:
            raise TypeError(f"kohakutt.dsl.metal: unknown compute options {unknown}")
        return self._kernel(
            "compute", Compute, fn, cores, None, {**COMPUTE_OPTIONS, **options}
        )

    def footprint(self):
        """Bytes of L1 per core the buffers and scratch areas take."""
        total = sum(
            as_expr(cb.pages)._static("pages") * cb.page_bytes for cb in self.cbs
        )
        return total + sum(as_expr(s.bytes)._static("bytes") for s in self.scratches)

    def core_count(self, runtime):
        """The core count for runtime values `runtime`: the smallest size class not
        below the cores the split's total fills (cores past the total take no work),
        or the largest class without a split."""
        if self._classes is None:
            self._classes = self.machine.classes(self.size_classes)
        if self.split_of is None:
            return self._classes[-1]
        name, least = self.split_of
        wanted = runtime[name] // max(
            1, runtime[least] if isinstance(least, str) else least
        )
        at = bisect.bisect_left(self._classes, wanted)
        return self._classes[min(at, len(self._classes) - 1)]

    def value_list(self, runtime, cores):
        """The values of `self.values`, in order, for one launch on `cores` cores."""
        full = dict(runtime)
        if self.split_of is not None and not self.strided:
            name = self.split_of[0]
            full[f"{name}_base"], full[f"{name}_extra"] = divmod(runtime[name], cores)
        for name, allowed in self.choices.items():
            if full[name] not in allowed:
                raise ValueError(
                    f"kohakutt.dsl.metal: {self.name}: {name} = {full[name]} is not "
                    f"one of {allowed}"
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
