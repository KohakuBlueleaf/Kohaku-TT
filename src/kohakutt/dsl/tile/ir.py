"""The tile IR of one kernel class: its tensors and values, the grid, the loads, the
block expressions and the stores. The Python syntax of `kohakutt.dsl` is one way to
build it; `KernelIR.dump()` prints it.
"""

import math
import os
import sys
from dataclasses import dataclass, field

BF16, F32 = "bfloat16", "float32"
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PlanError(ValueError):
    """A kernel the tile level cannot trace, plan or lower; names the source line."""


def caller_line():
    """`file:line` of the first frame outside the kohakutt.dsl package, or None."""
    frame = sys._getframe(1)
    while frame is not None:
        name = os.path.abspath(frame.f_code.co_filename)
        if not name.startswith(_HERE):
            return f"{os.path.basename(name)}:{frame.f_lineno}"
        frame = frame.f_back
    return None


def fail(message, line=None):
    """Raises PlanError with `message` and the source line (given or the caller's)."""
    line = line or caller_line()
    raise PlanError(f"kohakutt.dsl: {message}" + (f" (at {line})" if line else ""))


@dataclass(eq=False)
class TensorParam:
    """A tensor of the kernel; `dtype` is fixed per kernel class."""

    name: str
    output: bool = False
    dtype: str | None = None


@dataclass(eq=False)
class ValueParam:
    """A runtime value of the kernel ("float" or "int"): one argument, no recompile."""

    name: str
    kind: str = "float"


def _shape_tiles(shape):
    """(tile rows, tiles per row) of a tensor of `shape`, the last two dims padded."""
    shape = [1, 1, *shape][-max(2, len(shape)) :]
    lead = math.prod(int(n) for n in shape[:-2])
    return lead * -(-int(shape[-2]) // 32), -(-int(shape[-1]) // 32)


class Int:
    """An integer of the kernel: a constant, the grid's unit, a size of a tensor, a
    runtime int, or arithmetic on them. Without the unit it is known on the host."""

    def __init__(self, kind, args=(), value=None):
        self.kind, self.args, self.value = kind, tuple(args), value

    def _op(self, op, other, swap=False):
        other = as_int(other)
        a, b = (other, self) if swap else (self, other)
        if a.kind == "const" and b.kind == "const":
            return Int("const", value=_FOLD[op](a.value, b.value))
        return Int("op", (a, b), op)

    def __add__(self, other):
        return self._op("+", other)

    def __radd__(self, other):
        return self._op("+", other, True)

    def __sub__(self, other):
        return self._op("-", other)

    def __rsub__(self, other):
        return self._op("-", other, True)

    def __mul__(self, other):
        if isinstance(other, Block):
            return NotImplemented
        return self._op("*", other)

    def __rmul__(self, other):
        return self._op("*", other, True)

    def __floordiv__(self, other):
        return self._op("//", other)

    def __mod__(self, other):
        return self._op("%", other)

    def __truediv__(self, other):
        return NotImplemented

    def __bool__(self):
        fail(
            "a runtime size has no Python truth value: make the value static, or "
            "write the kernel for every size"
        )

    __lt__ = __le__ = __gt__ = __ge__ = lambda self, other: self.__bool__()

    @property
    def device(self):
        """True when the value depends on the grid's unit (known on the device only)."""
        return self.kind in ("unit", "loop") or any(a.device for a in self.args)

    def eval(self, env):
        """Its value on the host: `env` {tensor name: shape, value name: int}."""
        if self.kind == "const":
            return self.value
        if self.kind == "param":
            return int(env[self.value])
        if self.kind == "size":
            what, name, dim = self.value
            shape = env[name]
            if what == "size":
                return int(shape[dim])
            rows, wt = _shape_tiles(shape)
            return {"tiles": rows * wt, "tile_rows": rows, "row_tiles": wt}[what]
        if self.kind == "op":
            a, b = (x.eval(env) for x in self.args)
            return _FOLD[self.value](a, b)
        raise ValueError(f"kohakutt.dsl: {self.text()} is known on the device only")

    def text(self):
        """The value written out."""
        if self.kind == "const":
            return str(self.value)
        if self.kind == "unit":
            return "unit" if self.value is None else self.value
        if self.kind == "loop":
            return self.value
        if self.kind == "param":
            return self.value
        if self.kind == "size":
            what, name, dim = self.value
            return f"{what}({name})" if dim is None else f"{what}({name}, {dim})"
        a, b = (x.text() for x in self.args)
        if self.value == "min":
            return f"min({a}, {b})"
        return f"({a} {self.value} {b})"

    def leaves(self):
        """The size and param leaves it reads."""
        if self.kind in ("size", "param"):
            return [self]
        return [leaf for a in self.args for leaf in a.leaves()]


_FOLD = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "//": lambda a, b: a // b,
    "%": lambda a, b: a % b,
    "min": min,
}


def imin(a, b):
    """The smaller of Ints `a` and `b`."""
    if a.kind == "const" and b.kind == "const":
        return Int("const", value=min(a.value, b.value))
    return Int("op", (a, b), "min")


def as_int(value):
    """`value` (an int or an Int) as an Int."""
    if isinstance(value, Int):
        return value
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"not an integer of the kernel: {value!r}")
    return Int("const", value=value)


@dataclass(eq=False)
class View:
    """Tiles of `tensor`: one tile at flat tile index `index` ("tile"), the whole tile
    row `index` ("row"), or tiles `cols` (lo, hi) of that row ("cols", "at")."""

    tensor: TensorParam
    kind: str
    index: Int
    cols: tuple | None = None
    # A run's stride in tiles ("flat"; None: 1).
    step: object = None

    def text(self):
        at, name = self.index.text(), self.tensor.name
        if self.kind == "tile":
            return f"{name}.tiles[{at}]"
        if self.kind == "row":
            return f"{name}.tiles[{at}, :]"
        if self.kind == "flat":
            lo, hi = (c.text() for c in self.cols)
            return f"{name}.tiles[{lo}:{hi}]"
        if self.kind == "rows":
            lo, hi = (c.text() for c in self.cols)
            return f"{name}.tiles[{lo}:{hi}, :]"
        lo, hi = (c.text() for c in self.cols)
        return f"{name}.tiles[{at}, {lo}:{hi}]"


# Node kinds with one block operand that run per element on the vector unit.
UNARY = (
    "neg",
    "abs",
    "exp",
    "exp2",
    "sigmoid",
    "silu",
    "tanh",
    "sqrt",
    "rsqrt",
    "recip",
    "relu",
    "square",
)
BINARY = ("add", "sub", "mul", "div", "max", "min")


class Node:
    """A block value: a load or resident view, a scalar, or an operation on blocks.
    `attrs` holds the kind's fields; `line` is where the kernel wrote it."""

    def __init__(self, kind, args=(), attrs=None, line=None):
        self.kind, self.args = kind, tuple(args)
        self.attrs = dict(attrs or {})
        self.line = line or caller_line()

    def text(self, names):
        """The node written out, operands by their names in `names`."""
        args = ", ".join(names[id(a)] for a in self.args)
        shown = {k: v for k, v in self.attrs.items() if k not in ("view",)}
        if "view" in self.attrs:
            shown = {"view": self.attrs["view"].text(), **shown}
        for key, value in shown.items():
            if isinstance(value, Int):
                shown[key] = value.text()
            elif isinstance(value, ValueParam):
                shown[key] = value.name
        extra = ", ".join(f"{k}={v}" for k, v in shown.items())
        return f"{self.kind}({', '.join(x for x in (args, extra) if x)})"


class Block:
    """The Python face of a node while a kernel is traced: operators record
    operations (see kohakutt.dsl.tile.trace)."""

    __slots__ = ("node",)

    def __init__(self, node):
        self.node = node


@dataclass(eq=False)
class Store:
    """`view` = `value`."""

    view: View
    value: Node
    line: str | None = None


@dataclass(eq=False)
class Grid:
    """Units 0 .. count - 1 split over the cores; `block` units per NoC barrier."""

    count: Int
    block: int
    unit: Int = field(default_factory=lambda: Int("unit"))
    line: str | None = None
    cols: Int | None = None
    # Cores per unit: a row split over a group of cores (None: the plan picks).
    group: int | None = None


@dataclass(eq=False)
class Range:
    """A device loop k < count inside the grid."""

    count: Int
    var: Int
    line: str | None = None


@dataclass(eq=False)
class KernelIR:
    """One kernel class: its tensors, runtime values, static values, the grid, the
    stores inside it, and a device loop with its accumulation (`acc += a @ b`)."""

    name: str
    tensors: list
    values: list
    statics: dict
    grid: Grid | None = None
    stores: list = field(default_factory=list)
    loop: Range | None = None
    accumulate: tuple | None = None
    # [(carry node, its next value)] of a kd.range loop.
    carries: list = field(default_factory=list)

    def nodes(self):
        """Every node the stores read, operands before their users."""
        order, seen = [], set()

        def visit(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            for a in node.args:
                visit(a)
            order.append(node)

        if self.accumulate is not None:
            visit(self.accumulate[1])
        for _, value in self.carries:
            visit(value)
        for store in self.stores:
            visit(store.value)
        return order

    def dump(self):
        """The IR as text."""
        lines = [f"kernel {self.name}"]
        lines += [
            f"  {'out' if t.output else 'in '} {t.name}: {t.dtype}"
            for t in self.tensors
        ]
        lines += [f"  runtime {v.name}: {v.kind}" for v in self.values]
        lines += [f"  static {n} = {v!r}" for n, v in self.statics.items()]
        nodes = self.nodes()
        names = {id(node): f"%{i}" for i, node in enumerate(nodes)}
        g = self.grid
        if g is not None and g.cols is not None:
            lines.append(f"  grid i < {g.count.text()}, j < {g.cols.text()}")
        elif g is not None:
            lines.append(f"  grid unit < {g.count.text()}, block {g.block}")
        if self.loop is not None:
            lines.append(f"    range {self.loop.var.text()} < {self.loop.count.text()}")
        lines += [f"    {names[id(n)]} = {n.text(names)}" for n in nodes]
        if self.accumulate is not None:
            acc, product = self.accumulate
            names.setdefault(id(acc), "acc")
            lines.append(f"    acc += {names[id(product)]}")
        for c, value in self.carries:
            lines.append(f"    next {names[id(c)]} = {names[id(value)]}")
        lines += [
            f"    store {s.view.text()} = {names[id(s.value)]}" for s in self.stores
        ]
        return "\n".join(lines)
