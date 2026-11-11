"""The tile-level syntax: the functions and proxy values a kernel body calls while it
is traced. Each call records a node of the tile IR (kohakutt.dsl.tile.ir); nothing
runs on a device here.
"""

from kohakutt.dsl.tile.ir import (
    BF16,
    F32,
    Block,
    Grid,
    Int,
    Node,
    Range,
    Store,
    TensorParam,
    ValueParam,
    View,
    as_int,
    caller_line,
    fail,
    imin,
)

bf16, f32 = BF16, F32
_STACK = []


def _ir():
    if not _STACK:
        fail("kd.load, kd.store and kd.grid belong inside a @kd.kernel body")
    return _STACK[-1]


class Tensor:
    """Annotation of an input tensor of a kernel."""


class Out:
    """Annotation of an output tensor: its sizes and dtype are those of the first
    input unless the call names `out_dtype`."""


class Runtime:
    """Annotation of a runtime value, ``kd.Runtime[float]`` or ``kd.Runtime[int]``: a
    new value compiles nothing."""

    def __class_getitem__(cls, item):
        return _Typed(cls, item)


class Static:
    """Annotation of a value fixed per kernel class, ``kd.Static[int]``."""

    def __class_getitem__(cls, item):
        return _Typed(cls, item)


class _Typed:
    def __init__(self, kind, item):
        self.kind, self.item = kind, item


class Auto:
    """Candidates of a static value: the first whose program verifies and fits L1."""

    def __init__(self, *candidates):
        if not candidates:
            raise ValueError("kohakutt.dsl: auto() needs a candidate")
        self.candidates = candidates

    def __repr__(self):
        return f"auto{self.candidates}"


def auto(*candidates):
    """A static default tried in the order of `candidates`."""
    return Auto(*candidates)


class _Tiles:
    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, at):
        if isinstance(at, slice):
            if at.start is None or at.stop is None:
                fail("a run of tiles is x.tiles[a:b] or x.tiles[a:b:step]")
            lo, hi = as_int(at.start), as_int(at.stop)
            step = None if at.step in (None, 1) else as_int(at.step)
            return View(self.tensor, "flat", lo, (lo, hi), step)
        if not isinstance(at, tuple):
            return View(self.tensor, "tile", as_int(at))
        if len(at) != 2:
            fail(
                "a view is x.tiles[t], x.tiles[r, :], x.tiles[r, c] or x.tiles[r, a:b]"
            )
        row, col = at
        if isinstance(row, slice) and col == slice(None):
            if row.start is None or row.stop is None or row.step not in (None, 1):
                fail("a block of tile rows is x.tiles[a:b, :]")
            lo, hi = as_int(row.start), as_int(row.stop)
            return View(self.tensor, "rows", lo, (lo, hi))
        if col == slice(None):
            return View(self.tensor, "row", as_int(row))
        if isinstance(col, slice):
            if col.step not in (None, 1) or col.start is None or col.stop is None:
                fail("a column range of tiles is a:b with no step")
            cols = (as_int(col.start), as_int(col.stop))
            return View(self.tensor, "cols", as_int(row), cols)
        return View(self.tensor, "at", as_int(row), (as_int(col), as_int(col) + 1))


class TensorProxy:
    """A tensor while the kernel is traced: `x.tiles[...]`, `x.dtype`."""

    def __init__(self, param):
        self.param = param
        self.tiles = _Tiles(param)

    @property
    def dtype(self):
        """The tensor's dtype in this kernel class."""
        return self.param.dtype

    @property
    def name(self):
        return self.param.name


def _tensor(x):
    if not isinstance(x, TensorProxy):
        fail(f"{x!r} is not a tensor of the kernel")
    return x.param


def num_tiles(x, dim=None):
    """Tiles of `x` (the last two dims padded to 32): all of them, or along `dim`
    (-1: tiles per row, -2: tile rows over every leading dim)."""
    if dim is None:
        return Int("size", value=("tiles", _tensor(x).name, None))
    if dim in (-1, 1):
        return Int("size", value=("row_tiles", _tensor(x).name, None))
    if dim in (-2, 0):
        return num_tile_rows(x)
    fail("num_tiles takes dim -1 or -2")


def num_tile_rows(x):
    """Tile rows of `x`: 32 rows of the last dim's width, leading dims folded in."""
    return Int("size", value=("tile_rows", _tensor(x).name, None))


def size(x, dim):
    """Size `dim` of tensor `x`: a runtime value."""
    return Int("size", value=("size", _tensor(x).name, int(dim)))


def cdiv(a, b):
    """(a + b - 1) // b."""
    return (as_int(a) + as_int(b) - 1) // as_int(b)


def grid(count, block=1, group=1):
    """Units 0 .. count - 1 split over cores, `block` at a time: `for t in
    kd.grid(n):`. `group=g`: each unit (a tile row) runs on g cores, each taking a
    slice of the row's tiles. The body runs once, with the unit as a symbolic value."""
    ir = _ir()
    if ir.grid is not None:
        fail("one kd.grid per kernel")
    for name, value in (("block", block), ("group", group)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            fail(f"a grid's {name} is a static int >= 1, not {value!r}")
    ir.grid = Grid(as_int(count), block, line=caller_line())
    ir.grid.group = group
    yield ir.grid.unit


class Grid2d:
    """Units (i, j) of `rows` x `cols`: `for i, j in g.units():`."""

    def __init__(self, ir):
        self.ir = ir

    def units(self):
        """Yields the unit (row i, column j) once, as symbolic values."""
        yield Int("unit", value="i"), Int("unit", value="j")

    @property
    def row(self):
        """The units of one grid row (i fixed): `share=g.row`."""
        return "row"

    @property
    def col(self):
        """The units of one grid column (j fixed): `share=g.col`."""
        return "col"


def grid2d(rows, cols):
    """A 2-D grid of units (i, j): `for i, j in kd.grid2d(rows=R, cols=C).units():`.
    Units sharing a column j run on the same cores, which keep j's weights."""
    ir = _ir()
    if ir.grid is not None:
        fail("one kd.grid per kernel")
    ir.grid = Grid(as_int(rows), 1, line=caller_line())
    ir.grid.cols = as_int(cols)
    return Grid2d(ir)


def range(count):
    """A device loop `for k in kd.range(n):` (n a runtime integer); its body runs
    once while tracing, with `k` symbolic."""
    ir = _ir()
    if ir.grid is None or ir.loop is not None:
        fail("kd.range sits inside the grid loop, one level deep")
    var = Int("loop", value="k")
    ir.loop = Range(as_int(count), var, caller_line())
    yield var


class Accumulator(Block):
    """A sum kept in the registers across a kd.range loop: `acc += a @ b`."""

    __slots__ = ("shape",)

    def __init__(self, shape):
        super().__init__(Node("acc", (), {"shape": tuple(shape)}))
        self.shape = tuple(shape)

    def __iadd__(self, value):
        ir = _ir()
        node = _node(value)
        if node.kind != "matmul":
            fail("an accumulator adds products: acc += a @ b")
        if ir.loop is None:
            fail("acc += a @ b sits inside a kd.range loop")
        if ir.accumulate is not None:
            fail("one accumulation per kernel")
        ir.accumulate = (self.node, node)
        return self


def accumulator(shape, dtype=F32):
    """A (rows, cols) block of fp32 sums in the destination registers."""
    if dtype != F32:
        fail("an accumulator is fp32")
    if len(shape) != 2 or shape[0] != 1 or not 1 <= shape[1] <= 4:
        fail("accumulators are (1, n) blocks with n <= 4 (the fp32 registers)")
    return Accumulator(shape)


def load(view, pad=None, share=None):
    """A block read from DRAM: `view` is x.tiles[t] or x.tiles[r, :]; `pad` declares
    the value past a ragged edge (reductions need it); `share=g.col` (or `g.row`):
    the units of a grid column (row) take the block one core read and multicast."""
    _check_view(view)
    if view.tensor.output:
        fail(f"{view.tensor.name} is an output; it is stored, not loaded")
    if share not in (None, "row", "col"):
        fail("share= is g.row or g.col of the kernel's kd.grid2d")
    ir = _ir()
    if share is not None and ir.grid is not None and ir.grid.cols is None:
        fail("share= names the cores of a kd.grid2d row or column")
    return Block(Node("load", (), {"view": view, "pad": pad, "share": share}))


def resident(view):
    """A block read once per core and kept for the whole kernel (a weight row)."""
    _check_view(view)
    if view.index.device:
        fail("a resident view is the same on every unit (no unit in its index)")
    return Block(Node("resident", (), {"view": view}))


def store(view, value):
    """Writes block `value` to `view` of an output tensor."""
    _check_view(view)
    if not view.tensor.output:
        fail(f"{view.tensor.name} is an input; a kernel stores into kd.Out tensors")
    ir = _ir()
    if ir.grid is None:
        fail("kd.store belongs inside the kd.grid loop")
    ir.stores.append(Store(view, _node(value), caller_line()))


def _check_view(view):
    if not isinstance(view, View):
        fail(f"{view!r} is not a view: write x.tiles[...]")


def _node(value):
    """`value` (a Block, a number, a runtime value or an Int size) as a Node."""
    if isinstance(value, Block):
        return value.node
    if isinstance(value, ValueParam):
        if value.kind != "float":
            return Node("size", (), {"value": Int("param", value=value.name)})
        return Node("param", (), {"value": value})
    if isinstance(value, Int):
        if value.device:
            fail("a block operand is a host value, not one of the unit")
        return Node("size", (), {"value": value})
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return Node("const", (), {"value": float(value)})
    fail(f"{value!r} is not a block, a number or a runtime value")


def _op(kind, *args, **attrs):
    return Block(Node(kind, [_node(a) for a in args], attrs))


def _binary(kind, swap=False):
    def method(self, other):
        if isinstance(other, TensorProxy):
            return NotImplemented
        return _op(kind, other, self) if swap else _op(kind, self, other)

    return method


def _slice(self, at):
    """`blk[..., lo:hi]` or `blk[:, lo:hi]`: columns lo .. hi - 1 of a row block."""
    if (
        not isinstance(at, tuple)
        or len(at) != 2
        or at[0] not in (Ellipsis, slice(None))
    ):
        fail("a block slice is blk[..., lo:hi] along the last dim")
    cut = at[1]
    if not isinstance(cut, slice) or cut.step not in (None, 1):
        fail("a block slice is blk[..., lo:hi] with no step")
    lo = as_int(0 if cut.start is None else cut.start)
    hi = None if cut.stop is None else as_int(cut.stop)
    return Block(Node("slice", [self.node], {"lo": lo, "hi": hi}))


Block.__add__ = _binary("add")
Block.__radd__ = _binary("add", True)
Block.__sub__ = _binary("sub")
Block.__rsub__ = _binary("sub", True)
Block.__mul__ = _binary("mul")
Block.__rmul__ = _binary("mul", True)
Block.__truediv__ = _binary("div")
Block.__matmul__ = _binary("matmul")
Block.__rtruediv__ = _binary("div", True)
Block.__neg__ = lambda self: _op("neg", self)
Block.__abs__ = lambda self: _op("abs", self)
Block.__getitem__ = _slice
Block.__bool__ = lambda self: fail(
    "a block has no Python truth value: use kd.where / kd.maximum"
)
Block.cast = lambda self, dtype: _op("cast", self, dtype=_dtype(dtype))
Block.all = lambda self: _op("all", self)
Block.all.__doc__ = "A per-unit value (a whole-block reduction) over every element."
Block.each = lambda self, k: _op("each", self, k=as_int(k))
Block.each.__doc__ = (
    "Tile j of the stored run reads tile j // k of this block (per-channel values)."
)
Block.T = property(lambda self: _op("transpose", self))
Block.T.__doc__ = "The block transposed: the right operand of `q @ k.T`."
Block.row = lambda self: _op("row", self)
Block.col = lambda self: _op("col", self)
Block.tile = lambda self, k: _op("tile", self, index=as_int(k))
Block.tile.__doc__ = (
    "Tile `k` of a row block (k a kd.range variable): a matmul operand."
)
Block.even = property(lambda self: _op("even", self))
Block.odd = property(lambda self: _op("odd", self))
Block.row.__doc__ = "The block's first row repeated over its rows (a row broadcast)."
Block.col.__doc__ = "The block's first column repeated over its columns."


def _dtype(dtype):
    if dtype not in (BF16, F32):
        fail(f"dtype {dtype!r}: the tile level takes kd.bf16 and kd.f32")
    return dtype


def _unary(kind, doc):
    def fn(x):
        return _op(kind, x)

    fn.__name__, fn.__doc__ = kind, doc
    return fn


exp = _unary("exp", "e^x on the vector unit.")
sigmoid = _unary("sigmoid", "1 / (1 + e^-x) on the vector unit.")
silu = _unary("silu", "x * sigmoid(x) on the vector unit.")
tanh = _unary("tanh", "tanh(x) on the vector unit (fp32 registers).")
sqrt = _unary("sqrt", "The square root on the vector unit.")
rsqrt = _unary("rsqrt", "1 / sqrt(x) on the vector unit.")
reciprocal = _unary("recip", "1 / x on the vector unit.")
relu = _unary("relu", "max(x, 0).")
square = _unary("square", "x * x.")


def maximum(a, b):
    """The lanewise larger of `a` and `b`."""
    return _op("max", a, b)


def minimum(a, b):
    """The lanewise smaller of blocks `a` and `b`, or the smaller of two integers of
    the kernel (an Int: e.g. a block's row count, `kd.minimum(B, n - j * B)`)."""
    if all(isinstance(x, (int, Int)) and not isinstance(x, bool) for x in (a, b)):
        return imin(as_int(a), as_int(b))
    return _op("min", a, b)


class Carry(Block):
    """A value carried across the iterations of a kd.range loop: it reads as the
    previous iteration's value (its `init` before the first); `c.next(value)` sets
    the next one. After the loop it reads as the last value."""

    __slots__ = ()

    def next(self, value):
        """Sets the value the next iteration (and the code after the loop) reads."""
        ir = _ir()
        if ir.loop is None:
            fail("carry.next(...) sits inside the kd.range loop")
        if any(c is self.node for c, _ in ir.carries):
            fail("a carried value takes one .next(...) per iteration")
        ir.carries.append((self.node, _node(value)))


def carry(init):
    """A value carried across a kd.range loop, starting at constant `init` in every
    element: `m = kd.carry(-1e30)`, then `m.next(...)` inside the loop."""
    ir = _ir()
    if ir.grid is None or ir.loop is not None:
        fail("kd.carry is made inside the grid loop, before its kd.range")
    if isinstance(init, bool) or not isinstance(init, (int, float)):
        fail("a carry starts at a constant number")
    return Carry(Node("carry", (), {"init": float(init)}))


def mask_scores(s, masks, q, k, keys, causal=0):
    """Score block `s` (query tile rows from head-relative tile row `q`, key tiles
    from `k`) with the scores of keys past `keys` (a ragged last key tile) and, when
    `causal`, of keys after their query set to -1e30: adds one tile of `masks` (see
    kohakutt.dsl.tile.score_masks) per score tile."""
    return Block(
        Node(
            "mask",
            [_node(s)],
            {
                "tensor": _tensor(masks),
                "q": as_int(q),
                "k": as_int(k),
                "keys": as_int(keys),
                "causal": int(causal),
            },
        )
    )


def reduce_sum(x, dim=-1, exact=False):
    """The sum of `x` along its last dim, one value per row (a column block: use
    `.col()` to broadcast it back): on the matrix unit (a loaded row, about 19 bits),
    or with `exact=True` in fp32 on the vector unit (any row block)."""
    if dim not in (-1, 1, -2):
        fail("reduce_sum is along the last dim (dim=-1), or dim=-2 over a column run")
    if dim == -2 and not exact:
        fail("the sum over rows (dim=-2) is exact=True")
    return _op("reduce", x, op="sum", exact=bool(exact), dim=-2 if dim == -2 else -1)


def reduce_max(x, dim=-1):
    """The largest value of loaded row `x` along its last dim, one value per row, on
    the matrix unit (rows of whole tiles: padding is not masked)."""
    if dim not in (-1, 1):
        fail("reduce_max is along the last dim (dim=-1 or 1)")
    return _op("reduce", x, op="max", exact=False)


def cat(blocks, dim=-1):
    """Row blocks side by side along the last dim (tile-aligned pieces)."""
    if dim not in (-1,):
        fail("kd.cat joins blocks along the last dim (dim=-1)")
    blocks = list(blocks)
    if len(blocks) < 2:
        fail("kd.cat takes two blocks or more")
    return _op("cat", *blocks)


def pairs(even, odd):
    """A block of column pairs: `even` in the even columns, `odd` in the odd ones."""
    return _op("pairs", even, odd)


class Tracer:
    """Records one kernel class: `with Tracer(ir):` around the body's call."""

    def __init__(self, ir):
        self.ir = ir

    def __enter__(self):
        _STACK.append(self.ir)
        return self.ir

    def __exit__(self, *exc):
        _STACK.pop()


__all__ = [
    "Auto",
    "Out",
    "Runtime",
    "Static",
    "Tensor",
    "TensorParam",
    "TensorProxy",
    "Tracer",
    "ValueParam",
    "auto",
    "bf16",
    "cat",
    "cdiv",
    "exp",
    "f32",
    "grid",
    "load",
    "maximum",
    "minimum",
    "num_tile_rows",
    "num_tiles",
    "pairs",
    "reciprocal",
    "reduce_max",
    "reduce_sum",
    "relu",
    "resident",
    "rsqrt",
    "sigmoid",
    "silu",
    "size",
    "sqrt",
    "square",
    "store",
    "tanh",
]
