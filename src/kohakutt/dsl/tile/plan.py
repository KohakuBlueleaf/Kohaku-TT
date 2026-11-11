"""The plan: `design(ir)` fixes per kernel class the form (tiles or rows), the passes
over a row and each vector-unit chain with its register slots; `Plan` adds one call's
sizes: core count, held or re-read rows, L1 and the cost line of the report.
"""

import math
from dataclasses import dataclass, field

import kohakutt.dsl.metal as km
from kohakutt.dsl.tile.ir import BF16, F32, Int, Node, fail

f = km.sfpu
# Destination register slots of one half with fp32 registers.
SLOTS = 4
# Tiles per NoC barrier of a streamed row.
CHUNK = 8
# Most tiles a held row may have (the sized value's limit).
SEG_LIMIT = 256
# Vector-unit instructions per element group (estimates) of each operation.
COST = {
    "add": 1, "sub": 1, "mul": 1, "neg": 1, "abs": 1, "max": 2, "min": 2,
    "relu": 2, "square": 1, "exp": 14, "sigmoid": 16, "silu": 17, "tanh": 16,
    "sqrt": 12, "rsqrt": 12, "recip": 8, "div": 9, "fill": 1, "reduce": 4,
}  # fmt: skip
# Hardware rates (docs/concepts/hardware.md): DRAM, one reading core, the clock.
DRAM_RATE = 512e9
CORE_READ_RATE = 28e9
CLOCK = 1.35e9
TILE_BYTES = {BF16: 2048, F32: 4096}


@dataclass(eq=False)
class Leaf:
    """A register operand: tile `index` (a callable of the chunk context) of the
    operand `source` unpacked as `kind` ("copy", "bcast_row", "bcast_col")."""

    source: object
    kind: str
    offset: Int | None = None


@dataclass(eq=False)
class Fill:
    """A register operand filled with a runtime scalar `scalar`."""

    scalar: object


@dataclass(eq=False)
class Chain:
    """One vector-unit function: operands (Leaf / Fill / "stat") in slots 0 .., the
    function, its result slot; `when` (piece, start) when it is one branch of a cat."""

    operands: list
    fn: object
    out: int = 0
    pairs: bool = False
    ops: dict = field(default_factory=dict)


@dataclass(eq=False)
class Case:
    """The chains of one store under the cat branch `bound` (None: unconditional):
    tile j takes this case when j < bound and no earlier case took it."""

    bound: Int | None
    chain: Chain


@dataclass(eq=False)
class Operand:
    """A tensor view the kernel reads: per unit (`resident` False) or once per core;
    `row`: a tile row (else one tile); `held`: the whole row is kept in L1."""

    node: Node
    row: bool
    resident: bool
    held: bool = False
    random: bool = False
    passes: set = field(default_factory=set)

    @property
    def view(self):
        return self.node.attrs["view"]

    @property
    def tensor(self):
        return self.view.tensor


@dataclass(eq=False)
class Scalar:
    """A runtime scalar operand: a float value, or a size as a float, maybe its
    reciprocal; the launch gives its fp32 bits."""

    value: object
    recip: bool = False

    def key(self):
        v = self.value
        text = v.name if hasattr(v, "name") else v.text()
        return ("1/" if self.recip else "") + text


@dataclass(eq=False)
class Design:
    """A kernel class's plan before sizes: `form` "stream" or "row", the operands,
    the stores' cases (pass 2), the reduction (pass 1) and the statistic chain."""

    ir: object
    form: str
    operands: list
    stores: list
    reduce: object = None
    stat: Chain | None = None
    scalars: dict = field(default_factory=dict)
    checks: list = field(default_factory=list)
    stat_root: Node | None = None
    dest: str = "fp32"


def _is_bf16_exact(node):
    """True when the value is a bf16 load (rounding it to bf16 changes nothing)."""
    if node.kind in ("load", "resident"):
        return node.attrs["view"].tensor.dtype == BF16
    if node.kind == "cast":
        return node.attrs["dtype"] == BF16 or _is_bf16_exact(node.args[0])
    return False


class _Shapes:
    """The form of every node: "tile", "row", "col" (one value per row), "scalar"."""

    def __init__(self, row_form):
        self.row_form, self.of = row_form, {}

    def __call__(self, node):
        known = self.of.get(id(node))
        if known is not None:
            return known
        shape = self._shape(node)
        self.of[id(node)] = shape
        return shape

    def _shape(self, node):
        k = node.kind
        if k in ("load", "resident"):
            return "row" if node.attrs["view"].kind == "row" else "tile"
        if k in ("const", "param", "size"):
            return "scalar"
        args = [self(a) for a in node.args]
        if k == "reduce":
            if args[0] != "row":
                fail("kd.reduce_sum reduces a row: load x.tiles[r, :]", node.line)
            return "col"
        if k == "col":
            if args[0] not in ("col", "tile"):
                fail(".col() broadcasts a reduced value or a tile", node.line)
            return "bcol" if args[0] == "col" else "tile"
        if k in ("slice", "cat"):
            if any(a != "row" for a in args):
                fail("slices and kd.cat take row blocks", node.line)
            return "row"
        if "row" in args or "bcol" in args:
            if "col" in args or "tile" in args:
                fail(
                    "a row block meets a column block: broadcast it with .col()",
                    node.line,
                )
            return "row"
        if "tile" in args:
            if "col" in args:
                fail(
                    "a tile meets a reduced value: broadcast it with .col()", node.line
                )
            return "tile"
        if "col" in args:
            return "col"
        return "scalar"


class _Compiler:
    """Builds the vector-unit functions of a kernel class."""

    def __init__(self, name, shapes, scalars):
        self.name, self.shapes, self.scalars = name, shapes, scalars
        self.count = 0

    def scalar(self, node, recip=False):
        """The Scalar of runtime scalar node `node`, registered once."""
        value = node.attrs["value"]
        s = Scalar(value, recip)
        return self.scalars.setdefault(s.key(), s)

    def chain(self, root, leaf_of, cut=None):
        """The Chain computing `root`; `leaf_of(node, offset)` gives the operand of a
        load (or None); `cut` maps a cat node to the piece index taken."""
        operands, keys, ops = [], {}, {}

        def operand(item, key):
            if key not in keys:
                if len(operands) == SLOTS:
                    fail(f"the chain needs more than {SLOTS} register slots", root.line)
                keys[key] = f.operand(len(operands))
                operands.append(item)
            return keys[key]

        def leaf(node, offset, kind):
            got = leaf_of(node, offset, kind)
            if got is None:
                fail(f"{node.kind} cannot be read here", node.line)
            key = (id(got.source), kind, None if offset is None else offset.text())
            return operand(got, key)

        def build(node, offset=None, kind="copy"):
            k = node.kind
            ops[k] = ops.get(k, 0) + 1
            if k in ("load", "resident", "reduce"):
                return leaf(node, offset, kind)
            if k == "const":
                return node.attrs["value"]
            if k in ("param", "size"):
                s = self.scalar(node)
                return operand(Fill(s), ("fill", s.key()))
            if k == "cast":
                inner = node.args[0]
                rounds = node.attrs["dtype"] == BF16 and not _is_bf16_exact(inner)
                if rounds and node is not root:
                    fail(
                        "a cast to bf16 inside a chain is not built; cast at the store",
                        node.line,
                    )
                return build(inner, offset, kind)
            if k == "row":
                return build(node.args[0], offset, "bcast_row")
            if k == "col":
                if self.shapes(node.args[0]) == "col":
                    return leaf(node.args[0], None, "bcast_col")
                return build(node.args[0], None, "bcast_col")
            if k == "slice":
                lo = node.attrs["lo"]
                shift = _tiles(lo, node, self.checks)
                total = shift if offset is None else offset + shift
                return build(node.args[0], total, kind)
            if k == "cat":
                if cut is None or id(node) not in cut:
                    fail("one kd.cat per stored value", node.line)
                piece, start = cut[id(node)]
                zero = Int("const", value=0)
                shifted = (
                    None
                    if start is None
                    else ((zero - start) if offset is None else offset - start)
                )
                return build(node.args[piece], shifted, kind)
            if k in ("even", "odd"):
                inner = build(node.args[0], offset, kind)
                return inner.even if k == "even" else inner.odd
            if k == "div" and self.shapes(node.args[1]) == "scalar":
                a, b = build(node.args[0], offset, kind), node.args[1]
                if b.kind == "const":
                    return a * (1.0 / b.attrs["value"])
                s = self.scalar(b, recip=True)
                return a * operand(Fill(s), ("fill", s.key()))
            args = [build(a, offset, kind) for a in node.args]
            if k == "div":
                return args[0] * f.recip(args[1])
            return _VECTOR[k](*args)

        self.checks = []
        result = build(root)
        pairs = isinstance(result, tuple)
        if pairs:
            result = {"even": result[0], "odd": result[1]}
        self.count += 1
        try:
            fn = km.sfpu_fn(f"{self.name}_fn{self.count}", result)
        except ValueError as e:
            fail(str(e).replace("kohakutt.dsl.metal: ", ""), root.line)
        chain = Chain(operands, fn, 0, pairs, ops)
        chain.checks = self.checks
        return chain


_VECTOR = {
    "neg": lambda a: -a,
    "abs": abs,
    "exp": f.exp,
    "sigmoid": f.sigmoid,
    "silu": f.silu,
    "tanh": f.tanh,
    "sqrt": f.sqrt,
    "rsqrt": f.rsqrt,
    "recip": f.recip,
    "relu": f.relu,
    "square": f.square,
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "max": f.maximum,
    "min": f.minimum,
    "pairs": lambda a, b: (a, b),
}


def _tiles(elements, node, checks):
    """`elements` (a column offset) in tiles; the launch checks it is a multiple of
    32."""
    if elements.kind == "const":
        if elements.value % 32:
            fail("a slice of a row starts at a multiple of 32 columns", node.line)
        return Int("const", value=elements.value // 32)
    checks.append(("multiple", elements, 32, node.line))
    return elements // 32


def _width(node, shapes):
    """The tile count of row block `node` as an Int."""
    k = node.kind
    if k in ("load", "resident"):
        name = node.attrs["view"].tensor.name
        return Int("size", value=("row_tiles", name, None))
    if k == "slice":
        lo, hi = node.attrs["lo"], node.attrs["hi"]
        whole = _width(node.args[0], shapes)
        top = whole if hi is None else hi // 32
        return top - lo // 32
    if k == "cat":
        out = _width(node.args[0], shapes)
        for a in node.args[1:]:
            out = out + _width(a, shapes)
        return out
    rows = [a for a in node.args if shapes(a) == "row"]
    return _width(rows[0], shapes)


def _cats(node, found):
    if node.kind == "cat" and node not in found:
        found.append(node)
    for a in node.args:
        _cats(a, found)
    return found


def design(ir, dest=None):
    """The Design of traced kernel class `ir`; `dest` "fp32" or "bf16" fixes the
    destination register format (None: fp32 when a reduction, an fp32 tensor or a
    cast to fp32 asks for it, else bf16)."""
    if ir.grid is None or not ir.stores:
        fail(f"{ir.name}: a kernel has a kd.grid loop with a kd.store in it")
    nodes = ir.nodes()
    loads = [n for n in nodes if n.kind in ("load", "resident")]
    rows = {n.attrs["view"].kind == "row" for n in loads if n.kind == "load"}
    rows |= {s.view.kind == "row" for s in ir.stores}
    if len(rows) > 1:
        fail(
            f"{ir.name}: loads and stores are all tiles x.tiles[t] or all rows "
            "x.tiles[r, :]"
        )
    form = "row" if rows == {True} else "stream"
    shapes = _Shapes(form == "row")
    for n in nodes:
        shapes(n)
    operands = {
        id(n): Operand(n, shapes(n) == "row", n.kind == "resident") for n in loads
    }
    want = "row" if form == "row" else "tile"
    for s in ir.stores:
        if not s.view.index.device:
            fail("a store's view depends on the unit", s.line)
        if shapes(s.value) != want:
            fail(f"a stored value is a {want} block of the unit", s.line)
    reduces = [n for n in nodes if n.kind == "reduce"]
    out = Design(ir, form, list(operands.values()), [])
    compiler = _Compiler(ir.name, shapes, out.scalars)
    if form == "stream" and reduces:
        fail("a reduction needs a row: load x.tiles[r, :]", reduces[0].line)
    if len(reduces) > 1:
        fail(
            "stage 1 reduces once per kernel (the exact row statistic)", reduces[1].line
        )
    if reduces:
        out.reduce = _reduce(reduces[0], operands)
        _stat(out, compiler, shapes, nodes, reduces[0])

    def leaf_of(node, offset, kind):
        if shapes(node) == "col":
            return Leaf("stat", "bcast_col") if node is out.stat_root else None
        op = operands[id(node)]
        op.passes.add(2)
        if offset is not None:
            op.held = op.random = True
        if kind == "bcast_col" and op.row:
            fail(".col() of a row block", node.line)
        return Leaf(op, kind, offset)

    for s in ir.stores:
        cats = _cats(s.value, [])
        if len(cats) > 1:
            fail("one kd.cat per stored value", s.line)
        cases = []
        if cats:
            cat, start = cats[0], Int("const", value=0)
            for i, piece in enumerate(cat.args):
                width = _width(piece, shapes)
                end = start + width
                last = i == len(cat.args) - 1
                chain = compiler.chain(
                    s.value, leaf_of, {id(cat): (i, start if i else None)}
                )
                cases.append(Case(None if last else end, chain))
                out.checks += chain.checks
                start = end
            out.checks.append(("equal", start, _store_width(s), s.line))
        else:
            chain = compiler.chain(s.value, leaf_of)
            out.checks += chain.checks
            cases.append(Case(None, chain))
        out.stores.append((s, cases))
    if out.reduce is not None:
        out.reduce["operand"].held = True
    for op in out.operands:
        if op.row and not op.resident:
            out.checks.append(("equal", _width(op.node, shapes), _row_tiles(ir), None))
    chains = [c.chain for _, cases in out.stores for c in cases]
    chains += [out.stat] if out.stat is not None else []
    wants = bool(reduces) or any(t.dtype == F32 for t in ir.tensors)
    wants |= any(n.kind == "cast" and n.attrs["dtype"] == F32 for n in nodes)
    wants |= any(c.fn.fp32 for c in chains)
    if dest == "bf16" and wants:
        fail(
            f"{ir.name}: an fp32 tensor, cast, reduction or function needs fp32 "
            "registers"
        )
    out.dest = dest or ("fp32" if wants else "bf16")
    return out


def _store_width(store):
    return Int("size", value=("row_tiles", store.view.tensor.name, None))


def _row_tiles(ir):
    return _store_width(ir.stores[0])


def _reduce(node, operands):
    """The exact row statistic: the sum of squares of a loaded row (stage 1)."""
    e = node.args[0]
    base = None
    if e.kind == "square" or e.kind == "mul" and e.args[0] is e.args[1]:
        base = e.args[0]
    while base is not None and base.kind == "cast" and base.attrs["dtype"] == F32:
        base = base.args[0]
    if base is None or base.kind != "load":
        fail(
            "the exact reduction of stage 1 is the sum of squares of a loaded row "
            "(kd.reduce_sum(row * row, exact=True)); other sums wait for a metal "
            "construct",
            node.line,
        )
    if base.attrs.get("pad") is None:
        fail("a reduction needs the row's pad declared: kd.load(..., pad=0)", base.line)
    op = operands[id(base)]
    op.passes.add(1)
    return {"node": node, "operand": op}


def _stat(out, compiler, shapes, nodes, reduce):
    """The chain on the row statistic: the one column value read through .col()."""
    users = [n for n in nodes if n.kind == "col" and shapes(n.args[0]) == "col"]
    roots = {id(n.args[0]): n.args[0] for n in users}
    if len(roots) != 1:
        fail("stage 1 broadcasts one reduced value per kernel with .col()", reduce.line)
    (root,) = roots.values()

    def leaf_of(node, offset, kind):
        return Leaf("var", "stat") if node is reduce else None

    out.stat = compiler.chain(root, leaf_of)
    out.checks += out.stat.checks
    out.stat_root = root


def vector_cost(chain):
    """Estimated vector-unit cycles per tile of `chain` (32 cycles per instruction
    on a tile, plus a load per operand and a store)."""
    ops = sum(COST.get(k, 0) * n for k, n in chain.ops.items())
    return 32 * (ops + len(chain.operands) + 1)


@dataclass
class Plan:
    """One call's plan: sizes, cores, how rows are kept, L1, and the cost line."""

    design: Design
    env: dict
    units: int
    cores: int
    block: int
    mode: str
    l1: int
    l1_free: int | None
    bytes_read: int
    bytes_written: int
    vector_cycles: float
    program: object = None

    def bounds(self):
        """{name: seconds} of each lower bound on the kernel's time."""
        per_core = math.ceil(self.units / self.cores) / max(1, self.units)
        out = {
            "DRAM": (self.bytes_read + self.bytes_written) / DRAM_RATE,
            "per-core reads (estimate)": self.bytes_read * per_core / CORE_READ_RATE,
            "vector unit (estimate)": self.vector_cycles * per_core / CLOCK,
        }
        return out

    def report(self):
        """The plan report: what was chosen and what bounds it."""
        d = self.design
        ir = d.ir
        lines = [f"kernel    {ir.name} ({d.form} form, {d.dest} registers)"]
        what = "tile rows" if d.form == "row" else "tiles"
        lines.append(
            f"grid      {self.units} {what} over {self.cores} cores "
            f"({math.ceil(self.units / self.cores)} per core), block {self.block}"
        )
        for op in d.operands:
            t = op.tensor
            how = (
                "resident, read once per core"
                if op.resident
                else (
                    "row held in L1"
                    if op.held and self.mode == "hold"
                    else (
                        "row read twice (second DRAM read)"
                        if op.held
                        else "streamed" if op.row else "per tile"
                    )
                )
            )
            lines.append(f"stream    {t.name} ({t.dtype}) {op.view.text()}: {how}")
        for s, cases in d.stores:
            lines.append(f"store     {s.view.text()} ({s.view.tensor.dtype})")
            for case in cases:
                c = case.chain
                lines.append(
                    f"  vector  {c.fn.name}: {len(c.operands)} operands, "
                    f"setup {c.fn.init}"
                )
        if d.reduce is not None:
            lines.append(
                "pass 1    exact sum of squares (vector unit, fp32), "
                f"statistic chain {d.stat.fn.name}"
            )
        free = "unknown" if self.l1_free is None else f"{self.l1_free // 1024} KB"
        lines.append(f"L1        {self.l1 // 1024} KB of buffers per core, free {free}")
        moved = self.bytes_read + self.bytes_written
        lines.append(
            f"moved     {moved / 1e6:.2f} MB ({self.bytes_read / 1e6:.2f} read, "
            f"{self.bytes_written / 1e6:.2f} written), multiply work 0 "
            "(matrix unit), launches 1"
        )
        bounds = self.bounds()
        shown = ", ".join(f"{k} {v * 1e6:.1f} us" for k, v in bounds.items())
        lines.append(f"bound     {shown}")
        lines.append(f"binds     {max(bounds, key=bounds.get)}")
        return "\n".join(lines)

    __str__ = report
