"""Kernels over runs of tiles (`x.tiles[a:b]`, e.g. one group of a GroupNorm): each
unit holds its run in L1; `kd.reduce_sum(run, exact=True)` sums every element of the
run in fp32 (padding counted out), `.all()` broadcasts such a per-unit value back, and
`p.each(k)` reads tile j // k of a per-channel block for tile j of the run."""

from dataclasses import dataclass, field

import kohakutt.dsl.metal as km
from kohakutt.dsl.tile.ir import F32, Int, fail
from kohakutt.dsl.tile.lower import _Lowering, _Units
from kohakutt.dsl.tile.plan import Case, Design, Leaf, Operand, _Compiler

# Register slots a summed chain may use (2 and 3 hold the statistics).
SUM_SLOTS = 2


@dataclass(eq=False)
class UnitValue:
    """A per-unit value: "exact" (the fp32 sum of a run chain) or "chain"."""

    node: object
    kind: str
    name: str
    chain: object = None
    tensor: object = None


@dataclass(eq=False)
class FlatDesign(Design):
    """A run kernel: `values` in compute order, the stored runs in `stores`, the run
    tensor `run` (its view's start and length) and the per-channel operands."""

    values: list = field(default_factory=list)
    run: object = None
    length: object = None
    unit_stores: list = field(default_factory=list)
    out_like: dict = field(default_factory=dict)

    def describe(self):
        lines = [f"run       {self.run.view.text()} held in L1 per unit"]
        for v in self.values:
            what = {
                "exact": "exact fp32 sum of the whole run (per-row sums, then rows)",
                "colsum": "exact fp32 sums over the run's rows (one per column)",
            }.get(v.kind, "vector unit on per-unit values")
            lines.append(f"unit      {v.name}: {what}, function {v.chain.fn.name}")
        return lines


def _shape(node, of):
    k = node.kind
    if k in ("load", "resident"):
        return "flat" if node.attrs["view"].kind == "flat" else "other"
    if k in ("const", "param", "size"):
        return "scalar"
    args = [of(a) for a in node.args]
    if k == "reduce":
        if args[0] != "flat":
            fail("a run kernel reduces a run block", node.line)
        return "unit"
    if k == "all":
        if args[0] != "unit":
            fail(".all() broadcasts a per-unit value", node.line)
        return "flat"
    if k == "each":
        if args[0] != "flat":
            fail(".each(k) reads a run block", node.line)
        return "flat"
    if "other" in args:
        fail("a run kernel loads runs x.tiles[a:b] only", node.line)
    if "flat" in args:
        if "unit" in args:
            fail("a run meets a per-unit value: broadcast it with .all()", node.line)
        return "flat"
    return "unit" if "unit" in args else "scalar"


def wants_flat(ir, nodes):
    """True when the kernel loads or stores runs of tiles."""
    views = [n.attrs["view"] for n in nodes if n.kind in ("load", "resident")]
    views += [s.view for s in ir.stores]
    return any(v.kind == "flat" for v in views)


def design_flat(ir, dest=None):
    """The FlatDesign of traced kernel `ir`."""
    if dest == "bf16":
        fail(f"{ir.name}: a run kernel needs fp32 registers")
    nodes = ir.nodes()
    shapes = {}

    def of(node):
        if id(node) not in shapes:
            shapes[id(node)] = _shape(node, of)
        return shapes[id(node)]

    for n in nodes:
        of(n)
    operands, eaches = {}, {}
    for n in nodes:
        if n.kind == "load":
            operands[id(n)] = Operand(n, True, False, held=True)
        elif n.kind == "resident":
            fail("a run kernel has no resident blocks", n.line)
    for n in nodes:
        if n.kind == "each":
            base = n.args[0]
            if base.kind != "load":
                fail(".each(k) applies to a loaded run", n.line)
            eaches[id(n)] = (operands[id(base)], n.attrs["k"])
    if any(s.view.kind not in ("flat", "tile") for s in ir.stores):
        fail(f"{ir.name}: a run kernel stores runs x.tiles[a:b] or tiles x.tiles[u]")
    out = FlatDesign(ir, "flat", list(operands.values()), [])
    out.dest = "fp32"
    per_channel = {id(e[0].node) for e in eaches.values()}
    runs = [op for op in operands.values() if id(op.node) not in per_channel]
    if not runs:
        fail(f"{ir.name}: a run kernel loads a run x.tiles[a:b]")
    out.run = runs[0]
    length = run_length(out.run.view)
    for op in runs:
        out.checks.append(("equal", run_length(op.view), length, op.node.line))
    for op, k in eaches.values():
        op.each = k
        out.checks.append(("equal", run_length(op.view) * k, length, op.node.line))
    for s in ir.stores:
        if s.view.kind == "flat":
            out.checks.append(("equal", run_length(s.view), length, s.line))
    t = out.run.tensor.name
    if any(n.kind == "reduce" and n.attrs.get("dim", -1) == -1 for n in nodes):
        # Padding rows are counted out once per run: H <= 32 or a multiple of 32.
        out.checks.append(("rows", Int("size", value=("size", t, -2)), 32, None))
    out.length = length
    compiler = _Compiler(ir.name, lambda n: _compiler_shape(of(n)), out.scalars)
    made = {}

    def keep(node, kind, chain):
        value = UnitValue(node, kind, f"unit{len(out.values)}", chain)
        made[id(node)] = value
        out.values.append(value)
        return value

    def run_leaf(node, offset, kind):
        if id(node) in eaches:
            return Leaf(eaches[id(node)][0], "copy")
        if of(node) == "unit":
            value = made.get(id(node))
            return None if value is None else Leaf(value, "bcast_col")
        return Leaf(operands[id(node)], kind, offset)

    def unit_leaf(root):
        def leaf_of(node, offset, kind):
            if id(node) in made and (node is not root or node.kind == "reduce"):
                return Leaf(made[id(node)], "copy")
            return None

        return leaf_of

    stops = set(eaches)
    for n in nodes:
        if n.kind == "reduce":
            if not n.attrs["exact"]:
                fail("a run's sum is exact=True (the vector unit)", n.line)
            chain = compiler.chain(n.args[0], run_leaf, stops=stops, limit=SUM_SLOTS)
            out.checks += chain.checks
            dim = n.attrs.get("dim", -1)
            keep(n, "exact" if dim == -1 else "colsum", chain)
        elif n.kind == "all" and id(n.args[0]) not in made:
            inner = n.args[0]
            here = {i for i in made if i != id(inner)}
            chain = compiler.chain(inner, unit_leaf(inner), stops=here)
            keep(inner, "chain", chain)
    for s in ir.stores:
        if s.view.kind == "tile":
            if of(s.value) != "unit":
                fail("a tile store x.tiles[u] of a run kernel takes a per-unit value")
            here = {i for i in made if i != id(s.value)}
            chain = compiler.chain(s.value, unit_leaf(s.value), stops=here)
            out.unit_stores.append((s, chain))
            # The sums over rows of a [.., R, D] run: a [D] output.
            out.out_like[s.view.tensor.name] = ("last", out.run.tensor.name)
            continue
        if of(s.value) != "flat":
            fail("a stored run is a run block", s.line)
        chain = compiler.chain(s.value, run_leaf, stops=stops | set(made))
        out.checks += chain.checks
        out.stores.append((s, [Case(None, chain)]))
    return out


def run_length(view):
    """The tiles of run view `view` (the same for every unit): (b - a) / step."""
    a, b = view.cols
    n = at_unit0(b) - at_unit0(a)
    if view.step is None:
        return n
    return (n + view.step - 1) // view.step


def at_unit0(expr):
    """`expr` with the unit set to 0: a host value (a run's length is the same for
    every unit)."""
    if expr.kind == "unit":
        return Int("const", value=0)
    if expr.kind != "op":
        return expr
    a, b = (at_unit0(x) for x in expr.args)
    return {
        "+": lambda: a + b,
        "-": lambda: a - b,
        "*": lambda: a * b,
        "//": lambda: a // b,
        "%": lambda: a % b,
    }[expr.value]()


def _compiler_shape(shape):
    # The chain compiler asks for "col" (a broadcast value) and "scalar".
    return {"unit": "col", "flat": "tile"}.get(shape, shape)


class _FlatLowering(_Lowering):
    def index(self, leaf, i, j):
        src = leaf.source
        if not isinstance(src, Operand):
            return 0
        k = getattr(src, "each", None)
        at = j if k is None else j // self.v(k)
        return at.bounded(0, self.seg)

    def build(self):
        d, p, v = self.d, self.p, self.v
        units = v.host_value(d.ir.grid.count)
        lo = d.run.view.cols[0]
        length = v.host_value(d.length, "run_tiles")
        t = d.run.tensor.name
        wt = v.host_value(Int("size", value=("row_tiles", t, None)))
        width = v.host_value(Int("size", value=("size", t, -1)))
        height = v.host_value(Int("size", value=("size", t, -2)))
        last = p.runtime("last_cols")
        rows = p.runtime("valid_rows")
        self.last_rows = p.runtime("last_rows")

        def last_rows(env, cores):
            h = env[height.name]
            return h - (-(-h // 32) - 1) * 32

        v.extra = {
            "last_cols": lambda env, cores: env[width.name] - (env[wt.name] - 1) * 32,
            "valid_rows": lambda env, cores: min(env[height.name], 32),
            "last_rows": last_rows,
        }
        self.seg = p.sized("seg", of=length, limit=self.seg_limit)
        p.limit(length, low=1, high=self.seg)
        start, count = p.split(units)
        for op in d.operands:
            self.cbs[id(op)] = p.cb(like=self.ref(op), pages=self.seg)
        for value in d.values:
            self.cbs[id(value)] = p.cb(dtype=F32, pages=1, name=value.name)
        self.lanes = p.cb(dtype=F32, pages=1, name="lanes")
        outs = [
            (s, p.cb(like=self.refs[s.view.tensor.name], pages=2)) for s, _ in d.stores
        ]
        unit_outs = [
            (s, p.cb(like=self.refs[s.view.tensor.name], pages=2))
            for s, _ in d.unit_stores
        ]
        self.wt, self.last, self.rows, self.length = wt, last, rows, length

        def at(view, j):
            return v(view.cols[0]) + (j if view.step is None else j * v(view.step))

        k = p.kernel("dm0")
        with k.loop(count, name="u") as u:
            v.unit = k.let("r", start + u)
            for op in d.operands:
                n = v(run_length(op.view))
                slot = k.reserve(self.cbs[id(op)], self.seg)
                with k.loop(n, name="j") as j:
                    k.read(
                        self.ref(op).page(at(op.view, j)), slot[j.bounded(0, self.seg)]
                    )
            k.reads_done()
            for op in d.operands:
                k.push(self.cbs[id(op)], self.seg)

        c, units_ = self.compute(outs + unit_outs)
        with c.loop(count, name="u") as u:
            v.unit = c.let("r", start + u)
            for op in d.operands:
                c.wait(self.cbs[id(op)], self.seg)
            for value in d.values:
                if value.kind == "exact":
                    self.exact(c, units_, value, v(lo))
                elif value.kind == "colsum":
                    self.colsum(c, units_, value)
                else:
                    self.unit_chain(c, units_, value)
            if outs:
                with c.loop(length, name="j") as j:
                    for (_, cb), (_, cases) in zip(outs, d.stores, strict=True):
                        c.reserve(cb, 1)
                        self.scope(c, units_, cases, None, j, cb, 0)
                        c.push(cb, 1)
            for (_, cb), (_, chain) in zip(unit_outs, d.unit_stores, strict=True):
                c.reserve(cb, 1)
                self.scope(c, units_, [Case(None, chain)], None, None, cb, 0)
                c.push(cb, 1)
            for value in d.values:
                c.pop(self.cbs[id(value)], 1)
            for op in d.operands:
                c.pop(self.cbs[id(op)], self.seg)

        w = p.kernel("dm1")
        with w.loop(count, name="u") as u:
            v.unit = w.let("r", start + u)
            if outs:
                with w.loop(length, name="j") as j:
                    for s, cb in outs:
                        w.wait(cb, 1)
                        ref = self.refs[s.view.tensor.name]
                        w.write(cb, ref.page(at(s.view, j)))
                    w.flushed()
                    for _, cb in outs:
                        w.pop(cb, 1)
            for s, cb in unit_outs:
                w.wait(cb, 1)
                w.write(cb, self.refs[s.view.tensor.name].page(v(s.view.index)))
            if unit_outs:
                w.flushed()
                for _, cb in unit_outs:
                    w.pop(cb, 1)
        w.writes_done()
        return self.p

    def compute(self, outs):
        d, p = self.d, self.p
        cbs = [self.cbs[id(op)] for op in d.operands] + [cb for _, cb in outs]
        cbs += [self.cbs[id(v)] for v in d.values] + [self.lanes]
        c = p.kernel(
            "compute",
            fp32_dest=True,
            unpack_fp32=[cb for cb in cbs if cb.dtype == F32],
            exact=True,
        )
        first = self.cbs[id(d.operands[0])]
        c.startup(first, None, outs[0][1])
        units = _Units(c, first, False)
        units.pack = outs[0][1].dtype
        return c, units

    def exact(self, c, units, value, start):
        """The fp32 sum of the value's run chain: per-row sums over the run's tiles
        (the last tile column counted to the width), then the rows' sums added (the
        valid rows); the total in column 0 of the value's page."""
        c.acquire()
        units.forget()
        tile = self.runner(c, units, value.chain)
        units.setup("stats")
        c.stats_clear()
        c.stats_save(2)
        with c.loop(self.length, name="j") as j:
            units.forget()
            tile(j)
            units.setup("transpose_dest")
            c.op("transpose_dest", 0)
            units.setup("stats")
            c.stats_restore(2)
            edge = ((start + j) % self.wt).eq(self.wt - 1)
            c.stats_sum(0, count=km.select(edge, self.last, 32))
            c.stats_save(2)
        units.forget()
        units.setup("stats")
        c.stats_restore(2)
        c.stats_mean(1.0)
        c.stats_finalize(2, 1.0)
        c.reserve(self.lanes, 1)
        units.pack_into(self.lanes)
        c.pack(2, self.lanes)
        c.release()
        c.push(self.lanes, 1)
        c.wait(self.lanes, 1)
        # Row 0 holds the 32 rows' sums; every row of the broadcast holds them too,
        # so after a transpose each column sums the rows: every lane gets the total.
        cb_value = self.cbs[id(value)]
        c.acquire()
        units.forget()
        units.unpack_as("bcast_row", self.lanes)
        c.bcast(self.lanes, 0, 0, "row")
        units.setup("transpose_dest")
        c.op("transpose_dest", 0)
        units.setup("stats")
        c.stats_clear()
        c.stats_sum(0, count=self.rows)
        c.stats_mean(1.0)
        c.stats_finalize(2, 1.0)
        units.setup("transpose_dest")
        c.op("transpose_dest", 2)
        c.reserve(cb_value, 1)
        units.pack_into(cb_value)
        c.pack(2, cb_value)
        c.release()
        c.pop(self.lanes, 1)
        c.push(cb_value, 1)
        c.wait(cb_value, 1)

    def runner(self, c, units, chain):
        """A function of the run's tile j computing `chain` into slot 0, within slots
        0 and 1: per-unit operands unpacked once here (unless the chain is staged)."""
        if chain.prev is not None:
            return lambda j: self.run(c, units, chain, None, j)
        row_op = [x for x in chain.operands if isinstance(x.source, Operand)]
        fixed = [i for i, x in enumerate(chain.operands) if x not in row_op]
        moving = [i for i, x in enumerate(chain.operands) if x in row_op]
        order = moving + fixed
        slots = [order.index(i) for i in range(len(order))]
        self.unpack(
            c, units, _Some(chain, fixed), None, None, [slots[i] for i in fixed]
        )

        def tile(j):
            self.unpack(
                c, units, _Some(chain, moving), None, j, [slots[i] for i in moving]
            )
            units.setup(chain.fn.init)
            c.run(chain.fn, *slots, out=0, values=self.values(chain))

        return tile

    def colsum(self, c, units, value):
        """The fp32 sums over rows of the value's run chain (a column of tiles): each
        tile's rows added into the statistics lanes (one lane per column, the last
        tile counted to the height); row 0 of the value's page."""
        c.acquire()
        units.forget()
        tile = self.runner(c, units, value.chain)
        units.setup("stats")
        c.stats_clear()
        c.stats_save(2)
        with c.loop(self.length, name="j") as j:
            units.forget()
            tile(j)
            units.setup("stats")
            c.stats_restore(2)
            edge = j.eq(self.length - 1)
            c.stats_sum(0, count=km.select(edge, self.last_rows, 32))
            c.stats_save(2)
        units.forget()
        units.setup("stats")
        c.stats_restore(2)
        c.stats_mean(1.0)
        c.stats_finalize(2, 1.0)
        cb_value = self.cbs[id(value)]
        c.reserve(cb_value, 1)
        units.pack_into(cb_value)
        c.pack(2, cb_value)
        c.release()
        c.push(cb_value, 1)
        c.wait(cb_value, 1)

    def unit_chain(self, c, units, value):
        cb_value = self.cbs[id(value)]
        c.acquire()
        units.forget()
        self.run(c, units, value.chain, None, None)
        c.reserve(cb_value, 1)
        units.pack_into(cb_value)
        c.pack(0, cb_value)
        c.release()
        c.push(cb_value, 1)
        c.wait(cb_value, 1)


class _Some:
    def __init__(self, chain, which):
        self.operands = [chain.operands[i] for i in which]


def lower_flat(design, machine, seg_limit):
    """(km.Program, Values) of run design `design`."""
    low = _FlatLowering(design, machine, 1, "hold", 1)
    low.seg_limit = seg_limit
    return low.build(), low.v
