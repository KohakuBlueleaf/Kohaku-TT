"""Row kernels with column values: per tile row, values with one number per row (row
reductions, column inputs, chains on them) computed in order, each kept in a one-page
buffer, broadcast back into the row chains with `.col()` and stored as column tiles.
"""

from dataclasses import dataclass, field

import kohakutt.dsl.metal as km
from kohakutt.dsl.tile.ir import BF16, F32, Int, fail
from kohakutt.dsl.tile.lower import _Lowering, _Units
from kohakutt.dsl.tile.plan import (
    CHUNK,
    Case,
    Design,
    Leaf,
    Operand,
    _Compiler,
    _Shapes,
    _width,
)

# Register slots a vector-unit sum's chain may use (2 and 3 hold the statistics).
SUM_SLOTS = 2


@dataclass(eq=False)
class ColValue:
    """A column value kept per row: "max" / "sum" on the matrix unit over a loaded
    row, "exact" (vector-unit fp32 sum of a row chain) or "chain" (on columns)."""

    node: object
    kind: str
    name: str
    operand: Operand | None = None
    chain: object = None
    width: object = None


@dataclass(eq=False)
class ColumnDesign(Design):
    """A row kernel with column values: `values` in compute order, `col_stores`
    [(store, chain)], `row_tensor` (the row width's tensor), `out_like`."""

    values: list = field(default_factory=list)
    col_stores: list = field(default_factory=list)
    row_tensor: object = None
    out_like: dict = field(default_factory=dict)
    # {id(exact sum value): index of the row store whose packs it accumulates}
    fused: dict = field(default_factory=dict)
    # (sum, mean, M2) values computed by one statistics scope, or None.
    moments: tuple | None = None

    def describe(self):
        """Report lines: each column value in compute order, then column stores."""
        how = {
            "max": "row max on the matrix unit",
            "sum": "row sum on the matrix unit (about 19 bits)",
            "exact": "exact fp32 row sum on the vector unit",
            "chain": "vector unit on columns",
        }
        lines = []
        for v in self.values:
            what = how[v.kind]
            if self.moments and v is self.moments[0]:
                continue
            if self.moments and v in self.moments[1:]:
                what = (
                    "mean and sum of squared deviations of one statistics scope "
                    "(sums, mean, then M2 about it; the backend norms' calls)"
                )
                lines.append(f"column    {v.name}: {what}")
                continue
            if id(v) in self.fused:
                what = (
                    "fp32 row sum of a stored row: its tiles added by the packer "
                    "while stored, then one row sum"
                )
            if v.operand is not None:
                what += f" over {v.operand.view.text()}"
            if v.chain is not None:
                what += f", function {v.chain.fn.name}"
            lines.append(f"column    {v.name}: {what}")
        for s, chain in self.col_stores:
            lines.append(f"store     {s.view.text()} ({s.view.tensor.dtype}) column")
            lines.append(f"  vector  {chain.fn.name}: {len(chain.operands)} operands")
        return lines


class _ColShapes(_Shapes):
    """Shapes of a row kernel where a per-unit tile load is a column."""

    def _shape(self, node):
        if node.kind == "load" and node.attrs["view"].kind == "tile":
            return "col"
        return super()._shape(node)


def wants_columns(ir, nodes):
    """True when a row kernel needs column values: a matrix-unit reduction, several
    reductions, a sum other than of squares, or a column input or output."""
    loads = [n for n in nodes if n.kind == "load"]
    if not any(n.attrs["view"].kind == "row" for n in loads):
        return False
    reduces = [n for n in nodes if n.kind == "reduce"]
    if len(reduces) > 1 or any(
        n.attrs["op"] != "sum" or not n.attrs["exact"] for n in reduces
    ):
        return True
    if reduces and not _sum_of_squares(reduces[0]):
        return True
    tiles = [n for n in loads if n.attrs["view"].kind == "tile"]
    return bool(tiles) or any(s.view.kind == "tile" for s in ir.stores)


def _sum_of_squares(node):
    e = node.args[0]
    base = None
    if e.kind == "square" or e.kind == "mul" and e.args[0] is e.args[1]:
        base = e.args[0]
    while base is not None and base.kind == "cast" and base.attrs["dtype"] == F32:
        base = base.args[0]
    return base is not None and base.kind == "load"


def design_columns(ir, dest=None):
    """The ColumnDesign of traced row kernel `ir`."""
    if dest == "bf16":
        fail(f"{ir.name}: column values need fp32 registers")
    nodes = ir.nodes()
    shapes = _ColShapes(True)
    for n in nodes:
        shapes(n)
    operands = {}
    for n in nodes:
        if n.kind not in ("load", "resident"):
            continue
        row = shapes(n) == "row"
        if n.kind == "resident" and not row:
            fail("a resident tile in a kernel with column values is not built", n.line)
        if n.kind == "load" and not n.attrs["view"].index.device:
            fail("a loaded view depends on the unit", n.line)
        op = Operand(n, row, n.kind == "resident")
        # Every row stays whole in L1 for the unit: the passes read it again.
        op.held = op.random = row and not op.resident
        operands[id(n)] = op
    rows = [op for op in operands.values() if op.row and not op.resident]
    if not rows:
        fail(f"{ir.name}: a kernel with column values loads a row x.tiles[r, :]")
    out = ColumnDesign(ir, "columns", list(operands.values()), [])
    out.row_tensor = rows[0].tensor
    out.dest = "fp32"
    compiler = _Compiler(ir.name, shapes, out.scalars)
    made = {}

    def keep(node, kind, **extra):
        value = ColValue(node, kind, f"col{len(out.values)}", **extra)
        made[id(node)] = value
        out.values.append(value)
        return value

    def row_leaf(node, offset, kind):
        if shapes(node) == "col":
            if id(node) in operands:
                return Leaf(operands[id(node)], "bcast_col")
            value = made.get(id(node))
            return None if value is None else Leaf(value, "bcast_col")
        return Leaf(operands[id(node)], kind, offset)

    def col_leaf(root):
        def leaf_of(node, offset, kind):
            if id(node) in made and (node is not root or node.kind == "reduce"):
                return Leaf(made[id(node)], "copy")
            op = operands.get(id(node))
            if op is None or op.row:
                return None
            return Leaf(op, "copy")

        return leaf_of

    def col_chain(node):
        stops = {i for i in made if i != id(node)}
        chain = compiler.chain(node, col_leaf(node), stops=stops)
        out.checks += chain.checks
        return chain

    # Values in the order the stores' nodes come (operands before users).
    for n in nodes:
        if n.kind == "reduce":
            _reduction(n, out, operands, shapes, compiler, row_leaf, keep)
        elif n.kind == "col" and shapes(n.args[0]) == "col":
            inner = n.args[0]
            if id(inner) not in made and id(inner) not in operands:
                keep(inner, "chain")
                made[id(inner)].chain = col_chain(inner)
    width = _width(rows[0].node, shapes)
    for op in rows[1:]:
        out.checks.append(("equal", _width(op.node, shapes), width, op.node.line))
    for s in ir.stores:
        if s.view.kind == "row":
            if shapes(s.value) != "row":
                fail("a row store takes a row block", s.line)
            chain = compiler.chain(s.value, row_leaf)
            out.checks += chain.checks
            out.stores.append((s, [Case(None, chain)]))
            store_width = Int("size", value=("row_tiles", s.view.tensor.name, None))
            out.checks.append(("equal", store_width, width, s.line))
        elif s.view.kind == "tile":
            if shapes(s.value) != "col":
                fail("a column store x.tiles[r] takes a column value", s.line)
            out.col_stores.append((s, col_chain(s.value)))
        else:
            fail("a store is x.tiles[r, :] (a row) or x.tiles[r] (a column)", s.line)
        if not s.view.index.device:
            fail("a store's view depends on the unit", s.line)
    cols = [op.tensor.name for op in operands.values() if not op.row]
    for s, _ in out.col_stores:
        out.out_like[s.view.tensor.name] = cols[0] if cols else None
    _fuse(out)
    _moments(out)
    return out


def _moments(out):
    """Finds mean = reduce_sum(r, exact) / D then reduce_sum((r - mean.col())^2,
    exact): both come from one statistics scope (sums, mean, M2 about it), as the
    backend's norms compute them. Sets `out.moments` to (sum, mean, M2) values."""
    by_node = {id(v.node): v for v in out.values}
    chains = [c.chain for _, cases in out.stores for c in cases]
    chains += [ch for _, ch in out.col_stores] + [v.chain for v in out.values]
    for m2 in out.values:
        if m2.kind != "exact" or id(m2) in out.fused:
            continue
        sq = m2.node.args[0]
        if sq.kind == "square" or sq.kind == "mul" and sq.args[0] is sq.args[1]:
            d = sq.args[0]
        else:
            continue
        if d.kind != "sub" or d.args[1].kind != "col":
            continue
        mean = by_node.get(id(d.args[1].args[0]))
        if mean is None or mean.kind != "chain" or len(mean.chain.operands) != 1:
            continue
        total = mean.chain.operands[0].source
        if not isinstance(total, ColValue) or total.kind != "exact":
            continue
        if total.node.args[0] is not d.args[0] or id(total) in out.fused:
            continue
        vals = mean.chain.values
        if (
            len(vals) != 1
            or not vals[0].recip
            or set(mean.chain.ops) - {"reduce", "div"}
        ):
            continue
        readers = [ch for ch in chains if ch is not mean.chain]
        if any(
            getattr(x, "source", None) is total for ch in readers for x in ch.operands
        ):
            continue
        out.moments = (total, mean, m2)
        return


def _fuse(out):
    """Marks each exact sum of a stored row as summed while the row is stored (packer
    accumulation): rows of whole tiles (a matrix-unit reduction checks the width), the
    sum read only by column stores."""
    checked = {
        c[1].value[1] for c in out.checks if c[0] == "multiple" and c[1].kind == "size"
    }
    read = {
        id(leaf.source)
        for chain in [c.chain for _, cases in out.stores for c in cases]
        + [v.chain for v in out.values if v.chain is not None]
        for leaf in chain.operands
    }
    for value in out.values:
        if value.kind != "exact" or id(value) in read or value.width not in checked:
            continue
        arg = value.node.args[0]
        for k, (s, cases) in enumerate(out.stores):
            root = s.value.args[0] if s.value.kind == "cast" else s.value
            if (root is arg or s.value is arg) and len(cases) == 1:
                out.fused[id(value)] = k
                break


def _reduction(node, out, operands, shapes, compiler, row_leaf, keep):
    arg = node.args[0]
    if shapes(arg) != "row":
        fail("a reduction reduces a row block", node.line)
    if not node.attrs["exact"]:
        base = arg
        while base.kind == "cast" and base.attrs["dtype"] == F32:
            base = base.args[0]
        if base.kind not in ("load", "resident"):
            fail(
                "the matrix-unit reduction reads a loaded row; reduce a computed row "
                "with exact=True",
                node.line,
            )
        op = operands[id(base)]
        size = Int("size", value=("size", op.tensor.name, -1))
        # Tile padding is not masked on the matrix unit.
        out.checks.append(("multiple", size, 32, node.line))
        keep(node, node.attrs["op"], operand=op)
        return
    if node.attrs["op"] != "sum":
        fail("the exact reduction is a sum", node.line)
    chain = compiler.chain(arg, row_leaf, limit=SUM_SLOTS)
    out.checks += chain.checks
    stages, c = [], chain
    while c is not None:
        stages.append(c)
        c = c.prev
    if any(len(s.operands) > SUM_SLOTS for s in stages):
        fail(
            f"a summed row chain reads at most {SUM_SLOTS} blocks (two register "
            "slots hold the running sums)",
            node.line,
        )
    first = next(
        leaf.source
        for leaf in chain.operands
        if isinstance(leaf, Leaf) and isinstance(leaf.source, Operand)
    )
    keep(node, "exact", chain=chain, width=first.tensor.name)


class _ColumnLowering(_Lowering):
    def index(self, leaf, i, j):
        if not isinstance(leaf.source, Operand) or not leaf.source.row:
            return 0
        return super().index(leaf, i, j)

    def build(self):
        d, p, v = self.d, self.p, self.v
        units_value = v.host_value(d.ir.grid.count)
        wt = v.host_value(Int("size", value=("row_tiles", d.row_tensor.name, None)))
        self.wt = wt
        self.seg = p.sized("seg", of=wt, limit=self.seg_limit)
        p.limit(wt, low=1, high=self.seg)
        start, count = p.split(units_value)
        for op in d.operands:
            depth = 1 if op.resident else self.depth
            pages = depth * self.seg if op.row else 2
            self.cbs[id(op)] = p.cb(like=self.ref(op), pages=pages)
        # A sum whose mean comes from the statistics scope directly has no buffer.
        self.skip = {id(d.moments[0])} if d.moments else set()
        for value in d.values:
            if id(value) not in self.skip:
                self.cbs[id(value)] = p.cb(dtype=F32, pages=1, name=value.name)
        self.acc = {
            id(value): p.cb(dtype=F32, pages=1, name=f"{value.name}_tiles")
            for value in d.values
            if id(value) in d.fused
        }
        self.ones = None
        if any(value.kind in ("max", "sum") for value in d.values):
            self.ones = p.cb(dtype=BF16, pages=1, name="ones")
        self.lasts = {}
        for value in d.values:
            if value.kind == "exact" and value.width not in self.lasts:
                size = Int("size", value=("size", value.width, -1))
                tiles = Int("size", value=("row_tiles", value.width, None))
                self.lasts[value.width] = v.host_value(
                    size - (tiles - 1) * 32, f"{value.width}_last"
                )
        row_outs = [
            (s, p.cb(like=self.refs[s.view.tensor.name], pages=2 * CHUNK))
            for s, _ in d.stores
        ]
        col_outs = [
            (s, p.cb(like=self.refs[s.view.tensor.name], pages=2))
            for s, _ in d.col_stores
        ]
        once = [op for op in d.operands if op.resident]
        per = [op for op in d.operands if not op.resident]

        k = p.kernel("dm0")
        if once:
            for op in once:
                slot = k.reserve(self.cbs[id(op)], self.seg)
                with k.loop(wt, name="j") as j:
                    page = self._row_page(op, v(op.view.index), j)
                    k.read(page, slot[j.bounded(0, self.seg)])
            k.reads_done()
            for op in once:
                k.push(self.cbs[id(op)], self.seg)
        with k.loop(count, name="u") as u:
            v.unit = k.let("r", start + u)
            for op in per:
                cb = self.cbs[id(op)]
                if op.row:
                    slot = k.reserve(cb, self.seg)
                    with k.loop(wt, name="j") as j:
                        page = self._row_page(op, v(op.view.index), j)
                        k.read(page, slot[j.bounded(0, self.seg)])
                else:
                    slot = k.reserve(cb, 1)
                    k.read(self.ref(op).page(v(op.view.index)), slot)
            k.reads_done()
            for op in per:
                k.push(self.cbs[id(op)], self.seg if op.row else 1)

        outs = row_outs + col_outs
        c, units = self.compute(outs)
        if self.ones is not None:
            c.acquire()
            units.setup("fill")
            c.op("fill", 0, value=1.0)
            c.reserve(self.ones, 1)
            units.pack_into(self.ones)
            c.pack(0, self.ones)
            c.release()
            c.push(self.ones, 1)
            c.wait(self.ones, 1)
        for op in once:
            c.wait(self.cbs[id(op)], self.seg)
        with c.loop(count, name="u"):
            for op in per:
                c.wait(self.cbs[id(op)], self.seg if op.row else 1)
            fused = [v for v in d.values if id(v) in d.fused]
            for value in d.values:
                if d.moments and value is d.moments[1]:
                    self.moments(c, units)
                elif d.moments and value is d.moments[2]:
                    continue
                elif id(value) not in d.fused and id(value) not in self.skip:
                    self.value(c, units, value)
            for value in fused:
                self.zero(c, units, self.acc[id(value)])
            by_store = {d.fused[id(v)]: self.acc[id(v)] for v in fused}
            with c.block_loop(wt, CHUNK) as (done, n):
                for _, cb in row_outs:
                    c.reserve(cb, CHUNK)
                with c.loop(n) as i:
                    j = c.let("j", done + i)
                    for at, ((_, cb), (_, cases)) in enumerate(
                        zip(row_outs, d.stores, strict=True)
                    ):
                        self.scope(c, units, cases, i, j, cb, i, by_store.get(at))
                for _, cb in row_outs:
                    c.push(cb, CHUNK)
            for value in fused:
                self.fused_sum(c, units, value)
            for (_, cb), (_, chain) in zip(col_outs, d.col_stores, strict=True):
                c.reserve(cb, 1)
                self.scope(c, units, [Case(None, chain)], None, None, cb, 0)
                c.push(cb, 1)
            for value in d.values:
                if id(value) not in self.skip:
                    c.pop(self.cbs[id(value)], 1)
            for op in per:
                c.pop(self.cbs[id(op)], self.seg if op.row else 1)
        for op in once:
            c.pop(self.cbs[id(op)], self.seg)
        if self.ones is not None:
            c.pop(self.ones, 1)

        w = p.kernel("dm1")
        with w.loop(count, name="u") as u:
            v.unit = w.let("r", start + u)
            if row_outs:
                with w.block_loop(wt, CHUNK) as (done, n):
                    for _, cb in row_outs:
                        w.wait(cb, CHUNK)
                    with w.loop(n) as i:
                        for s, cb in row_outs:
                            ref = self.refs[s.view.tensor.name]
                            page = v(s.view.index) * wt + done + i
                            w.write(cb, ref.page(page), index=i)
                    w.flushed()
                    for _, cb in row_outs:
                        w.pop(cb, CHUNK)
            for s, cb in col_outs:
                w.wait(cb, 1)
                w.write(cb, self.refs[s.view.tensor.name].page(v(s.view.index)))
            if col_outs:
                w.flushed()
                for _, cb in col_outs:
                    w.pop(cb, 1)
        w.writes_done()
        return p

    def compute(self, outs):
        d, p = self.d, self.p
        cbs = [self.cbs[id(op)] for op in d.operands] + [cb for _, cb in outs]
        cbs += [self.cbs[id(v)] for v in d.values if id(v) not in self.skip]
        cbs += list(self.acc.values())
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

    def scope(self, c, units, cases, i, j, cb_out, at, acc=None):
        """One register scope of a store's cases packed into page `at` of `cb_out`;
        with `acc`, the same slot is also added into `acc`'s page (fp32)."""
        if acc is None:
            super().scope(c, units, cases, i, j, cb_out, at)
            return
        c.acquire()
        units.forget()
        self.cases(c, units, cases, i, j)
        units.pack_into(cb_out)
        c.pack(0, cb_out, at=at)
        units.pack_into(acc)
        c.pack_accumulate(1, acc)
        c.pack(0, acc, at=0)
        c.pack_accumulate(0)
        c.release()

    def zero(self, c, units, cb):
        """Reserves `cb`'s page and packs zeros into it."""
        c.reserve(cb, 1)
        c.acquire()
        units.forget()
        units.setup("fill")
        c.op("fill", 0, value=0.0)
        units.pack_into(cb)
        c.pack(0, cb, at=0)
        c.release()

    def fused_sum(self, c, units, value):
        """The row sums of the accumulated tile (the column sums of its transpose,
        fp32) into column 0 of the value's buffer."""
        acc = self.acc[id(value)]
        c.push(acc, 1)
        c.wait(acc, 1)
        cb_value = self.cbs[id(value)]
        c.acquire()
        units.forget()
        units.unpack_as("transpose", acc)
        c.transpose(acc, 0, 0)
        units.setup("stats")
        c.stats_clear()
        c.stats_sum(0)
        c.stats_mean(1.0)
        c.stats_finalize(2, 1.0)
        units.setup("transpose_dest")
        c.op("transpose_dest", 2)
        c.reserve(cb_value, 1)
        units.pack_into(cb_value)
        c.pack(2, cb_value)
        c.release()
        c.pop(acc, 1)
        c.push(cb_value, 1)
        c.wait(cb_value, 1)

    def moments(self, c, units):
        """The mean and M2 (sum of squared deviations) of the summed row in one
        statistics scope: shifted sums, the mean, then M2 about it in a second walk
        of the held row; into the mean's and M2's buffers (column 0)."""
        total, mean, m2 = self.d.moments
        leaf = total.chain.operands[0]
        cb = self.cbs[id(leaf.source)]
        last = self.lasts[total.width]
        recip = self.v.scalar(mean.chain.values[0])
        c.acquire()
        units.forget()
        units.unpack_as("transpose", cb)
        units.setup("stats")
        c.stats_clear()
        with c.loop(self.wt, name="j") as j:
            c.transpose(cb, j.bounded(0, self.seg), 0)
            n = km.select(j.eq(self.wt - 1), last, 32)
            with c.when(j.eq(0)):
                c.stats_sum(0, count=n, anchor=True)
            with c.otherwise():
                c.stats_sum(0, count=n)
        c.stats_mean(recip)
        with c.loop(self.wt, name="j") as j:
            c.transpose(cb, j.bounded(0, self.seg), 0)
            c.stats_m2(0, count=km.select(j.eq(self.wt - 1), last, 32))
        c.stats_finalize(2, 1.0)
        units.setup("transpose_dest")
        c.op("transpose_dest", 2)
        c.op("transpose_dest", 3)
        targets = ((self.cbs[id(mean)], 2), (self.cbs[id(m2)], 3))
        for target, _ in targets:
            c.reserve(target, 1)
        units.pack_into(targets[0][0])
        for target, slot in targets:
            c.pack(slot, target)
        c.release()
        for target, _ in targets:
            c.push(target, 1)
            c.wait(target, 1)

    def value(self, c, units, value):
        """Computes column value `value` of the unit into its one-page buffer."""
        cb_value = self.cbs[id(value)]
        c.acquire()
        units.forget()
        if value.kind in ("max", "sum"):
            self.matrix_reduce(c, units, value)
            slot = 0
        elif value.kind == "exact":
            self.exact_sum(c, units, value)
            slot = 2
        else:
            self.run(c, units, value.chain, None, None)
            slot = 0
        c.reserve(cb_value, 1)
        units.pack_into(cb_value)
        c.pack(slot, cb_value)
        c.release()
        if value.kind in ("max", "sum"):
            c.reduce_done()
            units.forget()
            units.srca = units.srcb = units.pack = None
        c.push(cb_value, 1)
        c.wait(cb_value, 1)

    def matrix_reduce(self, c, units, value):
        """The row's max or sum on the matrix unit into column 0 of slot 0."""
        cb = self.cbs[id(value.operand)]
        # The reduction unpacks the ones through srca and the row through srcb.
        if units.srca != self.ones.dtype:
            c.srca_format(self.ones)
        if units.srcb != cb.dtype:
            c.format(srcb=cb)
        units.srca, units.srcb = self.ones.dtype, cb.dtype
        r = c.reduce_init(value.kind, "row", cb, self.ones, self.cbs[id(value)])
        # The first tile outside the loop, so the slot's last writer is known.
        c.reduce(r, 0, 0, 0)
        with c.loop(self.wt - 1, name="j") as j:
            c.reduce(r, (j + 1).bounded(0, self.seg), 0, 0)

    def exact_sum(self, c, units, value):
        """The fp32 sum of the value's row chain, tile by tile, into column 0 of slot
        2; the statistics wait in slots 2 and 3 while the chain runs."""
        last = self.lasts[value.width]
        chain = value.chain
        # Column operands are unpacked once, into the slots after the row operands.
        tiled = [_is_row(x) for x in chain.operands]
        order = [i for i, t in enumerate(tiled) if t] + [
            i for i, t in enumerate(tiled) if not t
        ]
        slots = [order.index(i) for i in range(len(order))]
        fixed = [i for i, t in enumerate(tiled) if not t]
        staged = chain.prev is not None
        if not staged:
            self.unpack(
                c, units, _Some(chain, fixed), None, None, [slots[i] for i in fixed]
            )
        units.setup("stats")
        c.stats_clear()
        c.stats_save(2)
        moving = [i for i, t in enumerate(tiled) if t]
        with c.loop(self.wt, name="j") as j:
            units.forget()
            if staged:
                # Every stage within slots 0 and 1, the result in slot 0.
                self.run(c, units, chain, None, j)
            else:
                self.unpack(
                    c, units, _Some(chain, moving), None, j, [slots[i] for i in moving]
                )
                units.setup(chain.fn.init)
                c.run(chain.fn, *slots, out=0, values=self.values(chain))
            units.setup("transpose_dest")
            c.op("transpose_dest", 0)
            units.setup("stats")
            c.stats_restore(2)
            c.stats_sum(0, count=km.select(j.eq(self.wt - 1), last, 32))
            c.stats_save(2)
        units.forget()
        units.setup("stats")
        c.stats_restore(2)
        c.stats_mean(1.0)
        c.stats_finalize(2, 1.0)
        units.setup("transpose_dest")
        c.op("transpose_dest", 2)


def _is_row(leaf):
    return isinstance(leaf.source, Operand) and leaf.source.row


class _Some:
    """The operands `which` of `chain`, for unpacking part of them."""

    def __init__(self, chain, which):
        self.operands = [chain.operands[i] for i in which]


def lower_columns(design, machine, mode, depth, seg_limit):
    """(km.Program, Values) of ColumnDesign `design`: rows held whole in L1."""
    if mode != "hold":
        fail("a kernel with column values holds its rows in L1; this row does not fit")
    low = _ColumnLowering(design, machine, 1, mode, depth)
    low.seg_limit = seg_limit
    return low.build(), low.v
