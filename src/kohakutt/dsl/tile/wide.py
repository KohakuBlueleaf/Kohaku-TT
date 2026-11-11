"""Row kernels split over a group of cores (`kd.grid(n, group=g)`): each of the g
cores of a unit holds a slice of the row's tiles, sums its slice's squares, the group
exchanges the g partial sums (a sync point per group, `k.front` and `k.fetch`), and
every core finishes the statistic and stores its own slice."""

import kohakutt.dsl.metal as km
from kohakutt.dsl.tile.ir import F32, Int, fail
from kohakutt.dsl.tile.lower import _Lowering, _Units

f = km.sfpu


def unit_class(units, cores, g):
    """The units a launch is laid out for: `units` rounded up to the metal level's
    size classes while the groups fit `cores`, so a new size in a class builds no
    program."""
    for size in km.SIZE_CLASSES:
        if size != "grid" and size >= units and size * g <= cores:
            return size
    return units


def check_wide(design):
    """Refusals of the split form for row design `design`."""
    if design.form != "row" or design.reduce is None:
        fail("kd.grid(..., group=) splits a row kernel with a row statistic")
    if any(op.random for op in design.operands):
        fail("a row split over cores reads each tile at its own place (no slice/cat)")
    if any(op.row and not op.held and not op.resident for op in design.operands):
        fail("a row split over cores is held in L1 by every core of the group")


class _WideLowering(_Lowering):
    def build(self):
        d, p, v, g = self.d, self.p, self.v, self.d.ir.grid.group
        units = v.host_value(d.ir.grid.count)
        out_tensor = d.stores[0][0].view.tensor
        wt = v.host_value(Int("size", value=("row_tiles", out_tensor.name, None)))
        working = p.runtime("working")
        cw = p.runtime("cw")
        v.extra = {
            "working": lambda env, cores: unit_class(env[units.name], cores, g) * g,
            "cw": lambda env, cores: -(-env[wt.name] // g),
        }
        p.layout(cores=working)
        point = p.sync_point(group=g)
        self.seg = p.sized("seg", of=cw, limit=self.seg_limit)
        p.limit(cw, low=1, high=self.seg)
        unit = p.define("unit", p.core // g)
        j0 = p.define("j0", p.core % g * cw)
        # Groups past the last unit (the core count is a size class) get no tiles.
        n = p.define(
            "n", km.select(unit < units, km.select(j0 < wt, km.min(cw, wt - j0), 0), 0)
        )
        name = d.reduce["operand"].tensor.name
        width = Int("size", value=("size", name, -1))
        tiles = Int("size", value=("row_tiles", name, None))
        last = v.host_value(width - (tiles - 1) * 32, "last")
        for op in d.operands:
            self.cbs[id(op)] = p.cb(like=self.ref(op), pages=self.seg)
        cb_part = p.cb(dtype=F32, pages=1, name="part")
        cb_all = p.cb(dtype=F32, pages=g, name="parts")
        self.cbs["stat"] = p.cb(dtype=F32, pages=1, name="stat")
        outs = [
            (s, p.cb(like=self.refs[s.view.tensor.name], pages=2)) for s, _ in d.stores
        ]

        k = p.kernel("dm0")
        v.unit = unit
        for op in d.operands:
            slot = k.reserve(self.cbs[id(op)], self.seg)
            with k.loop(n, name="j") as j:
                page = self._row_page(op, v(op.view.index), j0 + j)
                k.read(page, slot[j.bounded(0, self.seg)])
        k.reads_done()
        for op in d.operands:
            k.push(self.cbs[id(op)], self.seg)
        mine = k.front(cb_part, 1)
        k.sync(point)
        every = k.reserve(cb_all, g)
        for i in range(g):
            k.fetch(mine, point.member(i), every[i], cb_part.page_bytes)
        k.reads_done()
        k.push(cb_all, g)
        k.sync(point)
        k.pop(cb_part, 1)

        c, units_ = self.compute(outs)
        for op in d.operands:
            c.wait(self.cbs[id(op)], self.seg)
        self.partial(c, units_, n, j0, wt, last, cb_part)
        self.combine(c, units_, cb_all, g)
        _, cases = d.stores[0]
        with c.loop(n, name="j") as j:
            for (_, cb), (_, cases) in zip(outs, d.stores, strict=True):
                c.reserve(cb, 1)
                self.scope(c, units_, cases, None, j, cb, 0)
                c.push(cb, 1)
        c.pop(self.cbs["stat"], 1)
        for op in d.operands:
            c.pop(self.cbs[id(op)], self.seg)

        w = p.kernel("dm1")
        v.unit = unit
        with w.loop(n, name="j") as j:
            for s, cb in outs:
                w.wait(cb, 1)
                ref = self.refs[s.view.tensor.name]
                w.write(cb, ref.page(v(s.view.index) * wt + j0 + j))
            w.flushed()
            for _, cb in outs:
                w.pop(cb, 1)
        w.writes_done()
        return p

    def compute(self, outs):
        d, p = self.d, self.p
        cbs = [self.cbs[id(op)] for op in d.operands] + [cb for _, cb in outs]
        cbs += [cb for key, cb in self.cbs.items() if not isinstance(key, int)]
        cbs += [cb for cb in p.cbs if cb.name in ("part", "parts")]
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

    def partial(self, c, units, n, j0, wt, last, cb_part):
        """The slice's sum of squares into column 0 of the part page (pushed)."""
        op = self.d.reduce["operand"]
        cb = self.cbs[id(op)]
        c.acquire()
        units.forget()
        units.setup("fill")
        c.op("fill", 1, value=0.0)
        c.op("fill", 2, value=0.0)
        units.unpack_as("transpose", cb)
        units.setup("stats")
        c.stats_clear()
        with c.loop(n, name="j") as j:
            c.transpose(cb, j.bounded(0, self.seg), 0)
            count = km.select((j0 + j).eq(wt - 1), last, 32)
            c.stats_m2(0, count=count, dual=True)
        c.stats_finalize(1, 1.0, dual=True)
        units.setup("transpose_dest")
        c.op("transpose_dest", 2)
        c.reserve(cb_part, 1)
        units.pack_into(cb_part)
        c.pack(2, cb_part)
        c.release()
        c.push(cb_part, 1)

    def combine(self, c, units, cb_all, g):
        """The group's g partial sums added (fp32), then the statistic chain; into the
        statistic buffer."""
        c.wait(cb_all, g)
        c.acquire()
        units.forget()
        units.unpack_as("copy", cb_all)
        c.copy(cb_all, 0, 0)
        for i in range(1, g):
            units.unpack_as("copy", cb_all)
            c.copy(cb_all, i, 1)
            units.setup(_ADD.init)
            c.run(_ADD, 0, 1, out=0)
        stat = self.d.stat
        free = iter((1, 2, 3))
        slots = [
            0 if getattr(x, "source", None) == "var" else next(free)
            for x in stat.operands
        ]
        units.sfpu = None
        self.run(c, units, stat, None, None, slots, out=0)
        cb_stat = self.cbs["stat"]
        c.reserve(cb_stat, 1)
        units.pack_into(cb_stat)
        c.pack(0, cb_stat)
        c.release()
        c.push(cb_stat, 1)
        c.wait(cb_stat, 1)
        c.pop(cb_all, g)


_ADD = km.sfpu_fn("tile_wide_add", f.operand(0) + f.operand(1))


def check_wide_columns(design):
    """Refusals of the split form for column design `design`."""
    if any(v.kind in ("max", "sum") for v in design.values):
        fail("a row split over cores sums exactly (exact=True), no matrix-unit reduce")
    if design.fused or design.col_stores:
        fail("a row split over cores stores rows only")
    whole = all(op.row and op.view.kind == "row" for op in design.operands)
    if not whole or any(len(cases) != 1 for _, cases in design.stores):
        fail("a row split over cores reads rows tile by tile (no slice, cat, column)")


def groups_of(cores, g):
    """The core groups of g a split launch lays out on `cores` cores."""
    return max(1, cores // g)


class _WideColumns:
    """Column kernels (LayerNorm) with each tile row split over a group of g cores:
    a group takes rows group, group + G, ... (G groups); its cores each hold a slice
    of ceil(W / g) tiles (resident rows: the slice only, read once); every exact
    sum is a slice's partial, exchanged through the group and added in fp32."""

    def __init__(self, low):
        self.low = low

    def build(self):
        low = self.low
        d, p, v = low.d, low.p, low.v
        g = d.ir.grid.group
        units = v.host_value(d.ir.grid.count)
        wt = v.host_value(Int("size", value=("row_tiles", d.row_tensor.name, None)))
        working, cw = p.runtime("working"), p.runtime("cw")
        v.extra = {
            "working": lambda env, cores: groups_of(cores, g) * g,
            "cw": lambda env, cores: -(-env[wt.name] // g),
        }
        p.layout(cores=working)
        point = p.sync_point(group=g)
        low.seg = p.sized("seg", of=cw, limit=low.seg_limit)
        p.limit(cw, low=1, high=low.seg)
        groups = p.define("groups", working // g)
        group = p.define("group", p.core // g)
        j0 = p.define("j0", p.core % g * cw)
        n_all = p.define("n", km.select(j0 < wt, km.min(cw, wt - j0), 0))
        # Rows group, group + G, ...: every core walks the most rows a group has
        # (its syncs the same on every core); past the last row it has no tiles.
        count = (units + groups - 1) // groups
        per_row = {}

        def row_of(b, u):
            r = b.let("r", group + u * groups)
            per_row[id(b)] = b.let("n_r", km.select(r < units, n_all, 0))
            return r

        for op in d.operands:
            depth = 1 if op.resident else low.depth
            low.cbs[id(op)] = p.cb(like=low.ref(op), pages=depth * low.seg)
        for value in d.values:
            low.cbs[id(value)] = p.cb(dtype=F32, pages=1, name=value.name)
        exact = [x for x in d.values if x.kind == "exact"]
        cb_part = p.cb(dtype=F32, pages=1, name="part")
        cb_all = p.cb(dtype=F32, pages=g, name="parts")
        low.acc, low.extra = {}, [cb_part, cb_all]
        low.lasts = {}
        for value in exact:
            if value.width not in low.lasts:
                size = Int("size", value=("size", value.width, -1))
                tiles = Int("size", value=("row_tiles", value.width, None))
                low.lasts[value.width] = v.host_value(
                    size - (tiles - 1) * 32, f"{value.width}_last"
                )
        outs = [
            (s, p.cb(like=low.refs[s.view.tensor.name], pages=2)) for s, _ in d.stores
        ]
        once = [op for op in d.operands if op.resident]
        per = [op for op in d.operands if not op.resident]

        k = p.kernel("dm0")
        for op in once:
            slot = k.reserve(low.cbs[id(op)], low.seg)
            with k.loop(n_all, name="j") as j:
                page = low._row_page(op, v(op.view.index), j0 + j)
                k.read(page, slot[j.bounded(0, low.seg)])
        if once:
            k.reads_done()
            for op in once:
                k.push(low.cbs[id(op)], low.seg)
        with k.loop(count, name="u") as u:
            v.unit = row_of(k, u)
            for op in per:
                slot = k.reserve(low.cbs[id(op)], low.seg)
                with k.loop(per_row[id(k)], name="j") as j:
                    page = low._row_page(op, v(op.view.index), j0 + j)
                    k.read(page, slot[j.bounded(0, low.seg)])
            k.reads_done()
            for op in per:
                k.push(low.cbs[id(op)], low.seg)
            for _ in exact:
                mine = k.front(cb_part, 1)
                k.sync(point)
                every = k.reserve(cb_all, g)
                for i in range(g):
                    k.fetch(mine, point.member(i), every[i], cb_part.page_bytes)
                k.reads_done()
                k.push(cb_all, g)
                k.sync(point)
                k.pop(cb_part, 1)

        c, units_ = low.compute(outs)
        for op in once:
            c.wait(low.cbs[id(op)], low.seg)
        with c.loop(count, name="u") as u:
            v.unit = row_of(c, u)
            n = per_row[id(c)]
            for op in per:
                c.wait(low.cbs[id(op)], low.seg)
            for value in d.values:
                if value.kind == "exact":
                    self.partial(c, units_, value, n, j0, wt, cb_part)
                    self.combine(c, units_, value, cb_all, g)
                else:
                    low.value(c, units_, value)
            with c.loop(n, name="j") as j:
                for (_, cb), (_, cases) in zip(outs, d.stores, strict=True):
                    c.reserve(cb, 1)
                    low.scope(c, units_, cases, None, j, cb, 0)
                    c.push(cb, 1)
            for value in d.values:
                c.pop(low.cbs[id(value)], 1)
            for op in per:
                c.pop(low.cbs[id(op)], low.seg)
        for op in once:
            c.pop(low.cbs[id(op)], low.seg)

        w = p.kernel("dm1")
        with w.loop(count, name="u") as u:
            v.unit = row_of(w, u)
            with w.loop(per_row[id(w)], name="j") as j:
                for s, cb in outs:
                    w.wait(cb, 1)
                    ref = low.refs[s.view.tensor.name]
                    w.write(cb, ref.page(v(s.view.index) * wt + j0 + j))
                w.flushed()
                for _, cb in outs:
                    w.pop(cb, 1)
        w.writes_done()
        return p

    def partial(self, c, units, value, n, j0, wt, cb_part):
        """The slice's fp32 sum of the value's chain (column 0), pushed to the part
        page: the column-value kernel's exact sum over the slice's tiles."""
        low = self.low
        last = low.lasts[value.width]
        c.acquire()
        units.forget()
        units.setup("stats")
        c.stats_clear()
        c.stats_save(2)
        with c.loop(n, name="j") as j:
            units.forget()
            low.run(c, units, value.chain, None, j)
            units.setup("transpose_dest")
            c.op("transpose_dest", 0)
            units.setup("stats")
            c.stats_restore(2)
            c.stats_sum(0, count=km.select((j0 + j).eq(wt - 1), last, 32))
            c.stats_save(2)
        units.forget()
        units.setup("stats")
        c.stats_restore(2)
        c.stats_mean(1.0)
        c.stats_finalize(2, 1.0)
        units.setup("transpose_dest")
        c.op("transpose_dest", 2)
        c.reserve(cb_part, 1)
        units.pack_into(cb_part)
        c.pack(2, cb_part)
        c.release()
        c.push(cb_part, 1)

    def combine(self, c, units, value, cb_all, g):
        """The group's g partials added (fp32) into the value's page."""
        cb_value = self.low.cbs[id(value)]
        c.wait(cb_all, g)
        c.acquire()
        units.forget()
        units.unpack_as("copy", cb_all)
        c.copy(cb_all, 0, 0)
        for i in range(1, g):
            units.unpack_as("copy", cb_all)
            c.copy(cb_all, i, 1)
            units.setup(_ADD.init)
            c.run(_ADD, 0, 1, out=0)
        units.sfpu = None
        c.reserve(cb_value, 1)
        units.pack_into(cb_value)
        c.pack(0, cb_value)
        c.release()
        c.pop(cb_all, g)
        c.push(cb_value, 1)
        c.wait(cb_value, 1)


def lower_wide(design, machine, seg_limit):
    """(km.Program, Values) of row design `design` split over groups of cores."""
    if design.form == "columns":
        from kohakutt.dsl.tile.columns import _ColumnLowering

        check_wide_columns(design)
        low = _ColumnLowering(design, machine, 1, "hold", 2)
        low.seg_limit = seg_limit
        low.skip = set()
        return _WideColumns(low).build(), low.v
    check_wide(design)
    low = _WideLowering(design, machine, 1, "hold", 1)
    low.seg_limit = seg_limit
    return low.build(), low.v
