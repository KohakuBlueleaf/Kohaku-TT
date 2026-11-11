"""Row kernels split over groups of cores: each tile row is cut into g slices held by
the g consecutive cores of a group; group k takes rows k, k + G, ... (G groups). Per
row the group exchanges each core's partial statistic once (a sync point per group,
`k.front` and `k.fetch`), every core finishes the statistic and stores its slice."""

import kohakutt.dsl.metal as km
from kohakutt.dsl.tile.ir import F32, Int, fail
from kohakutt.dsl.tile.lower import _Lowering, _Units, zone
from kohakutt.dsl.tile.plan import CHUNK

f = km.sfpu
# Row 0 of an fp32 tile: 16 values at the start of faces 0 and 1 (bytes).
ROW_FACES = (0, 1024)
ROW_BYTES = 64


def unit_class(units, cores, g):
    """The units a launch is laid out for: `units` rounded up to the metal level's
    size classes while the groups fit `cores`, so a new size in a class builds no
    program."""
    for size in km.SIZE_CLASSES:
        if size != "grid" and size >= units and size * g <= cores:
            return size
    return units


def groups_of(cores, g):
    """The core groups of g a split launch lays out on `cores` cores."""
    return max(1, cores // g)


def check_wide(design):
    """Refusals of the split form for row design `design`."""
    if design.form != "row" or design.reduce is None:
        fail("kd.grid(..., group=) splits a row kernel with a row statistic")
    if any(op.random for op in design.operands):
        fail("a row split over cores reads each tile at its own place (no slice/cat)")
    if any(op.row and not op.held and not op.resident for op in design.operands):
        fail("a row split over cores is held in L1 by every core of the group")


def check_wide_columns(design):
    """Refusals of the split form for column design `design`."""
    if any(v.kind in ("max", "sum") for v in design.values):
        fail("a row split over cores sums exactly (exact=True), no matrix-unit reduce")
    if design.fused or design.col_stores:
        fail("a row split over cores stores rows only")
    whole = all(op.row and op.view.kind == "row" for op in design.operands)
    if not whole or any(len(cases) != 1 for _, cases in design.stores):
        fail("a row split over cores reads rows tile by tile (no slice, cat, column)")


class _Split:
    """The layout of a split launch for lowering `low`: groups of g consecutive
    cores; member m of a group holds tiles m * cw .. of every row it takes."""

    def __init__(self, low, g, units, wt):
        p, v = low.p, low.v
        self.low, self.g, self.units, self.wt = low, g, units, wt
        working, cw = p.runtime("working"), p.runtime("cw")
        # Groups: one per row up to the cores' groups, the rows rounded up to the
        # metal level's size classes (a new size in a class builds no program).
        v.extra = {
            "working": lambda env, cores: g
            * min(groups_of(cores, g), unit_class(env[units.name], cores, g)),
            "cw": lambda env, cores: -(-env[wt.name] // g),
        }
        p.layout(cores=working)
        self.point = p.sync_point(group=g)
        low.seg = p.sized("seg", of=cw, limit=low.seg_limit)
        p.limit(cw, low=1, high=low.seg)
        self.groups = p.define("groups", working // g)
        self.group = p.define("group", p.core // g)
        self.member = p.define("member", p.core % g)
        self.j0 = p.define("j0", self.member * cw)
        self.n_all = p.define(
            "slice_n", km.select(self.j0 < wt, km.min(cw, wt - self.j0), 0)
        )
        # Every core walks the most rows a group has (its syncs the same on every
        # core); past the last row it has no tiles.
        self.count = (units + self.groups - 1) // self.groups
        self._n = {}

    def row(self, b, u):
        """Row `u` of the running core's group in builder `b` (sets its tile count)."""
        r = b.let("r", self.group + u * self.groups)
        self._n[id(b)] = b.let("n_r", km.select(r < self.units, self.n_all, 0))
        return r

    def n(self, b):
        return self._n[id(b)]

    def read(self, k, ops, n):
        """Reader: the slices of `ops` (n tiles each) into their buffers, pushed."""
        low, v = self.low, self.low.v
        for op in ops:
            slot = k.reserve(low.cbs[id(op)], low.seg)
            with k.loop(n, name="j") as j:
                page = low._row_page(op, v(op.view.index), self.j0 + j)
                k.read(page, slot[j.bounded(0, low.seg)])
        if ops:
            k.reads_done()
        for op in ops:
            k.push(low.cbs[id(op)], low.seg)

    def exchange(self, k, cb_part, cb_all, pages):
        """Reader: every member's `pages` part pages fetched into `cb_all`: of each,
        only row 0 (the statistics' row: 16 fp32 in each of faces 0 and 1)."""
        g = self.g
        mine = k.front(cb_part, pages)
        k.sync(self.point)
        every = k.reserve(cb_all, g * pages)
        page = cb_part.page_bytes
        for i in range(g):
            for q in range(pages):
                for face in ROW_FACES:
                    src = mine.at(q * page + face)
                    dst = every.at((i * pages + q) * page + face)
                    k.fetch(src, self.point.member(i), dst, ROW_BYTES)
        k.reads_done()
        k.push(cb_all, g * pages)
        k.sync(self.point)
        k.pop(cb_part, pages)

    def stores(self, c, units, outs, n):
        """Compute: the row's stored slice, a chunk of tiles per output reserve."""
        low = self.low
        with zone(c, "Y"), c.block_loop(n, CHUNK) as (done, m):
            for _, cb in outs:
                c.reserve(cb, CHUNK)
            low.store_tiles(c, units, outs, low.d.stores, m, done, 0, n)
            for _, cb in outs:
                c.push(cb, CHUNK)

    def write(self, w, outs, n):
        """Writer: the row's stored slice, a chunk per flush."""
        low, v, wt = self.low, self.low.v, self.wt
        with w.block_loop(n, CHUNK) as (done, m):
            for _, cb in outs:
                w.wait(cb, CHUNK)
            with w.loop(m) as i:
                for s, cb in outs:
                    ref = low.refs[s.view.tensor.name]
                    page = v(s.view.index) * wt + self.j0 + done + i
                    w.write(cb, ref.page(page), index=i)
            w.flushed()
            for _, cb in outs:
                w.pop(cb, CHUNK)


def _compute(low, outs, extra):
    """The compute builder of a split lowering and its unit-state tracker."""
    d, p = low.d, low.p
    cbs = [low.cbs[id(op)] for op in d.operands] + [cb for _, cb in outs]
    cbs += [cb for key, cb in low.cbs.items() if not isinstance(key, int)] + extra
    c = p.kernel(
        "compute",
        fp32_dest=True,
        unpack_fp32=[cb for cb in cbs if cb.dtype == F32],
        exact=True,
    )
    first = low.cbs[id(d.operands[0])]
    c.startup(first, None, outs[0][1])
    units = _Units(c, first, False)
    units.pack = outs[0][1].dtype
    return c, units


class _WideLowering(_Lowering):
    """A row kernel with a row statistic (the sum of squares), split."""

    def build(self):
        d, p, v, g = self.d, self.p, self.v, self.g
        units = v.host_value(d.ir.grid.count)
        out_tensor = d.stores[0][0].view.tensor
        wt = v.host_value(Int("size", value=("row_tiles", out_tensor.name, None)))
        s = _Split(self, g, units, wt)
        name = d.reduce["operand"].tensor.name
        width = Int("size", value=("size", name, -1))
        tiles = Int("size", value=("row_tiles", name, None))
        last = v.host_value(width - (tiles - 1) * 32, "last")
        once = [op for op in d.operands if op.resident]
        per = [op for op in d.operands if not op.resident]
        for op in d.operands:
            pages = self.seg if op.resident else self.depth * self.seg
            self.cbs[id(op)] = p.cb(like=self.ref(op), pages=pages)
        cb_part = p.cb(dtype=F32, pages=1, name="part")
        cb_all = p.cb(dtype=F32, pages=g, name="parts")
        self.cbs["stat"] = p.cb(dtype=F32, pages=1, name="stat")
        outs = [
            (st, p.cb(like=self.refs[st.view.tensor.name], pages=2 * CHUNK))
            for st, _ in d.stores
        ]

        k = p.kernel("dm0")
        s.read(k, once, s.n_all)
        with k.loop(s.count, name="u") as u:
            v.unit = s.row(k, u)
            s.read(k, per, s.n(k))
            s.exchange(k, cb_part, cb_all, 1)

        c, units_ = _compute(self, outs, [cb_part, cb_all])
        for op in once:
            c.wait(self.cbs[id(op)], self.seg)
        with c.loop(s.count, name="u") as u:
            v.unit = s.row(c, u)
            n = s.n(c)
            for op in per:
                c.wait(self.cbs[id(op)], self.seg)
            with zone(c, "S"):
                self.partial(c, units_, n, s.j0, wt, last, cb_part)
                self.combine(c, units_, cb_all, g)
            s.stores(c, units_, outs, n)
            c.pop(self.cbs["stat"], 1)
            for op in per:
                c.pop(self.cbs[id(op)], self.seg)
        for op in once:
            c.pop(self.cbs[id(op)], self.seg)

        w = p.kernel("dm1")
        with w.loop(s.count, name="u") as u:
            v.unit = s.row(w, u)
            s.write(w, outs, s.n(w))
        w.writes_done()
        return p

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
        # Row 0 holds the sums (the exchange moves row 0 only).
        c.reserve(cb_part, 1)
        units.pack_into(cb_part)
        c.pack(2, cb_part)
        c.release()
        c.push(cb_part, 1)

    def combine(self, c, units, cb_all, g):
        """The group's g partial sums added (fp32), the statistic chain, then moved
        from row 0 to column 0; into the statistic buffer."""
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
        units.setup("transpose_dest")
        c.op("transpose_dest", 0)
        cb_stat = self.cbs["stat"]
        c.reserve(cb_stat, 1)
        units.pack_into(cb_stat)
        c.pack(0, cb_stat)
        c.release()
        c.push(cb_stat, 1)
        c.wait(cb_stat, 1)
        c.pop(cb_all, g)


_ADD = km.sfpu_fn("tile_wide_add", f.operand(0) + f.operand(1))
# Chan's combination of slice moments: the mean as the weighted sum of the slices'
# means, M2 as the sum of the slices' M2 plus n_k (mean_k - mean)^2.
_WSUM = km.sfpu_fn("tile_wide_wsum", f.operand(0) + f.operand(1) * f.scalar(0))
_D = f.operand(2) - f.operand(1)
_CHAN = km.sfpu_fn(
    "tile_wide_chan", f.operand(0) + f.operand(3) + f.scalar(0) * _D * _D
)


def _bits(x):
    import struct

    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


class _WideColumns:
    """Column kernels (LayerNorm) with each tile row split over a group of g cores:
    resident rows are read once as the slice; an exact sum is a slice's partial,
    exchanged through the group and added in fp32; the mean and M2 of the moments
    pattern come from one exchange of each slice's mean and M2 (Chan's formula)."""

    def __init__(self, low, g):
        self.low, self.g = low, g

    def build(self):
        low, g = self.low, self.g
        d, p, v = low.d, low.p, low.v
        units = v.host_value(d.ir.grid.count)
        wt = v.host_value(Int("size", value=("row_tiles", d.row_tensor.name, None)))
        s = self.s = _Split(low, g, units, wt)
        moments = d.moments
        low.skip = {id(moments[0])} if moments else set()
        for op in d.operands:
            depth = 1 if op.resident else low.depth
            low.cbs[id(op)] = p.cb(like=low.ref(op), pages=depth * low.seg)
        for value in d.values:
            if id(value) not in low.skip:
                low.cbs[id(value)] = p.cb(dtype=F32, pages=1, name=value.name)
        exact = [x for x in d.values if x.kind == "exact"]
        pages = 2 if moments else 1
        cb_part = p.cb(dtype=F32, pages=pages, name="part")
        cb_all = p.cb(dtype=F32, pages=g * pages, name="parts")
        low.acc, low.extra = {}, [cb_part, cb_all]
        low.lasts = {}
        for value in exact:
            if value.width not in low.lasts:
                size = Int("size", value=("size", value.width, -1))
                tiles = Int("size", value=("row_tiles", value.width, None))
                low.lasts[value.width] = v.host_value(
                    size - (tiles - 1) * 32, f"{value.width}_last"
                )
        if moments:
            self.slice_values(g, wt, moments[0].width)
        outs = [
            (st, p.cb(like=low.refs[st.view.tensor.name], pages=2 * CHUNK))
            for st, _ in d.stores
        ]
        once = [op for op in d.operands if op.resident]
        per = [op for op in d.operands if not op.resident]
        # Exchanges per row, in compute order: (value, pages).
        trades = []
        for value in d.values:
            if moments and value is moments[1]:
                trades.append((value, 2))
            elif value.kind == "exact" and not (moments and value in moments):
                trades.append((value, 1))

        k = p.kernel("dm0")
        s.read(k, once, s.n_all)
        with k.loop(s.count, name="u") as u:
            v.unit = s.row(k, u)
            s.read(k, per, s.n(k))
            for _, n_pages in trades:
                s.exchange(k, cb_part, cb_all, n_pages)

        c, units_ = low.compute(outs)
        for op in once:
            c.wait(low.cbs[id(op)], low.seg)
        with c.loop(s.count, name="u") as u:
            v.unit = s.row(c, u)
            n = s.n(c)
            for op in per:
                c.wait(low.cbs[id(op)], low.seg)
            with zone(c, "S"):
                for value in d.values:
                    if moments and value is moments[1]:
                        self.slice_moments(c, units_, n, wt, cb_part)
                        self.chan(c, units_, cb_all, g)
                    elif moments and value in moments:
                        continue
                    elif value.kind == "exact":
                        self.partial(c, units_, value, n, wt, cb_part)
                        self.combine(c, units_, value, cb_all, g)
                    else:
                        low.value(c, units_, value)
            s.stores(c, units_, outs, n)
            for value in d.values:
                if id(value) not in low.skip:
                    c.pop(low.cbs[id(value)], 1)
            for op in per:
                c.pop(low.cbs[id(op)], low.seg)
        for op in once:
            c.pop(low.cbs[id(op)], low.seg)

        w = p.kernel("dm1")
        with w.loop(s.count, name="u") as u:
            v.unit = s.row(w, u)
            s.write(w, outs, s.n(w))
        w.writes_done()
        return p

    def slice_values(self, g, wt, tensor):
        """Host values of the moments' combination: per member k, 1 / n_k, n_k / n and
        n_k (fp32 bits; n_k the slice's elements, n the row's); the running core's
        1 / n_k as `self.inv`."""
        p, v = self.low.p, self.low.v
        size = v.host_value(Int("size", value=("size", tensor, -1)))
        names = {}
        for kind in ("inv", "w", "n"):
            for m in range(g):
                names[kind, m] = p.runtime(f"slice_{kind}{m}")

        def count(env, cores, m):
            cw = -(-env[wt.name] // g)
            return max(0, min(env[size.name], (m + 1) * cw * 32) - m * cw * 32)

        def make(kind, m):
            def fn(env, cores):
                n_m, n = count(env, cores, m), env[size.name]
                if kind == "inv":
                    return _bits(1.0 / n_m if n_m else 0.0)
                return _bits(n_m / n if kind == "w" else n_m)

            return fn

        for (kind, m), value in names.items():
            v.extra[value.name] = make(kind, m)
        self.weights = [names["w", m] for m in range(g)]
        self.counts = [names["n", m] for m in range(g)]
        inv = names["inv", g - 1]
        for m in reversed(range(g - 1)):
            inv = km.select(self.s.member.eq(m), names["inv", m], inv)
        self.inv = p.define("inv_slice", inv)

    def slice_moments(self, c, units, n, wt, cb_part):
        """The slice's mean and M2 about it (one statistics scope over its tiles) into
        the two part pages (column 0), pushed."""
        low, s = self.low, self.s
        total = low.d.moments[0]
        cb = low.cbs[id(total.chain.operands[0].source)]
        last = low.lasts[total.width]
        c.acquire()
        units.forget()
        units.unpack_as("transpose", cb)
        units.setup("stats")
        c.stats_clear()
        with c.loop(n, name="j") as j:
            c.transpose(cb, j.bounded(0, low.seg), 0)
            count = km.select((s.j0 + j).eq(wt - 1), last, 32)
            with c.when(j.eq(0)):
                c.stats_sum(0, count=count, anchor=True)
            with c.otherwise():
                c.stats_sum(0, count=count)
        c.stats_mean(self.inv)
        with c.loop(n, name="j") as j:
            c.transpose(cb, j.bounded(0, low.seg), 0)
            c.stats_m2(0, count=km.select((s.j0 + j).eq(wt - 1), last, 32))
        c.stats_finalize(2, 1.0)
        # Rows 0 hold the mean and M2 (the exchange moves row 0 only).
        c.reserve(cb_part, 2)
        units.pack_into(cb_part)
        c.pack(2, cb_part, at=0)
        c.pack(3, cb_part, at=1)
        c.release()
        c.push(cb_part, 2)

    def chan(self, c, units, cb_all, g):
        """The row's mean and M2 from the g slices' (Chan's formula), into the mean's
        and M2's pages."""
        low = self.low
        _, mean, m2 = low.d.moments
        c.wait(cb_all, 2 * g)
        c.acquire()
        units.forget()
        units.setup("fill")
        c.op("fill", 0, value=0.0)
        c.op("fill", 1, value=0.0)
        for m in range(g):
            units.unpack_as("copy", cb_all)
            c.copy(cb_all, 2 * m, 2)
            units.setup(_WSUM.init)
            c.run(_WSUM, 0, 2, out=0, values=(self.weights[m],))
        for m in range(g):
            units.unpack_as("copy", cb_all)
            c.copy(cb_all, 2 * m, 2)
            c.copy(cb_all, 2 * m + 1, 3)
            units.setup(_CHAN.init)
            c.run(_CHAN, 1, 0, 2, 3, out=1, values=(self.counts[m],))
        units.setup("transpose_dest")
        c.op("transpose_dest", 0)
        c.op("transpose_dest", 1)
        targets = ((low.cbs[id(mean)], 0), (low.cbs[id(m2)], 1))
        for target, _ in targets:
            c.reserve(target, 1)
        units.pack_into(targets[0][0])
        for target, slot in targets:
            c.pack(slot, target)
        c.release()
        c.pop(cb_all, 2 * g)
        for target, _ in targets:
            c.push(target, 1)
            c.wait(target, 1)

    def partial(self, c, units, value, n, wt, cb_part):
        """The slice's fp32 sum of the value's chain (column 0), pushed to the part
        page: the column-value kernel's exact sum over the slice's tiles."""
        low, j0 = self.low, self.s.j0
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
        c.reserve(cb_part, 1)
        units.pack_into(cb_part)
        c.pack(2, cb_part)
        c.release()
        c.push(cb_part, 1)

    def combine(self, c, units, value, cb_all, g):
        """The group's g partials added (fp32), moved from row 0 to column 0, into the
        value's page."""
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
        units.setup("transpose_dest")
        c.op("transpose_dest", 0)
        c.reserve(cb_value, 1)
        units.pack_into(cb_value)
        c.pack(0, cb_value)
        c.release()
        c.pop(cb_all, g)
        c.push(cb_value, 1)
        c.wait(cb_value, 1)


def lower_wide(design, machine, seg_limit, g=None):
    """(km.Program, Values) of row design `design` split over groups of g cores (the
    kernel's group when None)."""
    g = g or design.ir.grid.group
    if design.form == "columns":
        from kohakutt.dsl.tile.columns import _ColumnLowering

        check_wide_columns(design)
        low = _ColumnLowering(design, machine, 1, "hold", 2)
        low.seg_limit = seg_limit
        low.skip = set()
        return _WideColumns(low, g).build(), low.v
    check_wide(design)
    low = _WideLowering(design, machine, 1, "hold", 2)
    low.seg_limit, low.g = seg_limit, g
    return low.build(), low.v
