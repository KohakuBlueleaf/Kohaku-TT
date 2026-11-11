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


def lower_wide(design, machine, seg_limit):
    """(km.Program, Values) of row design `design` split over groups of cores."""
    check_wide(design)
    low = _WideLowering(design, machine, 1, "hold", 1)
    low.seg_limit = seg_limit
    return low.build(), low.v
