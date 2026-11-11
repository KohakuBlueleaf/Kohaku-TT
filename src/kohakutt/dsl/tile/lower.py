"""Lowering: a Design becomes a metal-level program (`km.Program`), the same objects a
person builds with the metal level's builder. No C++ is written here.
"""

import kohakutt.dsl.metal as km
from kohakutt.dsl.tile.ir import F32, Int, fail
from kohakutt.dsl.tile.plan import CHUNK, SEG_LIMIT, SLOTS, Leaf


class Values:
    """The program's runtime values: each host integer of the IR one value, each
    scalar one value holding fp32 bits. `host` {name: Int}, `scalars` {name: Scalar}."""

    def __init__(self, program):
        self.p, self.host, self.scalars, self._names = program, {}, {}, {}
        self.unit = None
        # Runtime values the host computes from the others and the core count.
        self.extra = {}

    def name_of(self, expr):
        if expr.kind == "size":
            what, tensor, dim = expr.value
            tail = what if dim is None else f"size{dim}".replace("-", "m")
            return f"{tensor}_{tail}"
        if expr.kind == "param":
            return expr.value
        return f"e{len(self.host)}"

    def host_value(self, expr, name=None):
        """The runtime value holding host integer `expr` (named `name` if new)."""
        key = expr.text()
        if key not in self._names:
            name = name or self.name_of(expr)
            self._names[key] = self.p.runtime(name)
            self.host[name] = expr
        return self._names[key]

    def __call__(self, expr):
        """`expr` as a metal-level value: the unit is the current unit variable."""
        if expr.kind == "const":
            return expr.value
        if not expr.device:
            return self.host_value(expr)
        if expr.kind == "unit":
            return self.unit
        a, b = (self(x) for x in expr.args)
        op = expr.value
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "//":
            return a // b
        return a % b

    def scalar(self, s):
        """The runtime value holding the fp32 bits of scalar `s`."""
        v = s.value
        if hasattr(v, "name"):
            base = v.name
        elif v.kind == "size" and v.value[0] == "size":
            base = f"{v.value[1]}_size{v.value[2]}".replace("-", "m")
        else:
            base = f"scalar{len(self.scalars)}"
        name = ("inv_" if s.recip else "") + base + "_f32"
        if name not in self.scalars:
            self.scalars[name] = s
            self.p.runtime(name)
        return self.p.runtime(name)


def _terms(expr, sign=1):
    """[(sign, term)] of an Int written as a sum and difference of terms, so that
    each term is a runtime value of its own (none of them negative)."""
    if expr.kind == "op" and expr.value in ("+", "-"):
        a, b = expr.args
        return _terms(a, sign) + _terms(b, sign if expr.value == "+" else -sign)
    return [(sign, expr)]


class _Units:
    """What the compute kernel set last: unpacker formats and way, vector-unit setup,
    packer format; `forget()` at the start of each repeated section."""

    def __init__(self, c, startup, stable):
        self.c, self.stable = c, stable
        self.srca = self.srcb = startup.dtype
        self.pack = None
        self.unpack = self.sfpu = None

    def forget(self):
        self.unpack = self.sfpu = None
        if not self.stable:
            self.srca = self.srcb = self.pack = None

    def unpack_as(self, kind, cb):
        source = "srcb" if kind.startswith("bcast") and cb.dtype != F32 else "srca"
        if getattr(self, source) != cb.dtype:
            if source == "srca":
                self.c.srca_format(cb)
            else:
                self.c.format(srcb=cb)
            setattr(self, source, cb.dtype)
        if self.unpack != (kind, cb.dtype):
            self.c.init(kind, cb)
            self.unpack = (kind, cb.dtype)

    def setup(self, name):
        if self.sfpu != name:
            self.c.init(name)
            self.sfpu = name

    def pack_into(self, cb):
        if self.pack != cb.dtype:
            self.c.pack_format(cb)
            self.pack = cb.dtype


class _Lowering:
    def __init__(self, design, machine, block, mode, depth, share=None):
        self.d, self.block, self.mode, self.depth = design, block, mode, depth
        ir = design.ir
        if share is not None:
            # A part of another program: its Program, Values and tensor refs.
            self.p, self.v, self.refs = share
            self.cbs = {}
            return
        self.p = p = km.Program(ir.name, machine)
        self.v = Values(p)
        self.refs = {}
        read = {op.tensor.name for op in design.operands}
        for t in ir.tensors:
            if t.output:
                self.refs[t.name] = p.output(t.name, t.dtype)
            elif t.name in read:
                self.refs[t.name] = p.input(t.name, t.dtype)
        self.cbs = {}

    def ref(self, op):
        return self.refs[op.tensor.name]

    def held(self, op):
        """True when row operand `op` is kept whole in L1 for its unit."""
        return op.row and (op.resident or (op.held and self.mode == "hold"))

    def streamed_in(self, op, pass_):
        """True when the reader streams `op` in chunks for pass `pass_`."""
        return op.row and not self.held(op) and pass_ in op.passes

    def build(self):
        if self.d.form == "stream":
            self.stream()
        else:
            self.rows()
        return self.p

    # The stream form: a unit is one tile.
    def stream(self):
        d, p, v, block = self.d, self.p, self.v, self.block
        count_value = v.host_value(d.ir.grid.count)
        start, count = p.split(count_value, min_per_core=block)
        per = [op for op in d.operands if not op.resident]
        once = [op for op in d.operands if op.resident]
        for op in per:
            self.cbs[id(op)] = p.cb(like=self.ref(op), pages=2 * block)
        for op in once:
            self.cbs[id(op)] = p.cb(like=self.ref(op), pages=1)
        outs = [
            (s, p.cb(like=self.refs[s.view.tensor.name], pages=2 * block))
            for s, _ in d.stores
        ]

        k = p.kernel("dm0")
        for op in once:
            slot = k.reserve(self.cbs[id(op)], 1)
            k.read(self.ref(op).page(v(op.view.index)), slot)
            k.reads_done()
            k.push(self.cbs[id(op)], 1)
        with k.block_loop(count, block) as (done, n):
            slots = [k.reserve(self.cbs[id(op)], block) for op in per]
            with k.loop(n) as i:
                v.unit = k.let("t", start + done + i)
                for op, slot in zip(per, slots, strict=True):
                    k.read(self.ref(op).page(v(op.view.index)), slot[i])
            k.reads_done()
            for op in per:
                k.push(self.cbs[id(op)], block)

        c, units = self.compute(outs)
        for op in once:
            c.wait(self.cbs[id(op)], 1)
        with c.block_loop(count, block) as (done, n):
            for op in per:
                c.wait(self.cbs[id(op)], block)
            for _, cb in outs:
                c.reserve(cb, block)
            group = self.group_size(block)
            with c.block_loop(n, group, names=("base", "m")) as (base, _):
                for (_, cb), (_, cases) in zip(outs, d.stores, strict=True):
                    self.group(c, units, cases[0].chain, base, group, cb)
            for _, cb in outs:
                c.push(cb, block)
            for op in per:
                c.pop(self.cbs[id(op)], block)
        for op in once:
            c.pop(self.cbs[id(op)], 1)

        w = p.kernel("dm1")
        with w.block_loop(count, block) as (done, n):
            for _, cb in outs:
                w.wait(cb, block)
            with w.loop(n) as i:
                v.unit = w.let("t", start + done + i)
                for s, cb in outs:
                    ref = self.refs[s.view.tensor.name]
                    w.write(cb, ref.page(v(s.view.index)), index=i)
            w.flushed()
            for _, cb in outs:
                w.pop(cb, block)
        w.writes_done()

    def group_size(self, block):
        """Tiles per register scope: as many as the slots hold, dividing `block`."""
        slots = SLOTS if self.d.dest == "fp32" else 2 * SLOTS
        most = min(slots // len(cases[0].chain.operands) for _, cases in self.d.stores)
        return max(g for g in range(1, min(most, block) + 1) if block % g == 0)

    def compute(self, outs):
        """The compute builder, its startup done, and its unit-state tracker."""
        d, p = self.d, self.p
        cbs = [self.cbs[id(op)] for op in d.operands] + [cb for _, cb in outs]
        cbs += [self.cbs["stat"]] if "stat" in self.cbs else []
        fp32 = [cb for cb in cbs if cb.dtype == F32]
        c = p.kernel(
            "compute",
            fp32_dest=d.dest == "fp32",
            unpack_fp32=fp32,
            exact=d.reduce is not None,
        )
        first = self.cbs[id(d.operands[0])]
        c.startup(first, None, outs[0][1])
        stable = len({cb.dtype for cb in cbs}) == 1
        units = _Units(c, first, stable)
        units.pack = outs[0][1].dtype
        return c, units

    def index(self, leaf, i, j):
        """The tile index of `leaf`: `i` in the waited chunk, `j` in the row."""
        if leaf.source == "stat":
            return 0
        op = leaf.source
        if not op.row:
            return 0 if op.resident else i
        if not self.held(op):
            return i
        at = j
        terms = [] if leaf.offset is None else _terms(leaf.offset)
        for sign, term in sorted(terms, key=lambda st: -st[0]):
            if term.kind == "const" and term.value == 0:
                continue
            at = at + self.v(term) if sign > 0 else at - self.v(term)
        return at.bounded(0, self.seg)

    def group(self, c, units, chain, base, group, cb_out):
        """One register scope of tiles `base` .. `base + group - 1` of the waited
        block: every tile's operands unpacked, then the function run on each."""
        m = len(chain.operands)
        slots = [list(range(r * m, (r + 1) * m)) for r in range(group)]
        at = [(base + r).bounded(0, self.block) for r in range(group)]
        c.acquire()
        units.forget()
        for r in range(group):
            self.unpack(c, units, chain, at[r], None, slots[r])
        units.setup(chain.fn.init)
        values = self.values(chain)
        for r in range(group):
            c.run(chain.fn, *slots[r], out=slots[r][0], values=values)
        units.pack_into(cb_out)
        for r in range(group):
            c.pack(slots[r][0], cb_out, at=at[r])
        c.release()

    def values(self, chain):
        """The runtime values holding the fp32 bits of `chain`'s scalars."""
        return tuple(self.v.scalar(s) for s in chain.values)

    def run(self, c, units, chain, i, j, slots=None, out=0):
        """Unpacks `chain`'s operands into `slots` (default 0 ..) and runs it."""
        slots = slots or list(range(len(chain.operands)))
        self.unpack(c, units, chain, i, j, slots)
        units.setup(chain.fn.init)
        c.run(chain.fn, *slots, out=out, values=self.values(chain))

    def unpack(self, c, units, chain, i, j, slots):
        """Unpacks the tile operands of `chain` into `slots`."""
        for slot, item in zip(slots, chain.operands, strict=True):
            if isinstance(item, Leaf) and item.source != "var":
                cb = self.cbs["stat" if item.source == "stat" else id(item.source)]
                units.unpack_as(item.kind, cb)
                at = self.index(item, i, j)
                if item.kind == "copy":
                    c.copy(cb, at, slot)
                else:
                    c.bcast(cb, at, slot, item.kind[len("bcast_") :])

    def scope(self, c, units, cases, i, j, cb_out, at):
        """One register scope: the cases of a store at tile (`i`, `j`), packed into
        page `at` of the reserve of `cb_out`."""
        c.acquire()
        units.forget()
        self.cases(c, units, cases, i, j)
        units.pack_into(cb_out)
        c.pack(0, cb_out, at=at)
        c.release()

    def cases(self, c, units, cases, i, j):
        first, rest = cases[0], cases[1:]
        if not rest:
            self.run(c, units, first.chain, i, j)
            return
        saved = (units.unpack, units.sfpu, units.srca, units.srcb)
        with c.when(j < self.v(first.bound)):
            self.run(c, units, first.chain, i, j)
        units.unpack, units.sfpu, units.srca, units.srcb = saved
        with c.otherwise():
            self.cases(c, units, rest, i, j)
        units.forget()

    # The row form: a unit is one tile row.
    def rows(self):
        d, p, v = self.d, self.p, self.v
        units_value = v.host_value(d.ir.grid.count)
        out_tensor = d.stores[0][0].view.tensor
        wt = v.host_value(Int("size", value=("row_tiles", out_tensor.name, None)))
        self.seg = p.sized("seg", of=wt, limit=self.seg_limit)
        p.limit(wt, low=1, high=self.seg)
        start, count = p.split(units_value)
        passes = (1, 2) if d.reduce is not None else (2,)
        for op in d.operands:
            if self.held(op):
                depth = 1 if op.resident else self.depth
                pages = depth * self.seg
            elif op.row:
                pages = 2 * CHUNK
            else:
                pages = 1
            self.cbs[id(op)] = p.cb(like=self.ref(op), pages=pages)
        if d.reduce is not None:
            self.cbs["stat"] = p.cb(dtype=F32, pages=1, name="stat")
            last = v.host_value(
                Int("size", value=("size", d.reduce["operand"].tensor.name, -1))
                - (
                    Int(
                        "size",
                        value=("row_tiles", d.reduce["operand"].tensor.name, None),
                    )
                    - 1
                )
                * 32,
                "last",
            )
        outs = [
            (s, p.cb(like=self.refs[s.view.tensor.name], pages=2 * CHUNK))
            for s, _ in d.stores
        ]
        once = [op for op in d.operands if op.resident]
        held = [op for op in d.operands if self.held(op) and not op.resident]

        def row_page(k, op, row, j):
            tiles = v.host_value(Int("size", value=("row_tiles", op.tensor.name, None)))
            return self.ref(op).page(row * tiles + j)

        k = p.kernel("dm0")
        for op in once:
            cb = self.cbs[id(op)]
            if op.row:
                slot = k.reserve(cb, self.seg)
                with k.loop(wt, name="j") as j:
                    k.read(
                        row_page(k, op, v(op.view.index), j),
                        slot[j.bounded(0, self.seg)],
                    )
                k.reads_done()
                k.push(cb, self.seg)
            else:
                slot = k.reserve(cb, 1)
                k.read(self.ref(op).page(v(op.view.index)), slot)
                k.reads_done()
                k.push(cb, 1)
        with k.loop(count, name="u") as u:
            v.unit = k.let("r", start + u)
            for op in held:
                cb = self.cbs[id(op)]
                slot = k.reserve(cb, self.seg)
                with k.loop(wt, name="j") as j:
                    k.read(
                        row_page(k, op, v(op.view.index), j),
                        slot[j.bounded(0, self.seg)],
                    )
            if held:
                k.reads_done()
                for op in held:
                    k.push(self.cbs[id(op)], self.seg)
            for pass_ in passes:
                stream = [op for op in d.operands if self.streamed_in(op, pass_)]
                if not stream:
                    continue
                with k.block_loop(wt, CHUNK) as (done, n):
                    slots = [k.reserve(self.cbs[id(op)], CHUNK) for op in stream]
                    with k.loop(n) as i:
                        for op, slot in zip(stream, slots, strict=True):
                            k.read(row_page(k, op, v(op.view.index), done + i), slot[i])
                    k.reads_done()
                    for op in stream:
                        k.push(self.cbs[id(op)], CHUNK)

        c, units = self.compute(outs)
        for op in once:
            c.wait(self.cbs[id(op)], self.seg if op.row else 1)
        with c.loop(count, name="u"):
            for op in held:
                c.wait(self.cbs[id(op)], self.seg)
            if d.reduce is not None:
                self.pass_one(c, units, wt, last)
            stream = [op for op in d.operands if self.streamed_in(op, 2)]
            with c.block_loop(wt, CHUNK) as (done, n):
                for op in stream:
                    c.wait(self.cbs[id(op)], CHUNK)
                for _, cb in outs:
                    c.reserve(cb, CHUNK)
                with c.loop(n) as i:
                    j = c.let("j", done + i)
                    for (_, cb), (_, cases) in zip(outs, d.stores, strict=True):
                        self.scope(c, units, cases, i, j, cb, i)
                for _, cb in outs:
                    c.push(cb, CHUNK)
                for op in stream:
                    c.pop(self.cbs[id(op)], CHUNK)
            if d.reduce is not None:
                c.pop(self.cbs["stat"], 1)
            for op in held:
                c.pop(self.cbs[id(op)], self.seg)
        for op in once:
            c.pop(self.cbs[id(op)], self.seg if op.row else 1)

        w = p.kernel("dm1")
        with w.loop(count, name="u") as u:
            v.unit = w.let("r", start + u)
            with w.block_loop(wt, CHUNK) as (done, n):
                for _, cb in outs:
                    w.wait(cb, CHUNK)
                with w.loop(n) as i:
                    for s, cb in outs:
                        ref = self.refs[s.view.tensor.name]
                        w.write(cb, ref.page(v(s.view.index) * wt + done + i), index=i)
                w.flushed()
                for _, cb in outs:
                    w.pop(cb, CHUNK)
        w.writes_done()

    # A row kernel as the prologue of another program: rows held whole, the stored
    # row packed into a buffer of that program instead of written to DRAM.
    def prologue_buffers(self, wt):
        """Declares the row buffers for rows of `wt` tiles; returns the row's size, a
        sized value equal to `wt` (every page of the row buffer is packed)."""
        d, p, v = self.d, self.p, self.v
        self.seg = p.sized("seg", of=wt, exact=True, limit=self.seg_limit)
        for op in d.operands:
            if op.row and not self.held(op):
                fail("a row feeding a matmul is held whole in L1", op.node.line)
            pages = self.seg if op.row else 1
            self.cbs[id(op)] = p.cb(like=self.ref(op), pages=pages)
        self.last = None
        if d.reduce is not None:
            self.cbs["stat"] = p.cb(dtype=F32, pages=1, name="stat")
            name = d.reduce["operand"].tensor.name
            width = Int("size", value=("size", name, -1))
            tiles = Int("size", value=("row_tiles", name, None))
            self.last = v.host_value(width - (tiles - 1) * 32, "last")
        return self.seg

    def _row_page(self, op, row, j):
        tiles = self.v.host_value(
            Int("size", value=("row_tiles", op.tensor.name, None))
        )
        return self.ref(op).page(row * tiles + j)

    def prologue_read(self, k, wt, once):
        """Reader statements: the resident rows (`once`) or the unit's rows."""
        ops = [op for op in self.d.operands if op.resident == once]
        for op in ops:
            cb = self.cbs[id(op)]
            pages = self.seg if op.row else 1
            slot = k.reserve(cb, pages)
            if op.row:
                with k.loop(wt, name="j") as j:
                    page = self._row_page(op, self.v(op.view.index), j)
                    k.read(page, slot[j.bounded(0, self.seg)])
            else:
                k.read(self.ref(op).page(self.v(op.view.index)), slot)
        if ops:
            k.reads_done()
        for op in ops:
            k.push(self.cbs[id(op)], self.seg if op.row else 1)

    def prologue_wait(self, c, once, pop=False):
        """Waits for (or pops) the resident rows (`once`) or the unit's rows."""
        for op in self.d.operands:
            if op.resident == once:
                (c.pop if pop else c.wait)(self.cbs[id(op)], self.seg if op.row else 1)

    def prologue_compute(self, c, units, wt, cb_out):
        """The unit's row: the statistic pass, then each tile of the stored row
        packed into `cb_out` (`seg` pages, pushed at the end)."""
        d = self.d
        if d.reduce is not None:
            self.pass_one(c, units, wt, self.last)
        _, cases = d.stores[0]
        c.reserve(cb_out, self.seg)
        with c.loop(self.seg, name="j") as j:
            self.scope(c, units, cases, None, j, cb_out, j)
        c.push(cb_out, self.seg)
        if d.reduce is not None:
            c.pop(self.cbs["stat"], 1)

    def pass_one(self, c, units, wt, last):
        """The exact sum of squares of the reduced row into the statistic buffer,
        then the statistic chain on it."""
        d = self.d
        op = d.reduce["operand"]
        cb = self.cbs[id(op)]
        c.acquire()
        units.forget()
        units.setup("fill")
        c.op("fill", 1, value=0.0)
        c.op("fill", 2, value=0.0)
        units.unpack_as("transpose", cb)
        units.setup("stats")
        c.stats_clear()

        def update(at, j):
            c.transpose(cb, at, 0)
            n = km.select(j.eq(wt - 1), last, 32)
            c.stats_m2(0, count=n, dual=True)

        if self.held(op):
            with c.loop(wt, name="j") as j:
                update(j.bounded(0, self.seg), j)
        else:
            with c.block_loop(wt, CHUNK) as (done, n):
                c.wait(cb, CHUNK)
                with c.loop(n) as i:
                    update(i, c.let("j", done + i))
                c.pop(cb, CHUNK)
        c.stats_finalize(1, 1.0, dual=True)
        units.setup("transpose_dest")
        c.op("transpose_dest", 2)
        stat = d.stat
        free = iter((0, 1, 3))
        slots = [
            2 if getattr(x, "source", None) == "var" else next(free)
            for x in stat.operands
        ]
        units.sfpu = None
        self.run(c, units, stat, None, None, slots, out=2)
        cb_stat = self.cbs["stat"]
        c.reserve(cb_stat, 1)
        units.pack_into(cb_stat)
        c.pack(2, cb_stat)
        c.release()
        c.push(cb_stat, 1)
        c.wait(cb_stat, 1)


def lower(design, machine, block=1, mode="hold", depth=2, seg_limit=SEG_LIMIT):
    """(km.Program, Values) of `design` on `machine`: `block` tiles per barrier;
    rows "hold" (in L1, `depth` rows of at most `seg_limit` tiles) or "reread"."""
    if mode == "reread" and any(op.random for op in design.operands):
        fail(
            "a row read at another tile's place (a slice or kd.cat) is held in L1; "
            "this row does not fit"
        )
    low = _Lowering(design, machine, block, mode, depth)
    low.seg_limit = seg_limit
    program = low.build()
    return program, low.v
