"""Lowering: a Design becomes a metal-level program (`km.Program`), the same objects a
person builds with the metal level's builder. No C++ is written here.
"""

import contextlib
import os

import kohakutt.dsl.metal as km
from kohakutt.dsl.tile.ir import F32, Int, fail
from kohakutt.dsl.tile.plan import CHUNK, SEG_LIMIT, SLOTS, Leaf, Operand

# Row units dealt out to the cores in turn rather than in contiguous runs.
STRIDE = os.environ.get("KTT_TILE_STRIDE", "1") == "1"
# Tiles of a row store per register scope: 0 the most that fit, 1 one (a test knob).
GROUP = int(os.environ.get("KTT_TILE_GROUP", "0"))
# Rows (tiles) up to which a row store's row operands are kept merged per pair.
BAND_LIMIT = 128
# Device-profiler zones around the parts of row kernels (a profiler build's knob).
ZONES = os.environ.get("KTT_TILE_ZONES") == "1"


class Values:
    """The program's runtime values: each host integer of the IR one value, each
    scalar one value holding fp32 bits. `host` {name: Int}, `scalars` {name: Scalar}."""

    def __init__(self, program):
        self.p, self.host, self.scalars, self._names = program, {}, {}, {}
        self.unit = self.loop = None
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
        if expr.kind == "loop":
            return self.loop
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
        if op == "min":
            return km.min(a, b)
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


def zone(c, name):
    """A device-profiler zone around a part of the compute kernel when ZONES."""
    return c.zone(name) if ZONES else contextlib.nullcontext()


class _One:
    """One operand as a chain's operand list (to unpack it alone)."""

    def __init__(self, leaf):
        self.operands = [leaf]


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

    def deal(self, total):
        """Deals the `total` units out to the cores: in turn (core c takes c, c +
        cores, ...) when STRIDE, else in contiguous runs."""
        self._total = total
        self._dealt = (self.p.stride if STRIDE else self.p.split)(total)

    @contextlib.contextmanager
    def each_unit(self, b):
        """A loop of builder `b` over the running core's units; yields the unit."""
        first, other = self._dealt
        if STRIDE:
            with b.loop(self._total, step=other, name="r", start=first) as r:
                yield r
        else:
            with b.loop(other, name="u") as u:
                yield b.let("r", first + u)

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
        if any(cases[0].chain.prev is not None for _, cases in self.d.stores):
            return 1
        most = min(slots // len(cases[0].chain.operands) for _, cases in self.d.stores)
        return max(g for g in range(1, min(most, block) + 1) if block % g == 0)

    def compute(self, outs):
        """The compute builder, its startup done, and its unit-state tracker."""
        d, p = self.d, self.p
        cbs = [self.cbs[id(op)] for op in d.operands] + [cb for _, cb in outs]
        cbs += [self.cbs["stat"]] if "stat" in self.cbs else []
        cbs += getattr(self, "band_cbs", [])
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
        if chain.prev is not None:
            self.run(c, units, chain, at[0], None)
            units.pack_into(cb_out)
            c.pack(0, cb_out, at=at[0])
            c.release()
            return
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
        """Unpacks `chain`'s operands into `slots` (default 0 ..) and runs it; a
        staged chain runs its previous stage first, into slot 0."""
        if chain.prev is not None:
            if slots is not None or out != 0:
                fail("a chain of more blocks than register slots runs here alone")
            self.run(c, units, chain.prev, i, j)
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

    def shared(self, leaf):
        """True when operand `leaf` is the same tile for every tile of the unit."""
        src = leaf.source
        if src == "stat":
            return True
        if isinstance(src, Operand):
            return not src.row and src.resident
        return src != "var"

    def group_plan(self, cases):
        """(G, [(stage, roles)]) for G tiles of a store per register scope, or None:
        roles per operand ("res": the tile's own slot, holding its first per-tile
        operand or the previous stage; ("shared", slot); ("tile", slot))."""
        if len(cases) != 1 or GROUP == 1:
            return None
        for g in (4, 2):
            chain = cases[0].groups.get(g)
            if chain is None or (GROUP and g != GROUP):
                continue
            stages = []
            while chain is not None:
                stages.insert(0, chain)
                chain = chain.prev
            planned = []
            for k, stage in enumerate(stages):
                roles, nxt, seeded = [], g, k > 0
                for leaf in stage.operands:
                    if leaf.source == "var":
                        roles.append(("res",))
                    elif self.shared(leaf):
                        roles.append(("shared", nxt))
                        nxt += 1
                    elif not seeded:
                        roles.append(("res",))
                        seeded = True
                    else:
                        roles.append(("tile", nxt))
                        nxt += 1
                if nxt > SLOTS:
                    break
                planned.append((stage, roles))
            else:
                return g, planned
        return None

    def group_scope(self, c, units, plan, ats, cb_out):
        """One register scope of G tiles of a store: per stage the shared operands
        unpacked once, then per tile its own operands and the function; `ats` [(i, j)]
        per tile (i: the page of `cb_out`)."""
        g, stages = plan
        c.acquire()
        units.forget()
        for k, (stage, roles) in enumerate(stages):
            i0, j0 = ats[0]
            for leaf, role in zip(stage.operands, roles, strict=True):
                if role[0] == "shared":
                    self.unpack(c, units, _One(leaf), i0, j0, [role[1]])
            if k == 0:
                for t, (i, j) in enumerate(ats):
                    for leaf, role in zip(stage.operands, roles, strict=True):
                        if role == ("res",) and leaf.source != "var":
                            self.unpack(c, units, _One(leaf), i, j, [t])
            values = self.values(stage)
            for t, (i, j) in enumerate(ats):
                slots = []
                for leaf, role in zip(stage.operands, roles, strict=True):
                    if role[0] == "tile":
                        self.unpack(c, units, _One(leaf), i, j, [role[1]])
                    slots.append(t if role[0] == "res" else role[1])
                units.setup(stage.fn.init)
                c.run(stage.fn, *slots, out=t, values=values)
        units.pack_into(cb_out)
        for t, (i, _) in enumerate(ats):
            c.pack(t, cb_out, at=i)
        c.release()

    def store_tiles(self, c, units, outs, stores, n, done, wt, accs=None):
        """The `n` tiles of a chunk of each row store into pages 0 .. of its buffer:
        one function over 2 tiles with broadcasts read in place, else G tiles per
        register scope where the stages fit, else one; `accs` {store index: fp32 page
        its packs also accumulate into} (one per scope)."""
        accs = accs or {}
        bands = {k: b for k, b in getattr(self, "bands", {}).items() if k not in accs}
        plans = [
            None if k in accs or k in bands else self.group_plan(cases)
            for k, (_, cases) in enumerate(stores)
        ]
        single = [k for k, plan in enumerate(plans) if plan is None and k not in bands]
        if single:
            with c.loop(n) as i:
                j = c.let("j", done + i)
                for k in single:
                    (_, cb), (_, cases) = outs[k], stores[k]
                    if k in accs:
                        self.scope(c, units, cases, i, j, cb, i, accs[k])
                    else:
                        self.scope(c, units, cases, i, j, cb, i)
        for k, plan in enumerate(plans):
            if plan is not None:
                g = plan[0]
                with c.block_loop(n, g, names=("t0", "m")) as (t0, _):
                    ats = self.group_ats(c, t0, g, done, wt)
                    self.group_scope(c, units, plan, ats, outs[k][1])
        for k, band in bands.items():
            with c.block_loop(n, 2, names=("t0", "m")) as (t0, _):
                ats = self.group_ats(c, t0, 2, done, wt)
                self.band_scope(c, units, band, ats, done + t0, outs[k][1])

    def group_ats(self, c, t0, g, done, wt):
        """[(i, j)] of the G tiles from chunk page `t0`: past the row's last tile, the
        last tile again (computed, packed, not written)."""
        out = []
        for t in range(g):
            i = (t0 + t).bounded(0, CHUNK)
            j = c.let(f"jg{t}", km.min(done + t0 + t, wt - 1))
            out.append((i, j))
        return out

    # Row stores as one function over 2 tiles (km.sfpu along= operands): the column
    # operands in one slot (S), the row operands of a pair of tiles in another (P).
    def band_buffers(self, stores, skip=()):
        """{store index: band plan} for the stores (but `skip`) whose chain runs
        banded and whose row operands are resident rows; declares their P (resident,
        a page per pair of tiles) and S (per unit, two column operands) buffers."""
        p, self.bands = self.p, {}
        for k, (_, cases) in enumerate(stores):
            chain = cases[0].bands if len(cases) == 1 and k not in skip else None
            if chain is None:
                continue
            rows = [x for x in chain.operands if x.kind == "bcast_row"]
            cols = [x for x in chain.operands if x.kind == "bcast_col"]
            if not all(
                isinstance(x.source, Operand) and x.source.resident for x in rows
            ):
                continue
            if rows and self.seg_limit > BAND_LIMIT:
                continue
            band = {"chain": chain, "rows": rows, "cols": cols}
            if rows:
                # In the row operands' format (the values only move).
                dtype = rows[0].source.tensor.dtype
                if any(x.source.tensor.dtype != dtype for x in rows):
                    dtype = F32
                pages = (self.seg + 1) // 2
                band["P"] = p.cb(dtype=dtype, pages=pages, name=f"rp{k}")
            if len(cols) == 2:
                band["S"] = p.cb(dtype=F32, pages=1, name=f"cs{k}")
            self.bands[k] = band
        return [
            cb for b in self.bands.values() for cb in (b.get("P"), b.get("S")) if cb
        ]

    def band_rows(self, c, units, wt):
        """P of every banded store, once: per pair of tiles, each row operand's tile
        broadcast and put in its band (row operand r of tile t in band 2 t + r)."""
        for band in self.bands.values():
            cb = band.get("P")
            if cb is None:
                continue
            # Every page of the buffer (pairs past the row repeat its last tile).
            c.reserve(cb, (self.seg + 1) // 2)
            with c.loop((self.seg + 1) // 2, name="q") as q:
                c.acquire()
                units.forget()
                for t in range(2):
                    # The last tile (0 on a split's member without tiles).
                    j = c.let(f"jp{t}", km.min(q * 2 + t, wt - km.min(wt, 1)))
                    for r, leaf in enumerate(band["rows"]):
                        self.unpack(c, units, _One(leaf), None, j, [r])
                    units.setup("mul")
                    for r in range(len(band["rows"])):
                        c.run(km.sfpu.put_rows(2 * t + r), r, out=3)
                units.forget()
                units.pack_into(cb)
                c.pack(3, cb, at=q.bounded(0, (self.seg + 1) // 2))
                c.release()
            c.push(cb, (self.seg + 1) // 2)
            c.wait(cb, (self.seg + 1) // 2)

    def band_cols(self, c, units):
        """S of every banded store with two column operands, per unit: the two column
        broadcasts joined (first in columns 0-15, second in 16-31)."""
        for band in self.bands.values():
            cb = band.get("S")
            if cb is None:
                continue
            c.acquire()
            units.forget()
            for r, leaf in enumerate(band["cols"]):
                self.unpack(c, units, _One(leaf), None, None, [r])
            units.setup("mul")
            c.run(km.sfpu.join_cols(), 0, 1, out=0)
            c.reserve(cb, 1)
            units.pack_into(cb)
            c.pack(0, cb)
            c.release()
            c.push(cb, 1)
            c.wait(cb, 1)

    def band_pop(self, c, resident=False):
        """Pops the banded stores' S pages (per unit) or P pages (`resident`)."""
        for band in self.bands.values():
            if resident and "P" in band:
                c.pop(band["P"], (self.seg + 1) // 2)
            if not resident and "S" in band:
                c.pop(band["S"], 1)

    def band_scope(self, c, units, band, ats, j0, cb_out):
        """One register scope of 2 tiles of a banded store: S (or the one column
        operand) in slot 2, the pair's P page in slot 3, the tiles in slots 0 and 1,
        one function call over both."""
        chain = band["chain"]
        c.acquire()
        units.forget()
        if "S" in band:
            units.unpack_as("copy", band["S"])
            c.copy(band["S"], 0, 2)
        elif band["cols"]:
            self.unpack(c, units, _One(band["cols"][0]), None, None, [2])
        if "P" in band:
            units.unpack_as("copy", band["P"])
            pair = (j0 // 2).bounded(0, (self.seg + 1) // 2)
            c.copy(band["P"], pair, 3)
        tile = next(
            x for x in chain.operands if x.kind not in ("bcast_col", "bcast_row")
        )
        for t, (i, j) in enumerate(ats):
            self.unpack(c, units, _One(tile), i, j, [t])
        slots = [
            2 if x.kind == "bcast_col" else 3 if x.kind == "bcast_row" else 0
            for x in chain.operands
        ]
        units.setup(chain.fn.init)
        c.run(chain.fn, *slots, out=0, values=self.values(chain), tiles=2)
        units.pack_into(cb_out)
        for t, (i, _) in enumerate(ats):
            c.pack(t, cb_out, at=i)
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
        self.deal(units_value)
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
        self.bands, self.band_cbs = {}, []
        if d.dest == "fp32":
            self.band_cbs = self.band_buffers(d.stores)
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
        with self.each_unit(k) as unit:
            v.unit = unit
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
        self.band_rows(c, units, wt)
        with self.each_unit(c):
            with zone(c, "W"):
                for op in held:
                    c.wait(self.cbs[id(op)], self.seg)
            if d.reduce is not None:
                with zone(c, "S"):
                    self.pass_one(c, units, wt, last)
            self.band_cols(c, units)
            stream = [op for op in d.operands if self.streamed_in(op, 2)]
            with zone(c, "Y"), c.block_loop(wt, CHUNK) as (done, n):
                for op in stream:
                    c.wait(self.cbs[id(op)], CHUNK)
                with zone(c, "O"):
                    for _, cb in outs:
                        c.reserve(cb, CHUNK)
                self.store_tiles(c, units, outs, d.stores, n, done, wt)
                for _, cb in outs:
                    c.push(cb, CHUNK)
                for op in stream:
                    c.pop(self.cbs[id(op)], CHUNK)
            self.band_pop(c)
            if d.reduce is not None:
                c.pop(self.cbs["stat"], 1)
            for op in held:
                c.pop(self.cbs[id(op)], self.seg)
        self.band_pop(c, resident=True)
        for op in once:
            c.pop(self.cbs[id(op)], self.seg if op.row else 1)

        w = p.kernel("dm1")
        with self.each_unit(w) as unit:
            v.unit = unit
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
