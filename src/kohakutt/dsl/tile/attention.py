"""Kernels with values carried across a kd.range loop over blocks of rows (flash
attention): per unit a block of query tile rows, per iteration a block of key tiles;
two matrix-unit products (`q @ k.T`, `p @ v`), row reductions of the scores, column
values and the carried state (running maximum and sum, the output accumulator)."""

import contextlib
import os
from dataclasses import dataclass, field

import kohakutt.dsl.metal as km
from kohakutt.dsl.tile.columns import _Some
from kohakutt.dsl.tile.ir import BF16, F32, Int, Node, fail
from kohakutt.dsl.tile.lower import Values, _Units
from kohakutt.dsl.tile.plan import SLOTS, Leaf, _Compiler

# Value of a masked score: far below any score, finite (no inf - inf).
MASKED = -1e30
# Mask tiles: none, keys past the last key, keys after the query, both, all.
MASK_TILES = 5
# Most tiles along the head dimension (D <= 256).
DT_LIMIT = 8
# Device-profiler zones around each phase of a step (a profiler build).
ZONES = False
# Matmul block calls over several key / head tiles.
BLOCKS = True
# Blocks of at most 3 key tiles: scores, row max and p in one register scope, the
# row sum as p @ ones beside p @ v.
FUSED = True
# In that scope, p computed on the pack thread after the commit.
PACK_EXP = os.environ.get("KTT_TILE_PACK_EXP", "0") == "1"
# In that step, the rescales skipped when no row's max moved (alpha exactly 1).
LAZY = os.environ.get("KTT_TILE_LAZY", "0") == "1"
# Two query rows per pass: both score scopes, then each row's rest.
PIPE = os.environ.get("KTT_TILE_PIPE", "1") == "1"


def score_masks(keys, device=None):
    """The mask tiles kd.mask_scores reads, for `keys` keys: a [5, 32, 32] fp32
    tensor (0, keys past the last key's column, keys after the query, both, all)."""
    import torch

    tiles = torch.zeros(MASK_TILES, 32, 32)
    col = torch.arange(32).view(1, 32).expand(32, 32)
    row = torch.arange(32).view(32, 1).expand(32, 32)
    last = keys % 32 or 32
    tiles[1][col >= last] = MASKED
    tiles[2][col > row] = MASKED
    tiles[3][(col >= last) | (col > row)] = MASKED
    tiles[4][:] = MASKED
    return tiles if device is None else tiles.to(device)


@dataclass(eq=False)
class Value:
    """A per-row (column) value: a row reduction ("max" / "sum") of a tile block, or
    a vector chain on column values; `phase` 0 when it needs only the scores, 1 when
    it reads the probabilities' reduction."""

    node: Node
    kind: str
    phase: int = 0
    chain: object = None
    of: object = None


@dataclass(eq=False)
class AttentionDesign:
    """The roles of a traced flash kernel: the query, key and value blocks, the score
    and probability chains, the column values, the carries and the stored chain."""

    ir: object
    q: Node
    k: Node
    v: Node
    score: object = None
    prob: object = None
    mask: Node | None = None
    values: list = field(default_factory=list)
    carries: dict = field(default_factory=dict)
    acc: Node | None = None
    rescale: object = None
    store: object = None
    rows: int = 0
    keys: int = 0
    scalars: dict = field(default_factory=dict)
    checks: list = field(default_factory=list)
    form: str = "attention"
    dest: str = "fp32"
    operands: list = field(default_factory=list)
    stores: list = field(default_factory=list)
    reduce: object = None
    stat: object = None
    share: object = None
    # The online-softmax roles of a step in one register scope (_flash_roles), or None.
    flash: object = None

    def describe(self):
        """Report lines of the design's roles."""
        out = [
            f"scores    {self.score.fn.name}: q @ k.T then {len(self.score.ops)} ops"
            + (", masked" if self.mask is not None else ""),
            f"probs     {self.prob.fn.name}: on the scores and column values",
        ]
        reads = ("of the scores", "of the probabilities")
        out += [f"column    {v.kind} {reads[v.phase]}" for v in self.values]
        out += [
            f"carried   {kind} value, start {node.attrs['init']}"
            for node, kind in self.carries.items()
        ]
        return out


def _count(view, line):
    """(first, count, static most) of a block view x.tiles[a:a + n, :]."""
    lo, hi = view.cols
    if hi.kind != "op" or hi.value != "+" or hi.args[0].text() != lo.text():
        fail("a block of tile rows is x.tiles[a : a + n, :]", line)
    n = hi.args[1]
    most = _static(n)
    if most is None:
        fail(
            "a block's tile row count n is static or kd.minimum(B, ...) with B static",
            line,
        )
    return lo, n, most


def _static(n):
    if n.kind == "const":
        return n.value
    if n.kind == "op" and n.value == "min":
        found = [x for x in (_static(a) for a in n.args) if x is not None]
        return min(found) if found else None
    return None


def _scaled(node, product):
    """True when `node` is `product` times runtime or constant scalars."""
    if node is product:
        return True
    if node.kind != "mul":
        return False
    a, b = node.args
    if b.kind in ("param", "const", "size"):
        return _scaled(a, product)
    return a.kind in ("param", "const", "size") and _scaled(b, product)


def _walk(node, seen=None):
    seen = set() if seen is None else seen
    if id(node) in seen:
        return []
    seen.add(id(node))
    return [n for a in node.args for n in _walk(a, seen)] + [node]


def design_attention(ir, dest=None):
    """The AttentionDesign of a traced kernel with carried values."""
    if dest == "bf16":
        fail(f"{ir.name}: carried values need fp32 registers")
    if (
        ir.accumulate is not None
        or ir.grid.cols is not None
        or (ir.grid.group or 1) > 1
    ):
        fail(f"{ir.name}: carried values run on kd.grid(n) without an accumulator")
    nodes = ir.nodes()
    products = [n for n in nodes if n.kind == "matmul"]
    scores = [n for n in products if n.args[1].kind == "transpose"]
    values_mm = [n for n in products if n.args[1].kind != "transpose"]
    if len(scores) != 1 or len(values_mm) != 1:
        fail(f"{ir.name}: a carried-value kernel has s = q @ k.T and p @ v")
    sm, pv = scores[0], values_mm[0]
    q, k = sm.args[0], sm.args[1].args[0]
    v = pv.args[1]
    for node, what in ((q, "q"), (k, "k"), (v, "v")):
        if node.kind != "load" or node.attrs["view"].kind != "rows":
            fail(f"{what} of attention is kd.load(x.tiles[a : a + n, :])", node.line)
    out = AttentionDesign(ir, q, k, v)
    _, _, out.rows = _count(q.attrs["view"], q.line)
    _, _, out.keys = _count(k.attrs["view"], k.line)
    if out.keys > 16 or out.rows > 16:
        fail("blocks of at most 16 query rows and 16 key tiles", k.line)
    if k.attrs["view"].cols[0].text() != v.attrs["view"].cols[0].text():
        fail("k and v blocks read the same key rows", v.line)
    carries = {id(c): c for c, _ in ir.carries}
    nexts = {id(c): value for c, value in ir.carries}
    if not carries:
        fail(f"{ir.name}: kd.carry values are updated with .next(...)")
    reduces = [n for n in nodes if n.kind == "reduce"]
    maxes = [n for n in reduces if n.attrs["op"] == "max"]
    if len(maxes) != 1:
        fail("one kd.reduce_max of the scores per iteration")
    score_root = maxes[0].args[0]
    prob_root = pv.args[0].args[0] if pv.args[0].kind == "cast" else pv.args[0]

    # Carries holding tiles (the accumulator) are those whose next reads p @ v.
    for cid, c in carries.items():
        tile = any(n is pv for n in _walk(nexts[cid]))
        out.carries[c] = "tile" if tile else "col"
    accs = [c for c, kind in out.carries.items() if kind == "tile"]
    if len(accs) != 1:
        fail("one carried accumulator: acc.next(acc * a.col() + p @ v)")
    out.acc = accs[0]

    def shape(node):
        k_ = node.kind
        if k_ in ("const", "param", "size"):
            return "scalar"
        if k_ == "carry":
            return out.carries[node]
        if k_ in ("load", "matmul", "transpose", "mask"):
            return "tile"
        if k_ == "reduce":
            return "col"
        if k_ == "col":
            return "tile"
        args = [shape(a) for a in node.args]
        if "tile" in args:
            return "tile"
        return "col" if "col" in args else "scalar"

    # Column values: reductions, the .col() operands and the carries' next values.
    wanted = {}
    for n in nodes:
        if n.kind == "reduce":
            wanted[id(n)] = n
        inner = n.args[0] if n.kind == "col" else None
        if inner is not None and shape(inner) == "col" and inner.kind != "carry":
            wanted[id(inner)] = inner
    for c, kind in out.carries.items():
        if kind == "col":
            nxt = nexts[id(c)]
            if nxt.kind == "carry":
                fail("a carry's next value is computed", nxt.line)
            wanted[id(nxt)] = nxt
    compiler = _Compiler(ir.name, shape, out.scalars)
    made = {}
    stops = set(wanted) | {id(c) for c in out.carries}

    def leaf_of(root):
        def fn(node, offset, kind):
            if node.kind == "carry":
                return Leaf(node, kind)
            if id(node) in made and (node is not root or node.kind == "reduce"):
                return Leaf(made[id(node)], kind)
            if node.kind == "mask_tile":
                return Leaf("mask", "copy")
            if node is sm:
                return Leaf("raw", "copy")
            if node is score_root and root is not score_root:
                return Leaf("score", "copy")
            return None

        return fn

    for n in nodes:
        if id(n) not in wanted:
            continue
        if n.kind == "reduce":
            if n.attrs["op"] == "max":
                if n.args[0] is not score_root:
                    fail("kd.reduce_max reads the scores", n.line)
                value = Value(n, "max", 0, of="score")
            else:
                if n.args[0] is not prob_root:
                    fail("kd.reduce_sum reads the probabilities p of p @ v", n.line)
                value = Value(n, "sum", 1, of="prob")
        else:
            here = stops - {id(n)}
            chain = compiler.chain(n, leaf_of(n), stops=here)
            phase = max(
                [x.source.phase for x in chain.operands if isinstance(x.source, Value)]
                + [0]
            )
            value = Value(n, "chain", phase, chain)
        made[id(n)] = value
        out.values.append(value)

    # The scores: a chain on q @ k.T and scalars; a mask tile is the product's
    # starting value, so a masked score is (q @ k.T) times scalars.
    root = score_root
    if root.kind == "mask":
        out.mask = root
        root = root.args[0]
        if not _scaled(root, sm):
            fail("a masked score is (q @ k.T) * scale", out.mask.line)
    out.score = compiler.chain(root, leaf_of(score_root), stops={id(sm)})
    if [x.source for x in out.score.operands] != ["raw"]:
        fail("the scores read q @ k.T and scalars only", score_root.line)
    # The probabilities: a chain on the scores and column values.
    out.prob = compiler.chain(
        prob_root, leaf_of(prob_root), stops=set(made) | {id(score_root)}
    )
    if out.prob.prev is not None or not all(
        (isinstance(x.source, Value) and x.source.phase == 0) or x.source == "score"
        for x in out.prob.operands
    ):
        fail("p reads the scores and column values of the scores", prob_root.line)
    # The accumulator: acc.next(X + p @ v), X a chain on acc and column values.
    nxt = nexts[id(out.acc)]
    if nxt.kind != "add" or pv not in nxt.args:
        fail("acc.next(acc * a.col() + p @ v): the product is added last", nxt.line)
    x = nxt.args[0] if nxt.args[1] is pv else nxt.args[1]
    out.rescale = compiler.chain(x, leaf_of(x), stops=set(made) | {id(out.acc)})
    if out.rescale.prev is not None or len(out.rescale.operands) > SLOTS - 1:
        fail("the accumulator's rescale reads at most 3 blocks", nxt.line)
    if len(ir.stores) != 1:
        fail(f"{ir.name}: a carried-value kernel stores one block")
    (store,) = ir.stores
    if store.view.kind != "rows":
        fail("the stored block is out.tiles[a : a + n, :]", store.line)
    if store.view.cols[0].text() != q.attrs["view"].cols[0].text():
        fail("the stored block is the query block's rows", store.line)
    out.store = compiler.chain(
        store.value,
        leaf_of(store.value),
        stops=set(made) | {id(c) for c in carries.values()},
    )
    if out.store.prev is not None:
        fail("the stored chain fits 4 register slots", store.line)
    for x in out.store.operands:
        src = x.source
        if isinstance(src, Node) and src.kind == "carry":
            continue
        if not isinstance(src, Value):
            fail("the stored block reads the carried values after the loop", store.line)
    out.stores = [(store, None)]
    if any(n.attrs["view"].tensor.dtype != BF16 for n in (q, k, v)):
        fail("q, k and v are bf16 (the matrix unit's row reductions)", sm.line)
    out.flash = _flash_roles(out, nexts)
    return out


def _flash_roles(d, nexts):
    """The online-softmax roles when a step fits one register scope (at most 3 key
    tiles per block): {"m", "den": carries, "max", "m_new", "alpha", "sum", "den_new":
    Values}, else None. m_new = max(m, reduce_max(s)); p reads s and m_new; the
    accumulator's rescale reads alpha; den.next(den * alpha + reduce_sum(p))."""
    if d.keys > SLOTS - 1 or not FUSED:
        return None
    by = {}
    for value in d.values:
        by.setdefault(value.kind, []).append(value)
    if len(by.get("max", ())) != 1 or len(by.get("sum", ())) != 1:
        return None
    mx, total = by["max"][0], by["sum"][0]
    node_of = {id(v.node): v for v in d.values}
    roles = {"max": mx, "sum": total}
    for c, kind in d.carries.items():
        if kind != "col":
            continue
        nxt = nexts[id(c)]
        pair = {id(a) for a in nxt.args} == {id(c), id(mx.node)}
        if nxt.kind in ("max", "lagmax") and pair:
            # Either takes max(m, row max) for the row max: the same result.
            if nxt.kind == "lagmax" and nxt.args[0] is not c:
                return None
            roles["m"], roles["m_new"] = c, node_of.get(id(nxt))
            roles["lag"] = nxt.kind == "lagmax"
        elif nxt.kind == "add" and any(a is total.node for a in nxt.args):
            other = nxt.args[0] if nxt.args[1] is total.node else nxt.args[1]
            if other.kind == "mul" and any(a is c for a in other.args):
                alpha = other.args[0] if other.args[1] is c else other.args[1]
                roles["den"], roles["den_new"] = c, node_of.get(id(nxt))
                roles["alpha"] = node_of.get(id(alpha))
    need = ("m", "m_new", "den", "den_new", "alpha")
    if any(roles.get(k) is None for k in need):
        return None
    alpha, m_new = roles["alpha"], roles["m_new"]
    rescale = [x.source for x in d.rescale.operands]
    if len(rescale) != 2 or not all(
        any(s is x for s in rescale) for x in (d.acc, alpha)
    ):
        return None
    prob = [x.source for x in d.prob.operands]
    if len(prob) != 2 or "score" not in prob or not any(s is m_new for s in prob):
        return None
    if set(d.values) != {mx, total, m_new, alpha, roles["den_new"]}:
        return None
    # Up to 2 key tiles: m, m_new and alpha also fit the score scope.
    roles["inscope"] = d.keys <= SLOTS - 2
    if roles["lag"] and not roles["inscope"]:
        return None
    return roles


class _Lower:
    """The km.Program of an AttentionDesign."""

    def __init__(self, d, machine):
        self.d = d
        ir = d.ir
        self.p = p = km.Program(ir.name, machine)
        self.v = Values(p)
        self.refs = {}
        for t in ir.tensors:
            used = (
                t.output
                or any(
                    n.kind == "load" and n.attrs["view"].tensor is t for n in ir.nodes()
                )
                or (d.mask is not None and d.mask.attrs["tensor"] is t)
            )
            if used:
                self.refs[t.name] = (p.output if t.output else p.input)(t.name, t.dtype)

    def build(self):
        d, p, v = self.d, self.p, self.v
        ir = d.ir
        R, KB = d.rows, d.keys
        qv, kv, vv = (n.attrs["view"] for n in (d.q, d.k, d.v))
        q0, nq, _ = _count(qv, d.q.line)
        k0, nk, _ = _count(kv, d.k.line)
        dt_value = v.host_value(Int("size", value=("row_tiles", qv.tensor.name, None)))
        dt = p.sized("dt", of=dt_value, exact=True, limit=DT_LIMIT)
        p.limit(dt_value, low=1, high=DT_LIMIT)
        units = v.host_value(ir.grid.count)
        start, count = p.split(units)
        q_ref, k_ref, v_ref = (self.refs[x.tensor.name] for x in (qv, kv, vv))
        out_ref = self.refs[d.stores[0][0].view.tensor.name]
        cb_q = p.cb(like=q_ref, pages=R * dt, name="q")
        cb_k = p.cb(like=k_ref, pages=KB * dt * 2, name="k")
        cb_v = p.cb(like=v_ref, pages=KB * dt * 2, name="v")
        # Two rows' p (and m_new, a carry's new page) ahead with the pipelined step.
        self.pipe = PIPE and d.flash is not None and d.flash["inscope"]
        ahead = 2 if self.pipe else 1
        cb_p = p.cb(dtype=v_ref.dtype, pages=ahead * KB, name="p")
        cb_ones = p.cb(dtype=BF16, pages=1, name="ones")
        cb_out = p.cb(like=out_ref, pages=2, name="o")
        flash = d.flash
        self.cbs = {"prob": cb_p}
        if flash is None:
            # The scores in fp32, and in bf16 for the matrix unit's row max.
            self.cbs["score"] = p.cb(dtype=F32, pages=KB, name="s")
            self.cbs["max"] = p.cb(dtype=BF16, pages=KB, name="smax")
        # A row of the accumulator: groups of `width` tiles (one register scope
        # each; at most 2: whole groups for even head dims), dtp pages.
        self.width = min(2, SLOTS - (len(d.rescale.operands) - 1))
        tiles = Int("size", value=("row_tiles", qv.tensor.name, None))
        groups_value = v.host_value((tiles + self.width - 1) // self.width)
        groups = p.sized("dtg", of=groups_value, exact=True, limit=DT_LIMIT)
        dtp = self.width * groups
        self.dtp = dtp
        queue = {}
        for c, kind in d.carries.items():
            pages = (R + 1) * (dtp if kind == "tile" else 1)
            queue[c] = p.cb(dtype=F32, pages=pages, name=f"carry{len(queue)}")
        temps = {}
        kept = d.values if flash is None else [flash["m_new"], flash["alpha"]]
        for value in kept:
            temps[value] = p.cb(dtype=F32, pages=ahead, name=f"col{len(temps)}")
        self.flag = None
        if flash is not None and LAZY:
            self.cb_flag = p.cb(dtype=F32, pages=1, name="moved")
        cb_mask = None
        if d.mask is not None:
            m_ref = self.refs[d.mask.attrs["tensor"].name]
            cb_mask = p.cb(like=m_ref, pages=MASK_TILES, name="masks")
        self.queue, self.temps, self.cb_mask = queue, temps, cb_mask
        self.dt = dt

        # Reader: the masks once; per unit the query block, per step k and v blocks.
        r = p.kernel("dm0")
        if cb_mask is not None:
            slot = r.reserve(cb_mask, MASK_TILES)
            for i in range(MASK_TILES):
                r.read(m_ref.page(i), slot[i])
            r.reads_done()
            r.push(cb_mask, MASK_TILES)
        with r.loop(count, name="u") as u:
            v.unit = r.let("unit", start + u)
            first, rows = r.let("q0", v(q0)), r.let("nq", v(nq))
            slot = r.reserve(cb_q, R * dt)
            with r.loop(rows, name="i") as i, r.loop(dt, name="t") as t:
                at = (i * dt + t).bounded(0, R * dt)
                r.read(q_ref.page((first + i) * dt + t), slot[at])
            r.reads_done()
            r.push(cb_q, R * dt)
            with r.loop(v(ir.loop.count), name="j") as j:
                v.loop = j
                kfirst, keys = r.let("k0", v(k0)), r.let("nk", v(nk))
                # K tile (c, t) at t * KB + c, V tile (c, t) at c * dt + t; tiles
                # past the block's keys repeat its last one.
                for ref, cb in ((k_ref, cb_k), (v_ref, cb_v)):
                    blk = r.reserve(cb, KB * dt)
                    with r.loop(KB, name="c") as c, r.loop(dt, name="t") as t:
                        place = t * KB + c if cb is cb_k else c * dt + t
                        at = place.bounded(0, KB * dt)
                        row = kfirst + km.min(c, keys - 1)
                        r.read(ref.page(row * dt + t), blk[at])
                r.reads_done()
                r.push(cb_k, KB * dt)
                r.push(cb_v, KB * dt)

        # Compute.
        fp32 = [cb for cb in p.cbs if cb.dtype == F32]
        c = p.kernel("compute", fp32_dest=True, unpack_fp32=fp32, exact=True)
        c.startup(cb_q, cb_k, cb_out)
        units_ = _Units(c, cb_q, False)
        self.c, self.units = c, units_
        # Ones in each reduced buffer's format (the matrix unit's two sources agree).
        self.ones = {BF16: cb_ones}
        for ones in (cb_ones,):
            c.acquire()
            units_.forget()
            units_.setup("fill")
            c.op("fill", 0, value=1.0)
            c.reserve(ones, 1)
            units_.pack_into(ones)
            c.pack(0, ones)
            c.release()
            c.push(ones, 1)
            c.wait(ones, 1)
        if cb_mask is not None:
            c.wait(cb_mask, MASK_TILES)
        with c.loop(count, name="u") as u:
            v.unit = c.let("unit", start + u)
            rows = c.let("nq", v(nq))
            c.wait(cb_q, R * dt)
            # Carries start at their constants, one row after the other.
            with c.loop(rows, name="i"):
                for cv, kind in d.carries.items():
                    if kind == "tile":
                        self.fill_tiles(queue[cv], cv.attrs["init"])
                    else:
                        self.fill(queue[cv], 1, cv.attrs["init"])
            with c.loop(v(ir.loop.count), name="j") as j:
                v.loop = j
                keys = c.let("nk", v(nk))
                c.wait(cb_k, KB * dt)
                c.wait(cb_v, KB * dt)
                if flash is None:
                    with c.loop(rows, name="i") as i:
                        self.step(i, keys, cb_q, cb_k, cb_v)
                else:
                    self.flash_rows(rows, keys, cb_q, cb_k, cb_v)
                c.pop(cb_k, KB * dt)
                c.pop(cb_v, KB * dt)
            with c.loop(rows, name="i"):
                self.finish(cb_out)
            c.pop(cb_q, R * dt)
        c.pop(cb_ones, 1)
        if cb_mask is not None:
            c.pop(cb_mask, MASK_TILES)

        # Writer.
        w = p.kernel("dm1")
        with w.loop(count, name="u") as u:
            v.unit = w.let("unit", start + u)
            first, rows = w.let("q0", v(q0)), w.let("nq", v(nq))
            with w.loop(rows, name="i") as i, w.loop(dt, name="t") as t:
                w.wait(cb_out, 1)
                w.write(cb_out, out_ref.page((first + i) * dt + t))
                w.flushed()
                w.pop(cb_out, 1)
        w.writes_done()
        return p

    # Pieces of the compute kernel.
    def fill_tiles(self, cb, value):
        """Pushes a row of the accumulator (its tiles in groups of `width` pages)
        holding `value` everywhere."""
        c, units, width = self.c, self.units, self.width
        with c.block_loop(self.dt, width, names=("t0", "nt")):
            c.reserve(cb, width)
            c.acquire()
            units.forget()
            units.setup("fill")
            c.op("fill", 0, value=value)
            units.pack_into(cb)
            for s in range(width):
                c.pack(0, cb, at=s)
            c.release()
            c.push(cb, width)

    def pop_tiles(self, cb):
        """Pops a row of the accumulator."""
        with self.c.block_loop(self.dt, self.width, names=("t1", "n1")):
            self.c.pop(cb, self.width)

    def fill(self, cb, n, value):
        """Pushes `n` pages of `cb` holding `value` everywhere."""
        c, units = self.c, self.units
        with c.loop(n, name="t"):
            c.reserve(cb, 1)
            c.acquire()
            units.forget()
            units.setup("fill")
            c.op("fill", 0, value=value)
            units.pack_into(cb)
            c.pack(0, cb)
            c.release()
            c.push(cb, 1)

    def unpack(self, chain, slots, at):
        """Unpacks `chain`'s operands into `slots`; `at` {"tile": index of a tile
        operand (the score or accumulator tile), "mask": mask tile index}."""
        c, units = self.c, self.units
        for slot, leaf in zip(slots, chain.operands, strict=True):
            src = leaf.source
            if src == "raw":
                continue
            if src == "mask":
                cb, index = self.cb_mask, at["mask"]
            elif src == "score":
                cb, index = self.cbs["score"], at["tile"]
            elif isinstance(src, Value):
                cb, index = self.temps[src], 0
            elif isinstance(src, Node) and src.kind == "carry":
                cb = self.queue[src]
                index = at["tile"] if self.d.carries[src] == "tile" else 0
            else:
                fail(f"no operand {src!r} in this kernel")
            units.unpack_as(leaf.kind, cb)
            if leaf.kind == "copy":
                c.copy(cb, index, slot)
            else:
                c.bcast(cb, index, slot, leaf.kind[len("bcast_") :])

    def run(self, chain, slots, out):
        units = self.units
        units.setup(chain.fn.init)
        values = tuple(self.v.scalar(s) for s in chain.values)
        self.c.run(chain.fn, *slots, out=out, values=values)

    def matmul_format(self, a, b):
        """The unpacker set up for products a @ b (b through the first source)."""
        c, units = self.c, self.units
        c.format(srca=b, srcb=a)
        units.srca, units.srcb, units.unpack, units.sfpu = b.dtype, a.dtype, None, None

    def mask_index(self, i, cc):
        """The mask tile of score tile (row i, key tile cc) of this step."""
        m = self.d.mask.attrs
        v = self.v
        qi, kj = v(m["q"]) + i, v(m["k"]) + cc
        last = v.host_value((m["keys"] + 31) // 32 - 1)
        ragged = v.host_value((m["keys"] + 31) // 32 * 32 - m["keys"])
        tail = km.select(kj.eq(last), km.select(ragged.eq(0), 0, 1), 0)
        if not m["causal"]:
            return tail
        diag = km.select(kj.eq(qi), 2, 0)
        return km.select(qi < kj, 4, tail + diag)

    def step(self, i, keys, cb_q, cb_k, cb_v):
        """One query row against the step's key block."""
        d, c = self.d, self.c
        cb_s, cb_sb, cb_p = self.cbs["score"], self.cbs["max"], self.cbs["prob"]
        zone = c.zone if ZONES else (lambda name: contextlib.nullcontext())
        for cb in (cb_s, cb_sb):
            c.reserve(cb, d.keys)
        with zone("S"):
            for g0 in range(0, d.keys, SLOTS):
                self.scores(i, keys, g0, min(SLOTS, d.keys - g0), cb_q, cb_k)
        for cb in (cb_s, cb_sb):
            c.push(cb, d.keys)
            c.wait(cb, d.keys)
        with zone("A"):
            self.values([x for x in d.values if x.phase == 0])
        c.pop(cb_sb, d.keys)
        c.reserve(cb_p, d.keys)
        width = SLOTS - (len(d.prob.operands) - 1)
        with zone("P"):
            for g0 in range(0, d.keys, width):
                self.probs(g0, min(width, d.keys - g0))
        c.push(cb_p, d.keys)
        c.wait(cb_p, d.keys)
        c.pop(cb_s, d.keys)
        with zone("B"):
            self.values([x for x in d.values if x.phase == 1])
        with zone("V"):
            self.accumulate(keys, cb_p, cb_v)
        c.pop(cb_p, d.keys)
        # The row's column values are done: the old carries leave the queues.
        for cv, kind in d.carries.items():
            if kind == "col":
                c.pop(self.queue[cv], 1)
        for value in d.values:
            c.pop(self.temps[value], 1)

    def flash_rows(self, rows, keys, cb_q, cb_k, cb_v):
        """The block's query rows in the online-softmax roles: two rows at a time,
        both rows' score scopes first (the second's on the math thread while the
        packer finishes the first's), then each row's rest."""
        c = self.c
        if not self.pipe:
            with c.loop(rows, name="i") as i:
                self.flash_scores(i, keys, cb_q, cb_k)
                self.flash_rest(keys, cb_v)
            return
        # Row i + 1's scores before row i's rest: one row ahead (m read at the
        # front: the score scope pops the old m it used).
        self.flash_scores(0, keys, cb_q, cb_k)
        with c.loop(rows - 1, name="i") as i:
            self.flash_scores(i + 1, keys, cb_q, cb_k)
            self.flash_rest(keys, cb_v)
        self.flash_rest(keys, cb_v)

    def flash_scores(self, i, keys, cb_q, cb_k):
        """Query row i's scope: m (the front of its queue), the scores, their row
        max with m into m_new, and p; p, m_new pushed (in scope: alpha too, and the
        old m popped)."""
        d, c, units, f = self.d, self.c, self.units, self.d.flash
        KB = d.keys
        cb_p, m_q = self.cbs["prob"], self.queue[f["m"]]
        t_m = self.temps[f["m_new"]]
        zone = c.zone if ZONES else (lambda name: contextlib.nullcontext())
        c.wait(m_q, 1)
        with zone("S"):
            c.acquire()
            units.forget()
            units.unpack_as("copy", m_q)
            c.copy(m_q, 0, KB)
            inscope = f["inscope"]
            if inscope:
                c.copy(m_q, 0, KB + 1)
            self.raw_scores(i, keys, 0, KB, cb_q, cb_k)
            self.guarded(0, KB, keys, lambda s: self.run(d.score, [s], s), MASKED)
            units.setup("mul")
            # m_new (in slot mn): max(m, row max), then its chain on m and that.
            mn = KB + 1 if inscope else KB
            c.row_reduce("max", 0, KB, into=mn)
            units.sfpu = None
            if inscope:
                place = {id(f["m"]): KB, id(f["max"]): mn, id(f["m_new"]): mn}
                for value, out in ((f["m_new"], mn), (f["alpha"], KB)):
                    slots = [place[id(x.source)] for x in value.chain.operands]
                    self.run(value.chain, slots, out)
                units.sfpu = None
            score = next(
                n for n, x in enumerate(d.prob.operands) if x.source == "score"
            )
            values = tuple(self.v.scalar(s) for s in d.prob.values)
            if PACK_EXP:
                # p on the pack thread, beside the math thread's next scope.
                c.commit()
            else:
                units.setup(d.prob.fn.init)
            for s in range(KB):
                slots = [s, mn] if score == 0 else [mn, s]
                thread = "pack" if PACK_EXP else "math"
                c.run(d.prob.fn, *slots, out=s, values=values, thread=thread)
            units.forget()
            t_a = self.temps[f["alpha"]]
            c.reserve(cb_p, KB)
            c.reserve(m_q, 1)
            c.reserve(t_m, 1)
            if inscope:
                c.reserve(t_a, 1)
            units.pack_into(cb_p)
            for s in range(KB):
                c.pack(s, cb_p, at=s)
            units.pack_into(m_q)
            c.pack(mn, m_q)
            c.pack(mn, t_m)
            if inscope:
                c.pack(KB, t_a)
            c.release()
            c.push(cb_p, KB)
            c.push(m_q, 1)
            c.push(t_m, 1)
            if inscope:
                c.push(t_a, 1)
                c.pop(m_q, 1)

    def flash_rest(self, keys, cb_v):
        """The front row's alpha, accumulator and den; its old carries and temps
        popped."""
        d, c, f = self.d, self.c, self.d.flash
        KB = d.keys
        cb_p, m_q, den_q = self.cbs["prob"], self.queue[f["m"]], self.queue[f["den"]]
        t_m, t_a = self.temps[f["m_new"]], self.temps[f["alpha"]]
        zone = c.zone if ZONES else (lambda name: contextlib.nullcontext())
        c.wait(t_m, 1)
        c.wait(cb_p, KB)
        with zone("A"):
            if f["inscope"]:
                c.wait(t_a, 1)
            else:
                self.value(f["alpha"])
            if LAZY:
                self.flag = self.moved(t_a, self.cb_flag)
        with zone("V"):
            self.accumulate(keys, cb_p, cb_v)
            self.den_update(keys, cb_p, den_q)
        if LAZY:
            self.flag = None
            c.pop(self.cb_flag, 1)
        c.pop(cb_p, KB)
        if not f["inscope"]:
            c.pop(m_q, 1)
        c.pop(den_q, 1)
        c.pop(t_m, 1)
        c.pop(t_a, 1)

    def den_update(self, keys, cb_p, den_q):
        """den * alpha (the accumulator's rescale) + p @ ones (the row sum of p on the
        matrix unit), pushed behind the old den."""
        d, c, units = self.d, self.c, self.units
        rescale = d.rescale
        own = next(n for n, x in enumerate(rescale.operands) if x.source is d.acc)
        others = [n for n in range(len(rescale.operands)) if n != own]
        place = {n: 1 + k for k, n in enumerate(others)}
        c.wait(den_q, 1)
        c.acquire()
        units.forget()
        units.unpack_as("copy", den_q)
        c.copy(den_q, 0, 0)
        self.unpack(_Some(rescale, others), [place[n] for n in others], {})
        slots = [0 if n == own else place[n] for n in range(len(others) + 1)]
        self.rescaled(lambda: self.run(rescale, slots, 0))
        units.forget()
        ones = self.ones[BF16]
        self.matmul_format(cb_p, ones)
        product = c.matmul_init(cb_p, ones, transpose=0, ct=1, rt=1, kt=d.keys)
        with c.loop(keys, name="c") as cc:
            c.matmul(product, cc.bounded(0, d.keys), 0, 0)
        units.forget()
        c.reserve(den_q, 1)
        units.pack_into(den_q)
        c.pack(0, den_q)
        c.release()
        c.push(den_q, 1)

    def raw_scores(self, i, keys, g0, width, cb_q, cb_k):
        """Slots 0 .. width - 1: the mask tiles, then q @ k.T of tiles g0 .. added."""
        d, c, units, dt = self.d, self.c, self.units, self.dt
        if d.mask is not None:
            for s in range(width):
                with c.when(g0 + s < keys):
                    index = c.let(f"mask{g0 + s}", self.mask_index(i, g0 + s))
                    with c.when(index.ne(0)):
                        units.forget()
                        units.unpack_as("copy", self.cb_mask)
                        c.copy(self.cb_mask, index.bounded(0, MASK_TILES), s)
            units.forget()
        self.matmul_format(cb_q, cb_k)
        if BLOCKS:
            product = c.matmul_init(cb_q, cb_k, transpose=1, ct=width, rt=1, kt=dt)
            with c.loop(dt, name="t") as t:
                ia = (i * dt + t).bounded(0, d.rows * dt)
                ib = (t * d.keys + g0).bounded(0, d.keys * dt - width + 1)
                c.matmul(product, ia, ib, 0)
        else:
            product = c.matmul_init(cb_q, cb_k, transpose=1, ct=1, rt=1, kt=dt)
            with c.loop(dt, name="t") as t:
                ia = (i * dt + t).bounded(0, d.rows * dt)
                for s in range(width):
                    ib = (t * d.keys + g0 + s).bounded(0, d.keys * dt)
                    c.matmul(product, ia, ib, s)
        units.forget()

    def values(self, values):
        """Column values in order: a reduction alone, consecutive chains in one
        register scope where their operands and results fit the slots."""
        run = []

        def flush():
            if len(run) > 1:
                self.chains(run)
            elif run:
                self.value(run[0])
            run.clear()

        for value in values:
            if value.kind != "chain":
                flush()
                self.value(value)
                continue
            if run and _chain_slots([*run, value]) > SLOTS:
                flush()
            run.append(value)
        flush()

    def chains(self, values):
        """Chain values in one register scope: each operand unpacked once (an earlier
        value's result read from its slot), each result packed into its page and its
        carry's queue."""
        c, units, d = self.c, self.units, self.d
        for cv, kind in d.carries.items():
            if kind == "col":
                c.wait(self.queue[cv], 1)
        c.acquire()
        units.forget()
        held, out_slot, nxt = {}, {}, 0
        for value in values:
            slots = []
            for leaf in value.chain.operands:
                src = leaf.source
                if isinstance(src, Value) and src in out_slot:
                    slots.append(out_slot[src])
                    continue
                key = (id(src), leaf.kind)
                if key not in held:
                    held[key] = nxt
                    self.unpack(_Some(value.chain, [len(slots)]), [nxt], {})
                    nxt += 1
                slots.append(held[key])
            out_slot[value] = nxt
            self.run(value.chain, slots, nxt)
            nxt += 1
        targets = []
        for value in values:
            targets.append((self.temps[value], out_slot[value]))
            for cv, kind in d.carries.items():
                if kind == "col" and _next(d.ir, cv) is value.node:
                    targets.append((self.queue[cv], out_slot[value]))
        for cb, _ in targets:
            c.reserve(cb, 1)
        units.pack_into(targets[0][0])
        for cb, slot in targets:
            c.pack(slot, cb)
        c.release()
        for cb, _ in targets:
            c.push(cb, 1)
        for value in values:
            c.wait(self.temps[value], 1)

    def guarded(self, g0, width, keys, body, fill):
        """`body(s)` on slots s of tiles g0 + s inside the block's `keys` key tiles,
        `fill` on the others; one branch when the whole group is inside."""
        c, units = self.c, self.units
        units.forget()
        with c.when((g0 + width - 1) < keys):
            for s in range(width):
                body(s)
        with c.otherwise():
            for s in range(width):
                with c.when(g0 + s < keys):
                    units.forget()
                    body(s)
                with c.otherwise():
                    units.forget()
                    units.setup("fill")
                    c.op("fill", s, value=fill)
        units.forget()

    def scores(self, i, keys, g0, width, cb_q, cb_k):
        """Score tiles g0 .. g0 + width - 1 of query row i in one register scope: mask
        tiles copied in, q @ k.T accumulated onto them, the score chain; packed fp32
        and bf16. Tiles past the block's keys hold MASKED."""
        d, c, units = self.d, self.c, self.units
        cb_s, cb_sb = self.cbs["score"], self.cbs["max"]
        c.acquire()
        units.forget()
        self.raw_scores(i, keys, g0, width, cb_q, cb_k)
        self.guarded(g0, width, keys, lambda s: self.run(d.score, [s], s), MASKED)
        units.pack_into(cb_s)
        for s in range(width):
            c.pack(s, cb_s, at=g0 + s)
        units.pack_into(cb_sb)
        for s in range(width):
            c.pack(s, cb_sb, at=g0 + s)
        c.release()

    def probs(self, g0, width):
        """Probability tiles g0 .. g0 + width - 1 in one register scope: the column
        operands unpacked once into the slots after the tiles."""
        d, c, units = self.d, self.c, self.units
        prob, cb_p = d.prob, self.cbs["prob"]
        others = [n for n, x in enumerate(prob.operands) if x.source != "score"]
        place = {n: width + k for k, n in enumerate(others)}
        c.acquire()
        units.forget()
        cols = _Some(prob, others)
        self.unpack(cols, [place[n] for n in others], {})
        score = next(n for n, x in enumerate(prob.operands) if x.source == "score")
        for s in range(width):
            units.unpack_as("copy", self.cbs["score"])
            c.copy(self.cbs["score"], g0 + s, s)
        for s in range(width):
            slots = [s if n == score else place[n] for n in range(len(prob.operands))]
            self.run(prob, slots, s)
        units.pack_into(cb_p)
        for s in range(width):
            c.pack(s, cb_p, at=g0 + s)
        c.release()

    def rescaled(self, run):
        """`run()` (a rescale), only where the step's flag says a row's max moved
        when there is one (no rescale by an alpha of exactly 1)."""
        flag = getattr(self, "flag", None)
        if flag is None:
            run()
            return
        with self.c.when(flag.ne(0)):
            run()
        self.units.forget()

    def moved(self, t_a, cb_flag):
        """The step's flag: 1 when some row's alpha is not exactly 1 (its max moved),
        read by every compute thread."""
        c, units = self.c, self.units
        c.acquire()
        units.forget()
        units.unpack_as("copy", t_a)
        c.copy(t_a, 0, 0)
        units.setup(_MOVED.init)
        c.run(_MOVED, 0, out=0)
        units.setup("transpose_dest")
        c.op("transpose_dest", 0)
        units.setup("mul")
        c.row_reduce("max", 0, 1, into=0, fresh=True)
        units.sfpu = None
        c.reserve(cb_flag, 1)
        units.pack_into(cb_flag)
        c.pack(0, cb_flag)
        c.release()
        c.push(cb_flag, 1)
        c.wait(cb_flag, 1)
        units.forget()
        return c.read_word(cb_flag, 0)

    def accumulate(self, keys, cb_p, cb_v):
        """The accumulator's tiles rescaled by their chain and p @ v added in the
        registers, a group of tiles per scope; the new tiles pushed behind the old."""
        d, c, units, dt = self.d, self.c, self.units, self.dt
        acc, rescale = self.queue[d.acc], d.rescale
        own = next(n for n, x in enumerate(rescale.operands) if x.source is d.acc)
        others = [n for n in range(len(rescale.operands)) if n != own]
        width = self.width
        place = {n: width + k for k, n in enumerate(others)}
        c.wait(acc, self.dtp)
        with c.block_loop(dt, width, names=("t0", "nt")) as (t0, nt):
            c.reserve(acc, width)
            c.acquire()
            units.forget()
            self.unpack(_Some(rescale, others), [place[n] for n in others], {})

            def body(s, t0=t0):
                units.unpack_as("copy", acc)
                c.copy(acc, (t0 + s).bounded(0, self.dtp), s)
                count = len(others) + 1
                slots = [s if n == own else place[n] for n in range(count)]
                self.rescaled(lambda: self.run(rescale, slots, s))

            self.guarded(0, width, nt, body, 0.0)
            self.matmul_format(cb_p, cb_v)
            # A whole group: one block call per key tile; the row's last, tile by tile.
            blocks = c.when(nt.eq(width)) if BLOCKS else contextlib.nullcontext()
            with blocks:
                if BLOCKS:
                    product = c.matmul_init(
                        cb_p, cb_v, transpose=0, ct=width, rt=1, kt=d.keys
                    )
                    with c.loop(keys, name="c") as cc:
                        ib = (cc * dt + t0).bounded(0, d.keys * dt - width + 1)
                        c.matmul(product, cc.bounded(0, d.keys), ib, 0)
            with c.otherwise() if BLOCKS else contextlib.nullcontext():
                product = c.matmul_init(cb_p, cb_v, transpose=0, ct=1, rt=1, kt=d.keys)
                with c.loop(keys, name="c") as cc:
                    for s in range(width):
                        with c.when(s < nt):
                            ib = (cc * dt + t0 + s).bounded(0, d.keys * dt)
                            c.matmul(product, cc.bounded(0, d.keys), ib, s)
            units.forget()
            units.pack_into(acc)
            for s in range(width):
                c.pack(s, acc, at=s)
            c.release()
            c.push(acc, width)
        self.pop_tiles(acc)

    def value(self, value):
        """Column value `value` of the row into its page (and into its carry's queue
        when it is a carry's next value)."""
        c, units, d = self.c, self.units, self.d
        cb = self.temps[value]
        targets = [cb]
        for cv, kind in d.carries.items():
            if kind == "col" and _next(d.ir, cv) is value.node:
                targets.append(self.queue[cv])
        if value.kind == "chain":
            for cv, kind in d.carries.items():
                if kind == "col":
                    c.wait(self.queue[cv], 1)
        c.acquire()
        units.forget()
        if value.kind in ("max", "sum"):
            src = self.cbs[value.kind if value.kind == "max" else "prob"]
            ones = self.ones[src.dtype]
            # A row sum takes the ones through the first source, a max the tiles.
            first, second = (ones, src) if value.kind == "sum" else (src, ones)
            c.format(srca=first, srcb=second)
            units.srca, units.srcb = first.dtype, second.dtype
            r = c.reduce_init(value.kind, "row", src, ones, cb)
            c.reduce(r, 0, 0, 0)
            with c.loop(d.keys - 1, name="c") as cc:
                c.reduce(r, cc + 1, 0, 0)
        else:
            slots = list(range(len(value.chain.operands)))
            self.unpack(value.chain, slots, {})
            self.run(value.chain, slots, 0)
        # A reduction packs under its masks into its own page only.
        now = targets[:1] if value.kind in ("max", "sum") else targets
        for target in now:
            c.reserve(target, 1)
        units.pack_into(cb)
        for target in now:
            c.pack(0, target)
        c.release()
        if value.kind in ("max", "sum"):
            c.reduce_done()
            units.forget()
            units.srca = units.srcb = units.pack = None
        for target in now:
            c.push(target, 1)
        c.wait(cb, 1)
        for target in targets[len(now) :]:
            c.reserve(target, 1)
            c.acquire()
            units.forget()
            units.unpack_as("copy", cb)
            c.copy(cb, 0, 0)
            units.pack_into(target)
            c.pack(0, target)
            c.release()
            c.push(target, 1)

    def finish(self, cb_out):
        """The stored chain of one query row from the final carries."""
        c, units, d = self.c, self.units, self.d
        dt = self.dt
        for cv, kind in d.carries.items():
            c.wait(self.queue[cv], self.dtp if kind == "tile" else 1)
        store = d.store
        slots = list(range(len(store.operands)))
        with c.loop(dt, name="t") as t:
            c.reserve(cb_out, 1)
            c.acquire()
            units.forget()
            self.unpack(store, slots, {"tile": t.bounded(0, self.dtp)})
            self.run(store, slots, 0)
            units.pack_into(cb_out)
            c.pack(0, cb_out)
            c.release()
            c.push(cb_out, 1)
        for cv, kind in d.carries.items():
            if kind == "tile":
                self.pop_tiles(self.queue[cv])
            else:
                c.pop(self.queue[cv], 1)


@dataclass
class AttentionPlan:
    """One call of an attention kernel: units, cores, the score tiles and bytes."""

    design: AttentionDesign
    units: int
    cores: int
    score_tiles: int
    bytes_read: int
    bytes_written: int
    l1: int
    program: object = None

    def bounds(self):
        """{name: seconds} of lower bounds: DRAM, the matrix unit, the vector unit."""
        from kohakutt.dsl.tile.plan import CLOCK, DRAM_RATE

        dt = self.design_dt
        per_core = self.score_tiles / max(1, self.cores)
        return {
            "DRAM (512 GB/s)": (self.bytes_read + self.bytes_written) / DRAM_RATE,
            # Two products of dt tile matmuls per score tile, about 64 cycles each.
            "matrix unit (estimate)": per_core * 2 * dt * 64 / CLOCK,
            # exp on a score tile: about 300 cycles (the backend's flash kernel).
            "vector unit (estimate)": per_core * 300 / CLOCK,
        }

    def report(self):
        """The plan report."""
        d = self.design
        bounds = self.bounds()
        worst = max(bounds, key=bounds.get)
        lines = [
            f"kernel    {d.ir.name} (attention form, fp32 registers)",
            f"grid      {self.units} blocks of up to {d.rows} query tile rows over "
            f"{self.cores} cores; key blocks of up to {d.keys} tiles",
            *d.describe(),
            f"L1        {self.l1 // 1024} KB of buffers per core",
            f"moved     {(self.bytes_read + self.bytes_written) / 1e6:.2f} MB "
            f"({self.bytes_read / 1e6:.2f} read: k and v once per block of query "
            "rows)",
            "bound     "
            + ", ".join(f"{k} {v * 1e6:.1f} us" for k, v in bounds.items()),
            f"binds     {worst}",
        ]
        return "\n".join(lines)


def attention_plan(d, program, runtime, env):
    """The AttentionPlan of one call."""
    import math

    from kohakutt.dsl.tile.ir import _shape_tiles
    from kohakutt.dsl.tile.plan import TILE_BYTES

    units = d.ir.grid.count.eval(env)
    cores = program.core_count(runtime)
    q, k = d.q.attrs["view"].tensor, d.k.attrs["view"].tensor
    qrows, dt = _shape_tiles(env[q.name])
    krows, _ = _shape_tiles(env[k.name])
    heads = math.prod(env[q.name][:-2]) or 1
    kheads = math.prod(env[k.name][:-2]) or 1
    per_head = krows // kheads
    score = qrows * per_head
    if d.mask is not None and d.mask.attrs["causal"]:
        rows = qrows // heads
        score = heads * rows * (rows + 1) // 2
    steps = units * per_head
    read = qrows * dt * TILE_BYTES[q.dtype] + 2 * steps * dt * TILE_BYTES[k.dtype]
    written = qrows * dt * TILE_BYTES[q.dtype]
    l1 = program.footprint(program.sizes(runtime))
    plan = AttentionPlan(d, units, cores, score, read, written, l1, program)
    plan.design_dt = dt
    return plan


_MOVED = km.sfpu_fn(
    "tile_attention_moved", km.sfpu.select(km.sfpu.operand(0) != 1.0, 1.0, 0.0)
)


def _chain_slots(values):
    """Register slots one scope of chain values `values` takes: each distinct operand
    once (an earlier value's result is read from its slot) and each result."""
    seen, made = set(), set()
    for value in values:
        for leaf in value.chain.operands:
            if not (isinstance(leaf.source, Value) and leaf.source in made):
                seen.add((id(leaf.source), leaf.kind))
        made.add(value)
    return len(seen) + len(made)


def _next(ir, carry):
    return next(value for c, value in ir.carries if c is carry)


def lower_attention(design, machine):
    """(km.Program, Values) of attention design `design`."""
    low = _Lower(design, machine)
    return low.build(), low.v
