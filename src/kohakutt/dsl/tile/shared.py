"""Matmul kernels with w shared (`share=g.col`), in the layout and plan of the
backend's block matmul (`dsl_metal/gemm_km.py`): each core owns output row tiles and
reads its own a, the top core of each core column multicasts w down it (a lane), and
the kernel's vector chain runs on the registers before the last pack."""

import functools
import math

import kohakutt.dsl.metal as km
from kohakutt.dsl.metal.binding import as_expr
from kohakutt.dsl.metal.passes import _partials
from kohakutt.dsl.tile.ir import F32, Int, fail
from kohakutt.dsl.tile.plan import Leaf

# Output subblocks (rows, columns) with fp32 registers, fewest unpacked tiles first.
SUBBLOCKS = ((1, 4), (4, 1), (2, 2), (1, 3), (3, 1), (1, 2), (2, 1), (1, 1))
# Most tiles of an a block, a w block and an output block.
A_MOST, W_MOST, OUT_MOST = 512, 512, 256
# Bytes of a bf16 tile and of an fp32 partials tile; L1 bytes the buffers may take.
TILE, PART_TILE = 2048, 4096
BUDGET = 1024 * 1024
NAMES = ("pcm", "bm", "bn", "nbh", "nbw", "kb", "nk", "kb_last", "sh", "sw")


def subblocks(extra):
    """The subblocks that leave `extra` register slots for the chain's operands."""
    return tuple(s for s in SUBBLOCKS if s[0] * s[1] <= 4 - extra)


def _l1(bm, bn, kb, sub, rows):
    part = bm * bn * PART_TILE
    out = 2 * bm * kb * TILE + 2 * kb * bn * TILE + 2 * sub * TILE + part
    return out + rows * bn * TILE


@functools.cache
def plan(mt, kt, nt, cores, extra, rows):
    """{name: value} of the block plan for [mt, kt] @ [kt, nt] tiles on `cores`
    cores (the backend's one-row-per-core search), or None when nothing fits."""
    pcm = math.ceil(mt / cores)
    subs = subblocks(extra)

    def values(bm, bn, nbh, nbw, kb, sh, sw):
        kb = min(kb, kt)
        nk = math.ceil(kt / kb)
        if _l1(bm, bn, kb, sh * sw, rows) > BUDGET:
            return None
        if bm * kb > A_MOST or kb * bn > W_MOST or bm * bn > OUT_MOST:
            return None
        found = (pcm, bm, bn, nbh, nbw, kb, nk, kt - (nk - 1) * kb, sh, sw)
        return dict(zip(NAMES, found))

    splits = []
    for nbh in range(1, pcm + 1):
        bm = math.ceil(pcm / nbh)
        if nbh > 1 and (nbh - 1) * bm >= pcm:
            continue
        for nbw in range(1, nt + 1):
            bn = math.ceil(nt / nbw)
            if (nbw > 1 and (nbw - 1) * bn >= nt) or (nt >= 8 and nbw < 2):
                continue
            splits.append((nbh * nbw, nbw * mt + nbh * nt, nbh, nbw))
    for _count, _cost, nbh, nbw in sorted(splits):
        bm, bn = math.ceil(pcm / nbh), math.ceil(nt / nbw)
        fit = [s for s in subs if bm % s[0] == 0 and bn % s[1] == 0]
        if not fit:
            continue
        for kb in (16, 8, 4, 2, 1):
            out = values(bm, bn, nbh, nbw, kb, *fit[0])
            if out is not None:
                return out
    return None


def check_shared(d):
    """Refusals of the shared forms for matmul design `d`."""
    if d.prologue is not None:
        fail("a shared w with a computed left operand (row.tile(k)) is not built")
    a_share = d.a.node.attrs.get("share")
    if a_share not in (None, "row"):
        fail("a is shared by the units of a grid row: share=g.row", d.a.node.line)
    if d.w.node.attrs["share"] != "col":
        fail("w is shared by the units of a grid column: share=g.col", d.w.node.line)


@functools.cache
def plan2d(mt, kt, nt, cols, rows, extra, row_ops, budget):
    """{name: value} of the 2-D plan for [mt, kt] @ [kt, nt] tiles on a `cols` x
    `rows` grid: pr x pc output tiles per core, K blocks of kb, subblock sh x sw."""
    pr, pc = math.ceil(mt / rows), math.ceil(nt / cols)
    fit = [s for s in subblocks(extra) if pr % s[0] == 0 and pc % s[1] == 0]
    if not fit or pr * pc > OUT_MOST:
        return None
    sh, sw = fit[0]
    for kb in (16, 8, 4, 2, 1):
        kb = min(kb, kt)
        if pr * kb > A_MOST or kb * pc > W_MOST:
            continue
        if _l1(pr, pc, kb, sh * sw, row_ops) <= budget:
            nk = math.ceil(kt / kb)
            names = ("pr", "pc", "kb", "nk", "kb_last", "sh", "sw")
            return dict(zip(names, (pr, pc, kb, nk, kt - (nk - 1) * kb, sh, sw)))
    return None


def lower_grid2d(d, machine):
    """(km.Program, Values) of matmul design `d` with a shared along core rows and
    w down core columns: core (x, y) computes output tiles of rows y * pr .. and
    columns x * pc ..; the left column reads a, the top row reads w."""
    from kohakutt.dsl.tile.lower import Values

    check_shared(d)
    if machine.rows < 2 or machine.cols < 2:
        fail("two shared operands need a grid of two core rows and two columns")
    ir = d.ir
    p = km.Program(ir.name, machine)
    v = Values(p)
    a = p.input(d.a.tensor.name, d.a.tensor.dtype)
    w = p.input(d.w.tensor.name, d.w.tensor.dtype)
    rows_in = [p.input(op.tensor.name, op.tensor.dtype) for op in d.rows]
    store, _ = d.stores[0]
    out = p.output(store.view.tensor.name, store.view.tensor.dtype)
    kt = v.host_value(ir.loop.count)
    nt = v.host_value(Int("size", value=("row_tiles", store.view.tensor.name, None)))
    mt = v.host_value(ir.grid.count)
    extra = len(d.chain.operands) - 1
    sh, sw = p.choose(("sh", "sw"), subblocks(extra))
    pr, pc = p.runtime("pr", multiple_of="sh"), p.runtime("pc", multiple_of="sw")
    kb, nk, kb_last = p.runtime("kb"), p.runtime("nk"), p.runtime("kb_last")
    gx, gy = machine.cols, machine.rows

    def pick(name):
        def fn(env, cores):
            found = plan2d(
                env[mt.name], env[kt.name], env[nt.name], gx, gy, extra,
                len(rows_in), km.L1_BUDGET,
            )  # fmt: skip
            if found is None:
                fail(f"{ir.name}: no 2-D block plan fits L1")
            return found[name]

        return fn

    v.extra = {n: pick(n) for n in ("pr", "pc", "kb", "nk", "kb_last", "sh", "sw")}
    p.limit(nk, low=1)
    p.limit(kb_last, low=1, high=kb)
    a_block = p.sized("a_block", of=pr * kb, exact=True, limit=A_MOST)
    w_block = p.sized("w_block", of=kb * pc, exact=True, limit=W_MOST)
    block = p.sized("block", of=pr * pc, exact=True, limit=OUT_MOST)
    sub = p.sized("sub", of=sh * sw, exact=True, limit=4)
    cb_a = p.cb(0, like=a, pages=2 * a_block)
    cb_w = p.cb(1, like=w, pages=2 * w_block)
    cb_out = p.cb(16, like=out, pages=2 * sub)
    part = p.partials(24, pages=block, name="part")
    row_block = (
        p.sized("row_block", of=pc, exact=True, limit=OUT_MOST) if rows_in else 0
    )
    cb_rows = [p.cb(like=r, pages=row_block) for r in rows_in]
    x = p.define("x", p.core % gx)
    y = p.define("y", p.core // gx)
    r0, c0 = p.define("r0", y * pr), p.define("c0", x * pc)
    count = p.define("rows", km.select(r0 < mt, km.min(pr, mt - r0), 0))
    cols = p.define("cols", km.select(c0 < nt, km.min(pc, nt - c0), 0))

    w_lane = p.lane(p.cores.rows(0), p.cores.rows(1, gy - 1), name="w")
    a_lane = p.lane(p.cores.cols(0), p.cores.cols(1, gx - 1), name="a")

    def w_page(k0, k, c):
        return w.page((k0 + k) * nt + c0 + c)

    def a_page(k0, r, i):
        return a.page((r0 + r) * kt + k0 + i)

    def send(k, lane, cb, size, outer, inner, page, at):
        lane.open(k)

        def one(kblk, first, last):
            k0, kc = kblk * kb, kb_last if last else kb
            slot = k.reserve(cb, size)
            n_out, n_in = outer(kc), inner(kc)
            with k.loop(n_out, name="o") as o, k.loop(n_in, name="q") as q:
                k.read(page(k0, o, q), slot[at(o, q).bounded(0, size)])
            k.reads_done()
            lane.send(k, slot)
            k.push(cb, size)

        km.k_passes(k, nk, one)

    def receive(k, lane, cb, size):
        def one(kblk, first, last):
            k.reserve(cb, size)
            lane.receive(k)
            k.push(cb, size)

        km.k_passes(k, nk, one)
        k.atomics_done()

    w_send = p.kernel("dm0", cores=p.cores.rows(0), noc=0, name="w_send")
    send(
        w_send, w_lane, cb_w, w_block, lambda kc: kc, lambda kc: cols,
        w_page, lambda kk, c: kk * pc + c,
    )  # fmt: skip
    w_send.writes_done()
    receive(
        p.kernel("dm0", cores=p.cores.rows(1, gy - 1), noc=0, name="w_recv"),
        w_lane, cb_w, w_block,
    )  # fmt: skip

    def finish(k):
        """Bias block reads, then the output subblocks written as they come."""
        with k.loop(pr, sh, name="i") as i, k.loop(pc, sw, name="j") as j:
            k.wait(cb_out, sub)
            with k.loop(sh, name="r") as r, k.loop(sw, name="c") as c:
                row, col = r0 + i + r, c0 + j + c
                with k.when(row < mt), k.when(col < nt):
                    k.write(
                        cb_out, out.page(row * nt + col), (r * sw + c).bounded(0, sub)
                    )
            k.flushed()
            k.pop(cb_out, sub)
        k.writes_done()

    def bias(k):
        for ref, cb in zip(rows_in, cb_rows, strict=True):
            slot = k.reserve(cb, row_block)
            with k.loop(cols, name="c") as c:
                k.read(ref.page(c0 + c), slot[c.bounded(0, row_block)])
        if rows_in:
            k.reads_done()
        for cb in cb_rows:
            k.push(cb, row_block)

    a_send = p.kernel("dm1", cores=p.cores.cols(0), noc=1, name="a_send")
    bias(a_send)
    send(
        a_send, a_lane, cb_a, a_block, lambda kc: count, lambda kc: kc,
        a_page, lambda r, i: r * kb + i,
    )  # fmt: skip
    finish(a_send)
    a_recv = p.kernel("dm1", cores=p.cores.cols(1, gx - 1), noc=1, name="a_recv")
    bias(a_recv)
    receive(a_recv, a_lane, cb_a, a_block)
    finish(a_recv)

    fp32 = [cb for cb in (cb_a, cb_w, *cb_rows, cb_out) if cb.dtype == F32]
    c = p.kernel("compute", fp32_dest=True, approx=False, unpack_fp32=[part, *fp32])
    c.startup(cb_a, cb_w, cb_out, reverse=True)
    shape = {
        "bm": pr, "bn": pc, "sh": sh, "sw": sw, "kb": kb, "nk": nk,
        "kb_last": kb_last, "a_block": a_block, "w_block": w_block, "sub": sub,
    }  # fmt: skip
    chain = d.chain
    identity = len(chain.operands) == 1 and not chain.ops.keys() - {"acc", "cast"}
    epilogue = None
    if not identity:
        row_cbs = dict(zip(map(id, d.rows), cb_rows, strict=True))

        def epilogue(j):
            _epilogue(c, chain, sub, sw, j, 4 - extra, row_cbs, row_block, v)

    for cb in cb_rows:
        c.wait(cb, row_block)
    _passes(c, shape, cb_a, cb_w, part, cb_out, epilogue)
    for cb in cb_rows:
        c.pop(cb, row_block)
    return p, v


def lower_shared(d, machine):
    """(km.Program, Values) of matmul design `d` with w shared down core columns."""
    from kohakutt.dsl.tile.lower import Values

    check_shared(d)
    if machine.rows < 2:
        fail("a shared w needs a grid of two core rows or more")
    ir = d.ir
    p = km.Program(ir.name, machine)
    v = Values(p)
    a = p.input(d.a.tensor.name, d.a.tensor.dtype)
    w = p.input(d.w.tensor.name, d.w.tensor.dtype)
    rows = [p.input(op.tensor.name, op.tensor.dtype) for op in d.rows]
    store, _ = d.stores[0]
    out = p.output(store.view.tensor.name, store.view.tensor.dtype)
    kt = v.host_value(ir.loop.count)
    nt = v.host_value(Int("size", value=("row_tiles", store.view.tensor.name, None)))
    mt = v.host_value(ir.grid.count)
    extra = len(d.chain.operands) - 1
    subs = subblocks(extra)
    sh, sw = p.choose(("sh", "sw"), subs)
    pcm = p.runtime("pcm")
    bm, bn = p.runtime("bm", multiple_of="sh"), p.runtime("bn", multiple_of="sw")
    nbh, nbw = p.runtime("nbh"), p.runtime("nbw")
    kb, nk, kb_last = p.runtime("kb"), p.runtime("nk"), p.runtime("kb_last")

    def pick(name):
        def fn(env, cores):
            found = plan(
                env[mt.name], env[kt.name], env[nt.name], cores, extra, len(rows)
            )
            if found is None:
                fail(f"{ir.name}: no block plan of the shared form fits L1")
            return found[name]

        return fn

    v.extra = {name: pick(name) for name in NAMES}
    p.limit(nk, low=1)
    p.limit(kb_last, low=1, high=kb)
    a_block = p.sized("a_block", of=bm * kb, exact=True, limit=A_MOST)
    w_block = p.sized("w_block", of=kb * bn, exact=True, limit=W_MOST)
    block = p.sized("block", of=bm * bn, exact=True, limit=OUT_MOST)
    sub = p.sized("sub", of=sh * sw, exact=True, limit=4)
    cb_a = p.cb(0, like=a, pages=2 * a_block)
    cb_w = p.cb(1, like=w, pages=2 * w_block)
    cb_out = p.cb(16, like=out, pages=2 * sub)
    part = p.partials(24, pages=block, name="part")
    row_block = p.sized("row_block", of=bn, exact=True, limit=OUT_MOST) if rows else 0
    cb_rows = [p.cb(like=r, pages=row_block) for r in rows]
    blocks = nbh * nbw
    m0 = p.define("m0", p.core * pcm)
    m_end = p.define("m_end", km.min(m0 + pcm, mt))

    top, below = p.cores.rows(0), p.cores.rows(1, machine.rows - 1)
    lane = p.lane(top, below, name="w")

    def w_columns(k, blk):
        c0 = k.let("c0", blk % nbw * bn)
        return c0, k.let("cols", km.select(c0 < nt, km.min(bn, nt - c0), 0))

    send = p.kernel("dm0", cores=top, noc=0, name="w_send")
    lane.open(send)
    with send.loop(blocks, name="blk") as blk:
        c0, cols = w_columns(send, blk)

        def w_pass(kblk, first, last):
            k0, kc = kblk * kb, kb_last if last else kb
            slot = send.reserve(cb_w, w_block)
            with send.loop(kc, name="k") as k, send.loop(cols, name="c") as c:
                at = (k * bn + c).bounded(0, w_block)
                send.read(w.page((k0 + k) * nt + c0 + c), slot[at])
            send.reads_done()
            lane.send(send, slot)
            send.push(cb_w, w_block)

        km.k_passes(send, nk, w_pass)
    send.writes_done()

    recv = p.kernel("dm0", cores=below, noc=0, name="w_recv")
    with recv.loop(blocks, name="blk"):

        def w_receive(kblk, first, last):
            recv.reserve(cb_w, w_block)
            lane.receive(recv)
            recv.push(cb_w, w_block)

        km.k_passes(recv, nk, w_receive)
    recv.atomics_done()

    k = p.kernel("dm1", noc=1)
    with k.loop(blocks, name="blk") as blk:
        r0 = k.let("r0", m0 + blk // nbw * bm)
        c0 = k.let("c0", blk % nbw * bn)
        count = k.let("rows", km.select(r0 < mt, km.min(bm, mt - r0), 0))
        if rows:
            cols = k.let("cols", km.select(c0 < nt, km.min(bn, nt - c0), 0))
            for ref, cb in zip(rows, cb_rows, strict=True):
                slot = k.reserve(cb, row_block)
                with k.loop(cols, name="c") as c:
                    k.read(ref.page(c0 + c), slot[c.bounded(0, row_block)])
            k.reads_done()
            for cb in cb_rows:
                k.push(cb, row_block)

        def a_pass(kblk, first, last):
            k0, kc = kblk * kb, kb_last if last else kb
            slot = k.reserve(cb_a, a_block)
            with k.loop(count, name="r") as r, k.loop(kc, name="k") as i:
                at = (r * kb + i).bounded(0, a_block)
                k.read(a.page((r0 + r) * kt + k0 + i), slot[at])
            k.reads_done()
            k.push(cb_a, a_block)

        km.k_passes(k, nk, a_pass)
        with k.loop(bm, sh, name="i") as i, k.loop(bn, sw, name="j") as j:
            k.wait(cb_out, sub)
            with k.loop(sh, name="r") as r, k.loop(sw, name="c") as c:
                row, col = r0 + i + r, c0 + j + c
                with k.when(row < m_end), k.when(col < nt):
                    k.write(
                        cb_out, out.page(row * nt + col), (r * sw + c).bounded(0, sub)
                    )
            k.flushed()
            k.pop(cb_out, sub)
    k.writes_done()

    fp32 = [cb for cb in (cb_a, cb_w, *cb_rows, cb_out) if cb.dtype == F32]
    c = p.kernel("compute", fp32_dest=True, approx=False, unpack_fp32=[part, *fp32])
    c.startup(cb_a, cb_w, cb_out, reverse=True)
    shape = {
        "bm": bm,
        "bn": bn,
        "sh": sh,
        "sw": sw,
        "kb": kb,
        "nk": nk,
        "kb_last": kb_last,
        "a_block": a_block,
        "w_block": w_block,
        "sub": sub,
    }
    chain = d.chain
    identity = len(chain.operands) == 1 and not chain.ops.keys() - {"acc", "cast"}
    epilogue = None
    if not identity:
        free = 4 - extra
        row_cbs = dict(zip(map(id, d.rows), cb_rows, strict=True))

        def epilogue(j):
            _epilogue(c, chain, sub, sw, j, free, row_cbs, row_block, v)

    with c.loop(blocks, name="blk"):
        for cb in cb_rows:
            c.wait(cb, row_block)
        _passes(c, shape, cb_a, cb_w, part, cb_out, epilogue)
        for cb in cb_rows:
            c.pop(cb, row_block)
    return p, v


def _epilogue(c, chain, sub, sw, j, free, cbs, row_block, v):
    """The chain on each of the subblock's `sub` slots in place; tile t is column
    j + t % sw of the block; other operands take the slots from `free` on."""
    with c.loop(sub, name="t") as t:
        spare = iter(range(free, 4))
        slots = [
            t if getattr(item, "source", None) == "acc" else next(spare)
            for item in chain.operands
        ]
        for slot, item in zip(slots, chain.operands, strict=True):
            if isinstance(item, Leaf) and item.source != "acc":
                cb = cbs[id(item.source)]
                c.format(srca=cb, srcb=cb)
                c.init("bcast_row", cb)
                c.bcast(cb, (j + t % sw).bounded(0, row_block), slot, "row")
        c.init(chain.fn.init)
        values = tuple(v.scalar(s) for s in chain.values)
        c.run(chain.fn, *slots, out=t, values=values)


def _passes(c, shape, cb_a, cb_w, part, cb_out, epilogue):
    """The metal level's `km.matmul_passes` with `epilogue(j)` run on each output
    subblock of the last K block before its pack (the unpacker set up again for the
    next subblock's products)."""
    bm, bn, sh, sw = shape["bm"], shape["bn"], shape["sh"], shape["sw"]
    kb, sub = shape["kb"], shape["sub"]

    def init():
        return c.matmul_init(cb_a, cb_w, 0, ct=sw, rt=sh, kt=kb)

    state = {}

    def one(kblk, first, last, partial):
        if first:
            state["product"] = init()
        c.wait(cb_a, shape["a_block"])
        c.wait(cb_w, shape["w_block"])
        if partial and not first and not last:
            with _partials(c):
                c.wait(part, part.pages)
                c.pop(part, part.pages)
        inner = shape["kb_last"] if last else kb
        again = last and epilogue is not None
        with c.loop(bm, sh, name="i") as i, c.loop(bn, sw, name="j") as j:
            c.acquire()
            if last and partial:
                c.srca_format(part)
                c.init("copy", part)
                with _partials(c):
                    c.wait(part, sub)
                c.copy_block(part, 0, 0, sub)
                with _partials(c):
                    c.pop(part, sub)
            if (last and partial) or again:
                c.format(srca=cb_w, srcb=cb_a)
                state["product"] = init()
            with c.loop(inner, name="k") as k:
                a_span = (sh - 1) * kb + 1
                a_at = (i * kb + k).bounded(0, shape["a_block"] - a_span + 1)
                w_at = (j + k * bn).bounded(0, shape["w_block"] - sw + 1)
                c.matmul(state["product"], a_at, w_at, 0)
            if again:
                epilogue(j)
            c.commit()
            target = cb_out if last else part
            with _partials(c):
                c.reserve(target, sub)
            if last and partial:
                c.pack_format(cb_out)
                c.pack_accumulate(0)
            elif not last:
                c.pack_accumulate(0 if first else 1, part)
            with c.loop(sub, name="t") as t:
                c.pack(t, target)
            c.release()
            with _partials(c):
                c.push(target, sub)
        c.pop(cb_a, shape["a_block"])
        c.pop(cb_w, shape["w_block"])

    with c.when(shape["nk"].eq(1)):
        c.pack_format(cb_out)
        one(as_expr(0), True, True, False)
    with c.otherwise():
        c.pack_format(part)
        one(as_expr(0), True, False, True)
        with c.loop(shape["nk"] - 1, start=1, name="kblk") as kblk:
            one(kblk, False, False, True)
        one(shape["nk"] - 1, False, True, True)
        c.pack_accumulate(0)


def shared_reads(d, runtime, env, machine_cols, cores):
    """(bytes read, bytes written) of one call of the shared form."""
    from kohakutt.dsl.tile.plan import TILE_BYTES

    store, _ = d.stores[0]
    mt, kt = d.ir.grid.count.eval(env), d.ir.loop.count.eval(env)
    nt = Int("size", value=("row_tiles", store.view.tensor.name, None)).eval(env)
    if "pr" in runtime:
        # 2-D: a and w read once each by their senders; a bias block per core.
        read = mt * kt * TILE_BYTES[d.a.tensor.dtype]
        read += kt * nt * TILE_BYTES[d.w.tensor.dtype]
        read += sum(
            cores * runtime["pc"] * TILE_BYTES[op.tensor.dtype] for op in d.rows
        )
        return read, mt * nt * TILE_BYTES[store.view.tensor.dtype]
    nbh, nbw = runtime["nbh"], runtime["nbw"]
    read = nbw * mt * kt * TILE_BYTES[d.a.tensor.dtype]
    read += machine_cols * nbh * kt * nt * TILE_BYTES[d.w.tensor.dtype]
    read += sum(cores * nbh * nt * TILE_BYTES[op.tensor.dtype] for op in d.rows)
    return read, mt * nt * TILE_BYTES[store.view.tensor.dtype]
