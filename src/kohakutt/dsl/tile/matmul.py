"""Matmul kernels of the tile level: a 2-D grid of output blocks (i, j), an
accumulator summed over a kd.range of K tiles on the matrix unit, and a vector chain
run on the registers before the pack. Design and lowering to a km.Program.
"""

from dataclasses import dataclass, field

import kohakutt.dsl.metal as km
from kohakutt.dsl.tile.ir import F32, Int, fail
from kohakutt.dsl.tile.plan import Leaf, Operand, _Compiler

# Most inner (K) tiles per push of a.
CHUNK = 32
# Most pages of a core's block of w (K tiles x the accumulator's columns).
W_LIMIT = 256


def _at(expr, i=3, j=5, k=7):
    """`expr` (an Int of the grid's units and the loop) at sample values."""
    if expr.kind == "const":
        return expr.value
    if expr.kind == "unit":
        return {"i": i, "j": j}[expr.value]
    if expr.kind == "loop":
        return k
    if expr.kind != "op":
        return None
    a, b = (_at(x, i, j, k) for x in expr.args)
    if a is None or b is None:
        return None
    return {"+": a + b, "-": a - b, "*": a * b, "//": a // b, "%": a % b}[expr.value]


def _is(expr, want):
    """True when `expr` equals `want(i, j, k)` at two sample points."""
    return all(_at(expr, *p) == want(*p) for p in ((3, 5, 7), (11, 2, 13)))


@dataclass(eq=False)
class MatmulDesign:
    """A matmul kernel class: a [M, K] @ w [K, N] in blocks of (1, nt) output tiles,
    then `chain` (the stored value, the accumulator in its slot) and extra row
    operands (`rows`: broadcast rows of [N] tensors such as a bias)."""

    ir: object
    a: Operand
    w: Operand
    nt: int
    chain: object
    rows: list = field(default_factory=list)
    scalars: dict = field(default_factory=dict)
    form: str = "matmul"
    dest: str = "fp32"
    checks: list = field(default_factory=list)
    operands: list = field(default_factory=list)
    stores: list = field(default_factory=list)
    reduce: object = None
    stat: object = None
    share: str | None = None


def design_matmul(ir, dest=None):
    """The MatmulDesign of traced kernel class `ir`, or a PlanError naming what does
    not fit the form `acc += load(a.tiles[i, k]) @ load(w.tiles[k, j*n:(j+1)*n])`."""
    g, loop = ir.grid, ir.loop
    acc, product = ir.accumulate
    nt = acc.attrs["shape"][1]
    if g.cols is None:
        fail("an accumulator kernel runs on kd.grid2d(rows=, cols=)", g.line)
    if dest == "bf16":
        fail("an accumulator needs fp32 registers")
    a_node, w_node = product.args
    if w_node.kind != "load":
        fail("the right operand of acc += a @ b is a kd.load block", w_node.line)
    prologue = None
    if a_node.kind == "tile":
        index = a_node.attrs["index"]
        if index.kind != "loop":
            fail(
                "the left operand is row.tile(k), k the kd.range variable", a_node.line
            )
        prologue = _prologue(ir, a_node.args[0], w_node, a_node.line)
        av = None
    elif a_node.kind == "load":
        av = a_node.attrs["view"]
        ok_a = av.kind == "at" and av.index.kind == "unit" and av.index.value == "i"
        if not (ok_a and _is(av.cols[0], lambda i, j, k: k)):
            fail("the left operand is kd.load(a.tiles[i, k])", a_node.line)
    else:
        fail("the left operand is kd.load(a.tiles[i, k]) or row.tile(k)", a_node.line)
    wv = w_node.attrs["view"]
    ok_w = wv.kind == "cols" and wv.index.kind == "loop"
    ok_w = ok_w and _is(wv.cols[0], lambda i, j, k: j * nt)
    ok_w = ok_w and _is(wv.cols[1], lambda i, j, k: (j + 1) * nt)
    if not ok_w:
        fail(
            f"the right operand is kd.load(w.tiles[k, j * {nt}:(j + 1) * {nt}])",
            w_node.line,
        )
    if len(ir.stores) != 1:
        fail("a matmul kernel stores one block", g.line)
    (store,) = ir.stores
    sv = store.view
    ok_s = sv.kind == "cols" and sv.index.kind == "unit" and sv.index.value == "i"
    ok_s = ok_s and _is(sv.cols[0], lambda i, j, k: j * nt)
    ok_s = ok_s and _is(sv.cols[1], lambda i, j, k: (j + 1) * nt)
    if not ok_s:
        fail(
            f"a matmul kernel stores out.tiles[i, j * {nt}:(j + 1) * {nt}]", store.line
        )
    out = MatmulDesign(
        ir, Operand(a_node, False, False), Operand(w_node, False, False), nt, None
    )
    rows = {}

    def leaf_of(node, offset, kind):
        if node.kind == "acc":
            return Leaf("acc", "acc")
        v = node.attrs.get("view")
        same = v is not None and v.kind == "cols" and v.index.kind == "const"
        same = same and _is(v.cols[0], lambda i, j, k: j * nt)
        if not same or kind != "bcast_row":
            fail(
                f"a matmul's chain reads x.tiles[0, j * {nt}:(j + 1) * {nt}].row() "
                "(a row such as a bias) besides the accumulator",
                node.line,
            )
        op = rows.setdefault(id(node), Operand(node, False, True))
        return Leaf(op, kind)

    def shapes(node):
        return "scalar" if node.kind in ("const", "param", "size") else "tile"

    out.chain = _Compiler(ir.name, shapes, out.scalars).chain(store.value, leaf_of)
    out.rows = list(rows.values())
    if out.chain.prev is not None:
        fail("a matmul's chain reads at most 4 blocks with the accumulator", store.line)
    if not any(getattr(x, "source", None) == "acc" for x in out.chain.operands):
        fail("the stored value reads the accumulator", store.line)
    if nt + len(out.chain.operands) - 1 > 4:
        fail(
            f"blocks of {nt} columns leave {4 - nt} register slots; the chain needs "
            f"{len(out.chain.operands) - 1}: use a narrower accumulator",
            store.line,
        )
    a_dtype = prologue.h.dtype if prologue else av.tensor.dtype
    if a_dtype != wv.tensor.dtype:
        fail("a and w have one format (the matrix unit's source order)", product.line)
    out.prologue = prologue
    if prologue is not None:
        out.a = None
        out.operands = [out.w, *out.rows, *prologue.operands]
        out.checks = list(prologue.checks)
        width = Int("size", value=("row_tiles", prologue.x.name, None))
        out.checks.append(("equal", width, loop.count, loop.line))
    else:
        out.operands = [out.a, out.w, *out.rows]
        out.checks = [("equal", _row_tiles(av), loop.count, loop.line)]
    out.checks.append(("equal", _row_tiles(wv), _row_tiles(sv), store.line))
    out.stores = [(store, None)]
    out.share = w_node.attrs.get("share")
    if out.share is not None or (av is not None and a_node.attrs.get("share")):
        from kohakutt.dsl.tile.shared import check_shared

        check_shared(out)
    return out


def _prologue(ir, h, w_node, line):
    """The row design computing row block `h` (the matmul's left operand) for each
    unit i: a row kernel whose store is a buffer of the program."""
    from kohakutt.dsl.tile.ir import Grid, KernelIR, Store, TensorParam, View
    from kohakutt.dsl.tile.plan import design

    dtype = w_node.attrs["view"].tensor.dtype
    if h.kind != "cast" or h.attrs["dtype"] != dtype:
        fail(
            f"the row feeding the matmul is cast to w's format: .cast({dtype!r})", line
        )
    rows = [n for n in _walk(h) if n.kind == "load"]
    if not rows:
        fail("the row feeding the matmul loads a row of x", line)
    x = rows[0].attrs["view"].tensor
    param = TensorParam("_h", True, dtype)
    sub = KernelIR(f"{ir.name}_rows", [*ir.tensors, param], ir.values, ir.statics)
    sub.grid = Grid(ir.grid.count, 1, line=ir.grid.line)
    sub.stores = [Store(View(param, "row", ir.grid.unit), h, line)]
    sub.grid.unit = Int("unit", value="i")
    out = design(sub, "fp32")
    if out.form != "row":
        fail("the row feeding the matmul is a row block x.tiles[i, :]", line)
    for op in out.operands:
        if op.row and not op.resident:
            op.held = True
    out.h, out.x = param, x
    # The buffer's width is x's: its row-tile size reads x's sizes.
    out.checks = [c for c in out.checks if "_h" not in _texts(c)]
    return out


def _walk(node, seen=None):
    seen = set() if seen is None else seen
    if id(node) in seen:
        return []
    seen.add(id(node))
    return [node] + [n for a in node.args for n in _walk(a, seen)]


def _texts(check):
    return " ".join(x.text() for x in check[1:3] if hasattr(x, "text"))


def _row_tiles(view):
    return Int("size", value=("row_tiles", view.tensor.name, None))


def _rows(view):
    return Int("size", value=("tile_rows", view.tensor.name, None))


def lower_matmul(d, machine):
    """(km.Program, Values) of matmul design `d` on `machine`, in the layout of the
    backend's matmul program: column groups of `nt` tiles, a core holding its group's
    block of w and summing one output row tile at a time in the registers."""
    from kohakutt.dsl.tile.lower import Values, _Lowering, _Units

    ir, nt, pro = d.ir, d.nt, d.prologue
    p = km.Program(ir.name, machine)
    v = Values(p)
    refs = {}

    def ref(tensor):
        if tensor.name not in refs:
            refs[tensor.name] = p.input(tensor.name, tensor.dtype)
        return refs[tensor.name]

    a = None if pro else ref(d.a.tensor)
    w = ref(d.w.tensor)
    rows = [ref(op.tensor) for op in d.rows]
    store, _ = d.stores[0]
    out = p.output(store.view.tensor.name, store.view.tensor.dtype)
    kt = v.host_value(ir.loop.count)
    ntot = v.host_value(_row_tiles(store.view))
    mt = v.host_value(ir.grid.count)
    low = None
    if pro:
        for op in pro.operands:
            ref(op.tensor)
        low = _Lowering(pro, machine, 1, "hold", 1, share=(p, v, refs))
        low.seg_limit = W_LIMIT // 4
    kc = p.runtime("kc", choices=range(1, CHUNK + 1))
    groups, per_group = p.runtime("groups"), p.runtime("per_group")
    v.extra = {
        "kc": lambda env, cores: min(env[kt.name], CHUNK),
        "groups": lambda env, cores: -(-env[ntot.name] // nt),
        "per_group": lambda env, cores: max(
            1, min(env[mt.name], cores // -(-env[ntot.name] // nt))
        ),
    }
    p.layout(cores=groups * per_group)
    place = p.define("place", p.core % per_group)
    n0 = p.define("n0", p.core // per_group * nt)
    cols = p.define("cols", km.min(nt, ntot - n0))
    m0 = p.define("m0", mt * place // per_group)
    count = p.define("rows", mt * (place + 1) // per_group - m0)
    w_pages = p.sized("w_pages", of=kt * nt, limit=W_LIMIT)
    if pro:
        seg = low.prologue_buffers(kt)
        cb_a = p.cb(dtype=pro.h.dtype, pages=seg, name="h")
    else:
        a_pages = p.sized("a_pages", of=kc, limit=CHUNK)
        cb_a = p.cb(like=a, pages=2 * a_pages)
    cb_w = p.cb(like=w, pages=w_pages)
    cb_rows = [p.cb(like=r, pages=nt) for r in rows]
    cb_out = p.cb(like=out, pages=2 * nt)

    k = p.kernel("dm0")
    block = k.reserve(cb_w, w_pages)
    with k.loop(kt, name="k") as i, k.loop(cols, name="j") as j:
        n = k.let("n", n0 + j)
        k.read(w.page(i * ntot + n), block[(i * cols + j).bounded(0, w_pages)])
    k.reads_done()
    k.push(cb_w, w_pages)
    for ref, cb in zip(rows, cb_rows, strict=True):
        slot = k.reserve(cb, nt)
        with k.loop(cols, name="j") as j:
            k.read(ref.page(n0 + j), slot[j.bounded(0, nt)])
        k.reads_done()
        k.push(cb, nt)
    if pro:
        low.prologue_read(k, kt, once=True)
        with k.loop(count, name="m") as m:
            v.unit = k.let("r", m0 + m)
            low.prologue_read(k, kt, once=False)
    else:
        m_loop = k.loop(count, name="m")
        with m_loop as m, k.block_loop(kt, kc, names=("k0", "count")) as (k0, got):
            chunk = k.reserve(cb_a, a_pages)
            with k.loop(got, name="k") as i:
                k.read(a.page((m0 + m) * kt + k0 + i), chunk[i])
            k.reads_done()
            k.push(cb_a, a_pages)

    cbs = [cb_a, cb_w, *cb_rows, cb_out, *(low.cbs.values() if pro else ())]
    fp32 = [cb for cb in cbs if cb.dtype == F32]
    c = p.kernel(
        "compute", fp32_dest=True, unpack_fp32=fp32, approx=False, exact=bool(pro)
    )
    c.startup(cb_a, cb_w, cb_out)
    chain = d.chain
    identity = len(chain.operands) == 1 and not chain.ops.keys() - {"acc", "cast"}
    again = pro or not identity
    if not again:
        product = c.matmul_init(cb_a, cb_w, transpose=0, ct=cols, rt=1, kt=kc)
    c.wait(cb_w, w_pages)
    for cb in cb_rows:
        c.wait(cb, nt)
    if pro:
        units = _Units(c, cb_a, False)
        low.prologue_wait(c, once=True)
    with c.loop(count, name="m"):
        if pro:
            low.prologue_wait(c, once=False)
            low.prologue_compute(c, units, kt, cb_a)
            low.prologue_wait(c, once=False, pop=True)
            c.wait(cb_a, seg)
        c.reserve(cb_out, nt)
        c.acquire()
        if again:
            c.format(srca=cb_w, srcb=cb_a)
            product = c.matmul_init(cb_a, cb_w, transpose=0, ct=cols, rt=1, kt=kc)
        if pro:
            with c.loop(kt, name="k") as i:
                at = (i * cols).bounded(0, w_pages - cols + 1)
                c.matmul(product, i.bounded(0, seg), at, 0)
        else:
            with c.block_loop(kt, kc, names=("k0", "count")) as (k0, got):
                c.wait(cb_a, a_pages)
                with c.loop(got, name="k") as i:
                    at = ((k0 + i) * cols).bounded(0, w_pages - cols + 1)
                    c.matmul(product, i, at, 0)
                c.pop(cb_a, a_pages)
        if not identity:
            _chain(c, chain, cols, nt, dict(zip(map(id, d.rows), cb_rows)), v)
        c.commit()
        if pro:
            c.pack_format(cb_out)
        with c.loop(cols, name="j") as j:
            c.pack(j, cb_out)
        c.release()
        c.push(cb_out, nt)
        if pro:
            c.pop(cb_a, seg)
    if pro:
        low.prologue_wait(c, once=True, pop=True)
    for cb in cb_rows:
        c.pop(cb, nt)
    c.pop(cb_w, w_pages)

    wr = p.kernel("dm1")
    with wr.loop(count, name="m") as m:
        wr.wait(cb_out, nt)
        with wr.loop(cols, name="j") as j:
            wr.write(cb_out, out.page((m0 + m) * ntot + n0 + j), index=j)
        wr.writes_done()
        wr.pop(cb_out, nt)
    return p, v


# Matrix-unit rate of one core at HiFi2 (docs/concepts/hardware.md), FLOP per second.
MATRIX_RATE = 2.76e12


@dataclass
class MatmulPlan:
    """One call's plan of a matmul kernel: sizes, cores, bytes, multiply work."""

    design: MatmulDesign
    runtime: dict
    cores: int
    read: int
    written: int
    flops: int
    l1: int
    program: object = None

    def bounds(self):
        """{name: seconds} of each lower bound on the kernel's time."""
        from kohakutt.dsl.tile.plan import CORE_READ_RATE, DRAM_RATE

        per_core = self.read / self.cores
        return {
            "DRAM (512 GB/s)": (self.read + self.written) / DRAM_RATE,
            "per-core reads (estimate)": per_core / CORE_READ_RATE,
            "matrix unit (HiFi2)": self.flops / (self.cores * MATRIX_RATE),
        }

    def report(self):
        """The plan report: what was chosen and what bounds it."""
        d, r = self.design, self.runtime
        if d.share and "pr" in r:
            m = self.program.machine
            lines = [
                f"kernel    {d.ir.name} (matmul form, a and w shared, fp32 registers)",
                f"grid      {m.cols} x {m.rows} cores, {r['pr']} x {r['pc']} output "
                f"tiles each, K in {r['nk']} blocks of {r['kb']}, subblock "
                f"{r['sh']} x {r['sw']}",
                f"shared    {d.a.tensor.name}: the left core of each core row reads "
                "its rows' blocks and multicasts them along the row",
                f"shared    {d.w.tensor.name}: the top core of each core column reads "
                "its columns' blocks and multicasts them down the column",
            ]
            lines += [
                f"stream    {op.tensor.name}: the core's columns (row broadcast)"
                for op in d.rows
            ]
            return self._tail(lines)
        if d.share:
            lines = [
                f"kernel    {d.ir.name} (matmul form, w shared, fp32 registers)",
                f"grid      {self.cores} cores, {r['pcm']} output row tiles each, "
                f"blocks {r['bm']} x {r['bn']} tiles ({r['nbh']} x {r['nbw']}), "
                f"K in {r['nk']} blocks of {r['kb']}, subblock {r['sh']} x {r['sw']}",
                f"stream    {d.a.tensor.name}: each core its own rows, once per "
                "column block",
                f"shared    {d.w.tensor.name}: the top core of each core column reads "
                "each block and multicasts it down the column",
            ]
            lines += [
                f"stream    {op.tensor.name}: the block's columns (row broadcast)"
                for op in d.rows
            ]
            return self._tail(lines)
        lines = [
            f"kernel    {d.ir.name} (matmul form, fp32 registers)",
            f"grid      {r['groups']} column groups of {d.nt} tiles, "
            f"{r['per_group']} cores per group ({self.cores} cores)",
            (
                f"prologue  {d.prologue.x.name}: each row held, its row chain packed "
                f"in L1 as the left operand (exact statistic: {d.prologue.reduce
                is not None})"
                if d.prologue
                else f"stream    {d.a.tensor.name}: K chunks of {r['kc']} tiles per "
                "output row"
            ),
            f"resident  {d.w.tensor.name}: the core's block, K x {d.nt} tiles",
        ]
        lines += [
            f"resident  {op.tensor.name}: {d.nt} tiles (row broadcast)" for op in d.rows
        ]
        return self._tail(lines)

    def _tail(self, lines):
        lines.append(
            f"vector    {self.design.chain.fn.name} on the accumulator before the pack"
        )
        lines.append(f"L1        {self.l1 // 1024} KB of buffers per core")
        moved = self.read + self.written
        lines.append(
            f"moved     {moved / 1e6:.2f} MB ({self.read / 1e6:.2f} read, "
            f"{self.written / 1e6:.2f} written), multiply work "
            f"{self.flops / 1e9:.2f} GFLOP, launches 1"
        )
        bounds = self.bounds()
        shown = ", ".join(f"{k} {v * 1e6:.1f} us" for k, v in bounds.items())
        lines.append(f"bound     {shown}")
        lines.append(f"binds     {max(bounds, key=bounds.get)}")
        return "\n".join(lines)

    __str__ = report


def matmul_plan(d, program, runtime, env):
    """The MatmulPlan of one call with launch values `runtime` and sizes `env`."""
    from kohakutt.dsl.tile.plan import TILE_BYTES

    store, _ = d.stores[0]
    mt, kt = d.ir.grid.count.eval(env), d.ir.loop.count.eval(env)
    ntot = _row_tiles(store.view).eval(env)
    sizes = program.sizes(runtime)
    if d.share:
        from kohakutt.dsl.tile.shared import shared_reads

        cores = program.machine.cores
        read, written = shared_reads(d, runtime, env, program.machine.cols, cores)
        flops = 2 * mt * ntot * kt * 32**3
        l1 = program.footprint(sizes)
        return MatmulPlan(d, runtime, cores, read, written, flops, l1, program)
    groups, per_group = runtime["groups"], runtime["per_group"]
    cores = groups * per_group
    left = d.prologue.x if d.prologue else d.a.tensor
    a_bytes = TILE_BYTES[left.dtype]
    w_bytes = TILE_BYTES[d.w.tensor.dtype]
    read = groups * mt * kt * a_bytes + per_group * kt * ntot * w_bytes
    read += sum(per_group * ntot * TILE_BYTES[op.tensor.dtype] for op in d.rows)
    if d.prologue:
        once = [op for op in d.prologue.operands if op.resident and op.row]
        read += sum(cores * kt * TILE_BYTES[op.tensor.dtype] for op in once)
    written = mt * ntot * TILE_BYTES[store.view.tensor.dtype]
    flops = 2 * mt * ntot * kt * 32**3
    sizes = program.sizes(runtime)
    return MatmulPlan(
        d, runtime, cores, read, written, flops, program.footprint(sizes), program
    )


def _chain(c, chain, cols, nt, cbs, v):
    """Runs `chain` on each accumulator slot j < cols in place; its other operands
    take the slots from `nt` on."""
    with c.loop(cols, name="j") as j:
        free = iter(range(nt, 4))
        slots = [
            j if getattr(item, "source", None) == "acc" else next(free)
            for item in chain.operands
        ]
        for slot, item in zip(slots, chain.operands, strict=True):
            if isinstance(item, Leaf) and item.source != "acc":
                cb = cbs[id(item.source)]
                c.format(srca=cb, srcb=cb)
                c.init("bcast_row", cb)
                c.bcast(cb, j, slot, "row")
        c.init(chain.fn.init)
        values = tuple(v.scalar(s) for s in chain.values)
        c.run(chain.fn, *slots, out=j, values=values)
