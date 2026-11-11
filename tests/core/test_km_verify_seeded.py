"""Seeded bugs: one minimal failing program per verifier rule of the metal level; the
verifier must reject each with that rule, and accept the program they are cut from."""

import contextlib
import json
from types import SimpleNamespace

import pytest
from km_programs import MACHINE, rope, swiglu

import kohakutt.dsl.metal as km
from km_lanes import lane_program
from km_rows import rows_program
from km_tables import table_program
from kohakutt.dsl.metal import body as km_body

DTYPES = {"x": "bfloat16", "out": "bfloat16"}


def good_reader(k, e):
    for i in k.range(e.count):
        slot = e.cb_in.reserve(1)
        k.read(e.x.page(e.start + i), slot)
        k.reads_done()
        e.cb_in.push(1)


def good_compute(c, e):
    c.startup(e.cb_in, e.cb_in, e.cb_out)
    c.init("copy", e.cb_in)
    for _ in c.range(e.count):
        e.cb_in.wait(1)
        with c.registers():
            c.copy(e.cb_in, 0, 0)
            e.cb_out.reserve(1)
            c.pack(0, e.cb_out)
            e.cb_out.push(1)
        e.cb_in.pop(1)


def good_writer(k, e):
    for i in k.range(e.count):
        e.cb_out.wait(1)
        k.write(e.cb_out, e.out.page(e.start + i))
        k.flushed()
        e.cb_out.pop(1)
    k.writes_done()


def build(
    reader=good_reader,
    compute=good_compute,
    writer=good_writer,
    setup=None,
    pages=2,
    nocs=(None, None),
    budget=None,
):
    """Traces a copy program x -> out; the arguments replace its parts."""

    @km.program("seed")
    def seed(p, x: km.TensorRef, out: km.Out):
        start, count = p.split(p.runtime("tiles"))
        e = SimpleNamespace(p=p, x=x, out=out, start=start, count=count)
        e.cb_in = p.cb(0, like=x, pages=pages)
        e.cb_out = p.cb(16, like=out, pages=2)
        if setup:
            setup(e)
        p.dm0(noc=nocs[0])(lambda k: reader(k, e))
        p.compute(lambda c: compute(c, e))
        p.dm1(noc=nocs[1])(lambda k: writer(k, e))

    return seed.trace(MACHINE, DTYPES, l1_budget=budget)


def cb_1():
    def reader(k, e):
        for done, n in k.blocks(e.count, 3):
            slot = e.cb_in.reserve(3)
            for i in k.range(n):
                k.read(e.x.page(e.start + done + i), slot[i])
            k.reads_done()
            e.cb_in.push(3)

    build(reader=reader, pages=8)


def cb_2():
    def compute(c, e):
        c.startup(e.cb_in, e.cb_in, e.cb_out)
        c.init("copy", e.cb_in)
        for _ in c.range(e.count):
            e.cb_in.wait(1)
            with c.registers():
                c.copy(e.cb_in, 0, 0)
                e.cb_out.reserve(1)
                c.pack(0, e.cb_out)
                e.cb_out.push(1)
        e.cb_in.pop(1)

    build(compute=compute)


def cb_3():
    def reader(k, e):
        for i in k.range(e.count):
            slot = e.cb_in.reserve(2)
            k.read(e.x.page(e.start + i), slot[3])
            k.reads_done()
            e.cb_in.push(2)

    build(reader=reader)


def cb_4():
    def compute(c, e):
        good_compute(c, e)
        e.cb_in.reserve(1)
        e.cb_in.push(1)

    build(compute=compute)


def cb_5():
    def setup(e):
        e.kept = e.p.cb(24, dtype="float32", pages=1)

    def compute(c, e):
        c.startup(e.cb_in, e.cb_in, e.cb_out)
        c.init("copy", e.cb_in)
        for _ in c.range(e.count):
            e.cb_in.wait(1)
            with c.registers():
                c.copy(e.cb_in, 0, 0)
                e.kept.reserve(1)
                c.pack(0, e.kept)
            e.kept.wait(1)
            with c.registers():
                c.copy(e.kept, 0, 0)
                e.cb_out.reserve(1)
                c.pack(0, e.cb_out)
                e.cb_out.push(1)
            e.cb_in.pop(1)

    build(compute=compute, setup=setup)


def noc_1():
    build(nocs=(0, 0))


def noc_2():
    def reader(k, e):
        for i in k.range(e.count):
            slot = e.cb_in.reserve(1)
            k.read(e.x.span(e.start + i, 32, 32), slot.at(0))
            k.reads_done()
            e.cb_in.push(1)

    build(reader=reader)


def noc_3():
    def reader(k, e):
        for i in k.range(e.count):
            slot = e.cb_in.reserve(1)
            k.read(e.x.page(e.start + i), slot)
            e.cb_in.push(1)
        k.reads_done()

    build(reader=reader)


def sem_1():
    def setup(e):
        e.ready = e.p.semaphore(cores=e.p.cores.rows(0), name="ready")

    def reader(k, e):
        k.inline("// waits for the row's sender", sems=[e.ready])
        good_reader(k, e)

    build(reader=reader, setup=setup)


def arg_1():
    def setup(e):
        e.p.per_core("jobs", values=lambda core: [core], count=e.p.runtime("jobs"))

    build(setup=setup)


def bind_1():
    def setup(e):
        e.p.cb(2, dtype="bfloat16", pages=e.p.runtime("tiles"))

    build(setup=setup)


def bind_1_python_branch():
    def reader(k, e):
        if e.count < 4:
            return
        good_reader(k, e)

    build(reader=reader)


def bind_2():
    def setup(e):
        e.p.per_core("first", values=e.p.runtime("first"))

    build(setup=setup)


def sfpu_1():
    def compute(c, e):
        c.inline("TTI_SFPCONFIG(0, 11, 1);")
        good_compute(c, e)

    build(compute=compute)


def dst_1():
    def compute(c, e):
        c.startup(e.cb_in, e.cb_in, e.cb_out)
        c.init("copy", e.cb_in)
        for _ in c.range(e.count):
            e.cb_in.wait(1)
            with c.registers():
                c.copy(e.cb_in, 0, 5)
                e.cb_out.reserve(1)
                c.pack(5, e.cb_out)
                e.cb_out.push(1)
            e.cb_in.pop(1)

    build(compute=compute)


def reg_1():
    def compute(c, e):
        c.startup(e.cb_in, e.cb_in, e.cb_out)
        c.init("copy", e.cb_in)
        for _ in c.range(e.count):
            e.cb_in.wait(1)
            with c.registers():
                c.copy(e.cb_in, 0, 0)
            e.cb_out.reserve(1)
            c.pack(0, e.cb_out)
            e.cb_out.push(1)
            e.cb_in.pop(1)

    build(compute=compute)


def inl_1():
    def reader(k, e):
        good_reader(k, e)
        k.inline("cb_reserve_back(3, 1);")

    build(reader=reader)


def l1_1():
    build(budget=4096)


def counted_reader(choices, reads=None, whole=True):
    """A reader pushing `n` pages per step over `items` pages, `n` a runtime value
    of `choices` (None: unbounded); `reads` pages are read per step (default: n);
    `whole` declares `items` a multiple of `n`."""

    def reader(k, e):
        e.n = e.p.runtime("n", choices=choices)
        e.items = e.p.runtime("items", multiple_of="n" if whole else None)
        for r in k.range(e.items, step=e.n, name="r"):
            slot = e.cb_in.reserve(e.n)
            for i in k.range(e.n if reads is None else reads):
                k.read(e.x.page(e.start + r + i), slot[i])
            k.reads_done()
            e.cb_in.push(e.n)

    return reader


def counted_compute(c, e):
    """The compute kernel matching counted_reader: n pages per step."""
    c.startup(e.cb_in, e.cb_in, e.cb_out)
    c.init("copy", e.cb_in)
    for _ in c.range(e.items, step=e.n, name="r"):
        e.cb_in.wait(e.n)
        for i in c.range(e.n):
            with c.registers():
                c.copy(e.cb_in, i, 0)
                e.cb_out.reserve(1)
                c.pack(0, e.cb_out)
                e.cb_out.push(1)
        e.cb_in.pop(e.n)


def counted_writer(k, e):
    """The writer matching counted_compute: one page per item of each step."""
    for r in k.range(e.items, step=e.n, name="r"):
        for i in k.range(e.n):
            e.cb_out.wait(1)
            k.write(e.cb_out, e.out.page(e.start + r + i))
            k.flushed()
            e.cb_out.pop(1)
    k.writes_done()


def cb_1_choice():
    build(reader=counted_reader((2, 3)), compute=counted_compute, pages=8)


def bind_1_unbounded_count():
    build(reader=counted_reader(None), compute=counted_compute, pages=8)


def cb_3_beyond_a_counted_reserve():
    build(reader=counted_reader((1, 2), reads=2), compute=counted_compute)


def cb_2_counted():
    def compute(c, e):
        c.startup(e.cb_in, e.cb_in, e.cb_out)
        c.init("copy", e.cb_in)
        for _ in c.range(e.items):
            e.cb_in.wait(e.n)
            with c.registers():
                c.copy(e.cb_in, 0, 0)
                e.cb_out.reserve(1)
                c.pack(0, e.cb_out)
                e.cb_out.push(1)
            e.cb_in.pop(e.n)

    build(reader=counted_reader((1, 2)), compute=compute)


def dst_1_value():
    def compute(c, e):
        slot = e.p.runtime("slot", choices=(1, 2, 3, 4))
        c.startup(e.cb_in, e.cb_in, e.cb_out)
        c.init("copy", e.cb_in)
        for _ in c.range(e.count):
            e.cb_in.wait(1)
            with c.registers():
                c.copy(e.cb_in, 0, slot)
                e.cb_out.reserve(1)
                c.pack(slot, e.cb_out)
                e.cb_out.push(1)
            e.cb_in.pop(1)

    build(compute=compute)


def reg_1_commit_in_a_loop():
    def compute(c, e):
        c.startup(e.cb_in, e.cb_in, e.cb_out)
        c.init("copy", e.cb_in)
        for _ in c.range(e.count):
            e.cb_in.wait(1)
            e.cb_out.reserve(1)
            with c.registers():
                c.copy(e.cb_in, 0, 0)
                for _ in c.range(1, name="k"):
                    c.pack(0, e.cb_out)
            e.cb_out.push(1)
            e.cb_in.pop(1)

    build(compute=compute)


def loop_1():
    """A stepped loop over a count not declared a multiple of the step."""
    build(
        reader=counted_reader((1, 2), whole=False),
        compute=counted_compute,
        writer=counted_writer,
        pages=4,
    )


def looked_up_reader(bound=True, early=False, word=0):
    """A reader that reads a page, loads a word of it and reads the page that word
    names. `bound`: the word's range is declared; `early`: it is loaded before the
    first read lands; `word`: which word."""

    def reader(k, e):
        pages = e.p.runtime("pages")
        for i in k.range(e.count):
            slot = e.cb_in.reserve(1)
            k.read(e.x.page(e.start + i), slot)
            if not early:
                k.reads_done()
            found = slot.load32(word)
            found = k.let("found", found.bounded(0, pages) if bound else found)
            k.reads_done()
            k.read(e.x.page(found), slot)
            k.reads_done()
            e.cb_in.push(1)

    return reader


def data_1_page():
    build(reader=looked_up_reader(bound=False))


def data_1_loop_bound():
    def reader(k, e):
        for i in k.range(e.count):
            slot = e.cb_in.reserve(1)
            k.read(e.x.page(e.start + i), slot)
            k.reads_done()
            for j in k.range(slot.load32(0), name="j"):
                k.let("unused", j)
            e.cb_in.push(1)

    build(reader=reader)


def noc_3_load_before_the_read_lands():
    build(reader=looked_up_reader(early=True))


def cb_3_load_past_the_reserve():
    build(reader=looked_up_reader(word=512))


def cb_2_push_in_a_condition():
    def reader(k, e):
        for i in k.range(e.count):
            slot = e.cb_in.reserve(1)
            k.read(e.x.page(e.start + i), slot)
            k.reads_done()
            with k.when(i < 4):
                e.cb_in.push(1)

    build(reader=reader)


def half_page_reader(like=False, staged=64):
    """A reader of half pages at a value offset (a multiple of 32) into byte
    `staged` of its page; `like` builds the destination congruent to the source."""

    def reader(k, e):
        for i in k.range(e.count):
            slot = e.cb_in.reserve(1)
            src = k.let("src", (i & 31) * 32)
            dst = slot.at(staged, like=src) if like else slot.at(staged)
            k.read(e.x.span(e.start + i, src, 1024), dst)
            k.reads_done()
            e.cb_in.push(1)

    return reader


def noc_2_value_offset():
    build(reader=half_page_reader())


def cb_3_congruent_slot_overruns():
    build(reader=half_page_reader(like=True, staged=1024))


def moved(barrier="writes_done", settle=True):
    """A one-kernel program moving x to out through one held page, its items dealt
    out over the cores. `barrier`: what follows each write; `settle`: the read is
    waited for before the write."""

    @km.program("seed_moved")
    def program(p, x: km.TensorRef, out: km.Out):
        tiles = p.runtime("tiles")
        first, step = p.stride(tiles)
        held = p.cb(0, like=x, pages=1)

        @p.dm0
        def mover(k):
            slot = held.reserve(1)
            for u in k.range(tiles, start=first, step=step, name="u"):
                k.read(x.page(u), slot)
                if settle:
                    k.reads_done()
                k.write(slot, out.page(u))
                if not settle:
                    k.reads_done()
                if barrier:
                    getattr(k, barrier)()
            k.writes_done()

    return program.trace(MACHINE, DTYPES)


def noc_3_read_into_a_page_being_written():
    moved(barrier=None)


def noc_3_write_before_the_read_lands():
    moved(settle=False)


def units(body, before=None, pages=2, waited=1):
    """build() with a compute kernel whose register scope per item is `body(c, e)`
    after tile 0 is copied to slot 0 (reserve, pack and push of cb_out included when
    `body` returns None); `before(c, e)` runs once, ahead of the loop. An item is
    `waited` pages of cb_in."""

    def setup(e):
        e.kept = e.p.cb(24, dtype="float32", pages=1)

    def reader(k, e):
        for i in k.range(e.count):
            slot = e.cb_in.reserve(waited)
            k.read(e.x.page(e.start + i), slot)
            k.reads_done()
            e.cb_in.push(waited)

    def compute(c, e):
        c.startup(e.cb_in, e.cb_in, e.cb_out)
        if before:
            before(c, e)
        for _ in c.range(e.count):
            e.cb_in.wait(waited)
            with c.registers():
                c.init("copy", e.cb_in)
                c.copy(e.cb_in, 0, 0)
                if body(c, e) is None:
                    e.cb_out.reserve(1)
                    c.pack(0, e.cb_out)
                e.cb_out.push(1)
            e.cb_in.pop(waited)

    return build(reader=reader, compute=compute, setup=setup, pages=pages)


def dst_1_block():
    def body(c, e):
        product = c.matmul_init(e.cb_in, e.cb_in, ct=4, rt=2, kt=1)
        c.matmul(product, 0, 0, 0)

    units(body, pages=8, waited=4)


def mm_1():
    def body(c, e):
        product = c.matmul_init(e.cb_in, e.cb_in)
        c.init("copy", e.cb_in)
        c.copy(e.cb_in, 0, 1)
        c.matmul(product, 0, 0, 0)

    units(body)


def init_1():
    units(lambda c, e: c.op("rsqrt", 0))


def init_1_another_ops_setup():
    def body(c, e):
        c.init("relu")
        c.op("rsqrt", 0)

    units(body)


def _gate():
    """A generated function of three operands that needs the sigmoid's setup."""
    a, b, c = km.sfpu.operands(3)
    return km.sfpu_fn("gate", km.sfpu.sigmoid(a * b) + c)


def init_1_generated_function():
    def body(c, e):
        c.init("mul")
        c.run(_gate(), 0, 0, 0, out=0)

    units(body)


def dst_1_function_operand():
    def body(c, e):
        fn = _gate()
        c.init(fn.init)
        c.run(fn, 0, 0, 4, out=0)

    units(body)


def raw(code, clobbers=()):
    """units() running inline vector-unit function `fn(a, b, out)` with body `code`
    on slot 0; `clobbers`: the shared constants it declares."""
    text = f"inline void fn(uint32_t a, uint32_t b, uint32_t out) {{ {code} }}"

    def body(c, e):
        c.init("mul")
        c.sfpu_binary(km.SfpuFn("fn", text, clobbers), 0, 0, 0)

    return units(body)


LOAD = "TT_SFPLOAD(1, 0, ADDR_MOD_7, a * 64); TT_SFPLOAD(2, 0, ADDR_MOD_7, b * 64);"
STORE = "TT_SFPSTORE(5, 0, ADDR_MOD_7, out * 64);"


def sfpu_2():
    raw(f"{LOAD} TTI_SFPMAD(1, 2, 9, 5, 0); TTI_SFPSWAP(0, 5, 2, 1); {STORE}")


def sfpu_2_shift_of_a_product():
    raw(f"{LOAD} TTI_SFPMUL(1, 2, 9, 5, 0); TTI_SFPSHFT(1, 5, 5, 1); {STORE}")


def sfpu_2_table_lookup():
    raw(f"{LOAD} TTI_SFPLUTFP32(5, 2); TTI_SFPSETSGN(0, 5, 5, 0); {STORE}")


def sfpu_2_constant_from_a_sum():
    code = f"{LOAD} TTI_SFPADD(10, 1, 2, 0, 0); TTI_SFPCONFIG(0, 12, 0); {STORE}"
    raw(code, clobbers=["lreg11"])


def reg_2():
    def body(c, e):
        c.init("mul")
        c.op("mul", 0, 1, 0)

    units(body)


def statistics(fill=True):
    """Sums of squares of a held 4-page item streamed into slot 1 (filled with 0 when
    `fill`), then a second pass over the same pages into slot 2; slot 1 packed."""
    a, b = km.sfpu.operands(2)
    acc = km.sfpu_fn("acc", a * a + b)

    def reader(k, e):
        for i in k.range(e.count):
            slot = e.cb_in.reserve(4)
            for j in k.range(4, name="j"):
                k.read(e.x.page(e.start + i), slot[j])
            k.reads_done()
            e.cb_in.push(4)

    def compute(c, e):
        c.startup(e.cb_in, e.cb_in, e.cb_out)
        for _ in c.range(e.count):
            e.cb_in.wait(4)
            c.acquire()
            if fill:
                c.init("fill")
                c.op("fill", 1, value=0.0)
            for j in c.range(4, name="j"):
                c.init("copy", e.cb_in)
                c.copy(e.cb_in, j, 0)
                c.init(acc.init)
                c.run(acc, 0, 1, out=1)
            c.init("copy", e.cb_in)
            for j in c.range(4, name="j2"):
                c.copy(e.cb_in, j, 2)
            e.cb_out.reserve(1)
            c.pack(1, e.cb_out)
            e.cb_out.push(1)
            c.release()
            e.cb_in.pop(4)

    return build(reader=reader, compute=compute, pages=8)


def reg_2_statistics_slot_not_filled():
    statistics(fill=False)


def reduced(c, e, dims=("row",), done=True):
    """Reductions of tile 0 of cb_in along `dims` into slots 0, 1, ...; slot 0 packed
    into cb_out, then the masks cleared when `done`."""
    for slot, dim in enumerate(dims):
        r = c.reduce_init("sum", dim, e.cb_in, e.cb_in, e.cb_out)
        c.reduce(r, 0, 0, slot)
    e.cb_out.reserve(1)
    c.pack(0, e.cb_out)
    if done:
        c.reduce_done()
    return True


def red_1():
    def body(c, e):
        r = c.reduce_init("sum", "row", e.cb_in, e.cb_in, e.cb_out)
        c.reduce(r, 0, 0, 1)

    units(body)


def red_1_two_dims_in_one_scope():
    units(lambda c, e: reduced(c, e, dims=("row", "col")))


def red_1_masks_left_in_effect():
    units(lambda c, e: reduced(c, e, done=False))


def init_1_tiles_under_another_setup():
    def body(c, e):
        add = c.tiles_init("add", e.cb_in, e.cb_in)
        c.init("copy", e.cb_in)
        c.tiles(add, 0, 0, 1)

    units(body)


def init_1_another_way_of_unpacking():
    units(lambda c, e: c.transpose(e.cb_in, 0, 1))


def init_1_setup_under_a_condition():
    def body(c, e):
        with c.when(e.p.runtime("flag").ne(0)):
            c.init("abs")
        c.op("abs", 0)

    units(body)


def pack_1():
    def body(c, e):
        e.cb_out.reserve(1)
        c.pack(0, e.cb_out)
        c.pack(0, e.kept)
        return True

    units(body, before=lambda c, e: e.kept.reserve(1))


def acc_1():
    def body(c, e):
        e.cb_out.reserve(1)
        c.pack_accumulate(1, e.cb_out)
        c.pack(0, e.cb_out, at=0)
        c.pack_accumulate(0)
        return True

    units(body)


def acc_1_left_on():
    def body(c, e):
        e.cb_out.reserve(1)
        c.pack(0, e.cb_out, at=0)
        c.pack_accumulate(1, e.cb_out)
        c.pack(0, e.cb_out, at=0)
        return True

    units(body)


def acc_1_runtime_switch():
    def body(c, e):
        e.cb_out.reserve(1)
        c.pack(0, e.cb_out, at=0)
        c.pack_accumulate(e.p.runtime("on"), e.cb_out)
        c.pack(0, e.cb_out, at=0)
        c.pack_accumulate(0)
        return True

    units(body)


def cb_3_pack_past_the_reserve():
    def body(c, e):
        e.cb_out.reserve(1)
        c.pack(0, e.cb_out, at=1)
        return True

    units(body)


def cb_3_matmul_past_the_buffer():
    def body(c, e):
        product = c.matmul_init(e.cb_in, e.cb_in, ct=2)
        c.matmul(product, 0, 1, 0)

    units(body)


def cb_2_push_in_a_branch_the_consumer_lacks():
    def reader(k, e):
        for i in k.range(e.count):
            slot = e.cb_in.reserve(1)
            k.read(e.x.page(e.start + i), slot)
            k.reads_done()
            with k.when(e.p.runtime("flag").ne(0)):
                e.cb_in.push(1)

    build(reader=reader)


def branched(flag_of_writer="flag"):
    """A copy program whose kernels run their loops only where runtime value `flag`
    is not 0, the writer on `flag_of_writer`; otherwise nothing is pushed."""

    def under(name, kernel):
        def body(k, e):
            with k.when(e.p.runtime(name).ne(0)):
                kernel(k, e)
            with k.otherwise():
                k.let("idle", e.count)

        return body

    def writer(k, e):
        for i in k.range(e.count):
            e.cb_out.wait(1)
            k.write(e.cb_out, e.out.page(e.start + i))
            k.flushed()
            e.cb_out.pop(1)

    def writer_then_barrier(k, e):
        under(flag_of_writer, writer)(k, e)
        k.writes_done()

    def compute(c, e):
        c.startup(e.cb_in, e.cb_in, e.cb_out)
        c.init("copy", e.cb_in)
        with c.when(e.p.runtime("flag").ne(0)):
            for _ in c.range(e.count):
                e.cb_in.wait(1)
                with c.registers():
                    c.copy(e.cb_in, 0, 0)
                    e.cb_out.reserve(1)
                    c.pack(0, e.cb_out)
                    e.cb_out.push(1)
                e.cb_in.pop(1)

    return build(
        reader=under("flag", good_reader), compute=compute, writer=writer_then_barrier
    )


def cb_2_branches_on_different_values():
    branched(flag_of_writer="other")


def sized(pages=None, of=None, cores=None, slot=0, reads=None):
    """A copy program on a layout: every working core moves `per` tiles, `chunk` at
    a time, through an input buffer of `pages(block)` pages, where `block` is a
    sized value of `of(p, chunk)`. `cores(p)`: the layout's core count; `slot`: the
    register slot (or "block"); `reads(per, n)`: the tiles read per push."""

    @km.program("seed_sized")
    def program(p, x: km.TensorRef, out: km.Out):
        per = p.runtime("per")
        chunk = p.runtime("chunk", choices=range(1, 9))
        p.layout(cores=cores(p) if cores else p.runtime("used"))
        first = p.define("first", p.core * per)
        block = p.sized("block", of=of(p, chunk) if of else chunk, limit=8)
        cb_in = p.cb(0, like=x, pages=pages(block, chunk) if pages else 2 * block)
        cb_out = p.cb(16, like=out, pages=2)
        at = block if slot == "block" else slot

        @p.dm0
        def reader(k):
            for done, n in k.blocks(per, chunk):
                held = cb_in.reserve(block)
                for i in k.range(reads(per, n) if reads else n):
                    k.read(x.page(first + done + i), held[i])
                k.reads_done()
                cb_in.push(block)

        @p.compute
        def compute(c):
            c.startup(cb_in, cb_in, cb_out)
            c.init("copy", cb_in)
            for _, n in c.blocks(per, chunk):
                cb_in.wait(block)
                for i in c.range(n):
                    with c.registers():
                        c.copy(cb_in, i, at)
                        cb_out.reserve(1)
                        c.pack(at, cb_out)
                        cb_out.push(1)
                cb_in.pop(block)

        @p.dm1
        def writer(k):
            for i in k.range(per):
                cb_out.wait(1)
                k.write(cb_out, out.page(first + i))
                k.flushed()
                cb_out.pop(1)
            k.writes_done()

    return program.trace(MACHINE, DTYPES)


def size_1():
    sized(of=lambda p, chunk: chunk + p.core)


def size_1_register_slot():
    sized(slot="block")


def lay_1():
    sized(cores=lambda p: p.core + 1)


def cb_1_sized_buffer_not_a_multiple():
    sized(pages=lambda block, chunk: 2 * block + 1)


def cb_3_beyond_a_sized_reserve():
    sized(reads=lambda per, n: per)


def bind_1_buffer_sized_by_a_runtime_value():
    sized(pages=lambda block, chunk: 2 * chunk)


def by_code(compute, dtype="float32", pages=1, reader=None, inputs=("x",), made=1):
    """A program built by code: dm0 reads `pages` pages of each input per item into
    buffers 0, 1, ... of `pages` pages, `compute(c, p, cbs, count)` turns them into
    `made` pages per item of buffer 16, and dm1 writes the first out."""
    p = km.Program("seed_by_code", MACHINE)
    dtypes = dtype if isinstance(dtype, tuple) else (dtype,) * len(inputs)
    refs = [p.input(name, t) for name, t in zip(inputs, dtypes)]
    out = p.output("out", dtypes[-1])
    start, count = p.split(p.runtime("tiles"))
    cbs = [p.cb(i, like=ref, pages=pages) for i, ref in enumerate(refs)]
    cb_out = p.cb(16, like=out, pages=2)
    k = p.kernel("dm0")
    if reader:
        reader(k, refs[0], cbs[0], start, count)
    else:
        with k.loop(count) as i:
            for ref, cb in zip(refs, cbs):
                slot = k.reserve(cb, pages)
                with k.loop(pages, name="t") as t:
                    k.read(ref.page((start + i) * pages + t), slot[t])
            k.reads_done()
            for cb in cbs:
                k.push(cb, pages)
    compute(p.kernel("compute"), p, [*cbs, cb_out], count)
    w = p.kernel("dm1")
    with w.loop(count) as i:
        w.wait(cb_out, made)
        w.write(cb_out, out.page(start + i))
        w.flushed()
        w.pop(cb_out, made)
    w.writes_done()
    km.verify(p)
    return p


def one_tile(c, cb, cb_out, index=0, waited=1, pages=1):
    """Copies tile `index` of `waited` pages of `cb` and packs it into `cb_out`."""
    c.wait(cb, waited)
    c.acquire()
    c.init("copy", cb)
    c.copy(cb, index, 0)
    c.reserve(cb_out, 1)
    c.pack(0, cb_out)
    c.release()
    c.push(cb_out, 1)
    c.pop(cb, pages)


def reg_1_scope_across_else_branches():
    def compute(c, p, cbs, count):
        cb, cb_out = cbs
        c.startup(cb, cb, cb_out)
        with c.loop(count):
            c.wait(cb, 1)
            c.reserve(cb_out, 1)
            with c.when(p.runtime("a").eq(0)):
                pass
            with c.otherwise():
                c.acquire()
                c.init("copy", cb)
                c.copy(cb, 0, 0)
                c.commit()
            with c.when(p.runtime("b").eq(0)):
                pass
            with c.otherwise():
                c.pack(0, cb_out)
                c.release()
            c.push(cb_out, 1)
            c.pop(cb, 1)

    by_code(compute)


def pack_1_format_set_in_a_loop():
    def compute(c, p, cbs, count):
        cb, cb_out = cbs
        c.startup(cb, cb, p.cb(1, dtype="bfloat16"))
        with c.loop(p.runtime("n")):
            c.pack_format(cb_out)
        with c.loop(count):
            one_tile(c, cb, cb_out)

    by_code(compute)


def init_1_setup_in_a_loop():
    def compute(c, p, cbs, count):
        cb, cb_out = cbs
        c.startup(cb, cb, cb_out)
        c.init("bcast_row", cb)
        with c.loop(p.runtime("n")):
            c.init("copy", cb)
        with c.loop(count):
            c.wait(cb, 1)
            c.acquire()
            c.copy(cb, 0, 0)
            c.reserve(cb_out, 1)
            c.pack(0, cb_out)
            c.release()
            c.push(cb_out, 1)
            c.pop(cb, 1)

    by_code(compute, dtype="bfloat16")


def mm_1_setup_in_a_loop():
    def compute(c, p, cbs, count):
        cb, cb_out = cbs
        c.startup(cb, cb, cb_out)
        c.init("copy", cb)
        with c.loop(p.runtime("n")):
            product = c.matmul_init(cb, cb)
        with c.loop(count):
            c.wait(cb, 1)
            c.acquire()
            c.matmul(product, 0, 0, 0)
            c.reserve(cb_out, 1)
            c.pack(0, cb_out)
            c.release()
            c.push(cb_out, 1)
            c.pop(cb, 1)

    by_code(compute, dtype="bfloat16")


def acc_1_onto_a_page_never_packed():
    def compute(c, p, cbs, count):
        cb, cb_out = cbs
        c.startup(cb, cb, cb_out)
        with c.loop(count):
            c.wait(cb, 2)
            c.acquire()
            c.init("copy", cb)
            c.copy(cb, 0, 0)
            c.copy(cb, 1, 1)
            c.commit()
            c.reserve(cb_out, 2)
            c.pack(0, cb_out, at=0)
            c.pack_accumulate(1, cb_out)
            c.pack(1, cb_out, at=1)
            c.pack_accumulate(0)
            c.release()
            c.push(cb_out, 2)
            c.pop(cb, 2)

    by_code(compute, pages=2, made=2)


def block_reads(read):
    """A compute kernel reading four waited pages of its input with `read(c, p, cb)`
    in one register scope and packing the tile of slot 0."""

    def compute(c, p, cbs, count):
        cb, cb_out = cbs
        c.startup(cb, cb, cb_out, reverse=True)
        with c.loop(count):
            c.wait(cb, 4)
            c.acquire()
            read(c, p, cb)
            c.reserve(cb_out, 1)
            c.pack(0, cb_out)
            c.push(cb_out, 1)
            c.release()
            c.pop(cb, 4)

    return compute


def cb_3_bounded_block_copy():
    def read(c, p, cb):
        c.init("copy", cb)
        c.copy_block(cb, p.runtime("j").bounded(0, 4), 0, 4)

    by_code(block_reads(read), pages=4)


def cb_3_bounded_matmul_block():
    def read(c, p, cb):
        j = p.runtime("j").bounded(0, 4)
        c.matmul(c.matmul_init(cb, cb, ct=2, rt=2, kt=2), j, j, 0)

    by_code(block_reads(read), dtype="bfloat16", pages=4)


def cb_3_span_past_the_last_page():
    def reader(k, x, cb, start, count):
        with k.loop(count) as i:
            slot = k.reserve(cb, 4)
            with k.loop(4, name="t") as t:
                k.read(x.span(start + i, 0, 2 * cb.page_bytes), slot[t])
            k.reads_done()
            k.push(cb, 4)

    def compute(c, p, cbs, count):
        c.startup(cbs[0], cbs[0], cbs[1])
        with c.loop(count):
            one_tile(c, cbs[0], cbs[1], waited=4, pages=4)

    by_code(compute, pages=4, reader=reader)


def cb_3_two_packs_into_one_page():
    def compute(c, p, cbs, count):
        cb, cb_out = cbs
        c.startup(cb, cb, cb_out)
        with c.loop(count):
            c.wait(cb, 1)
            c.acquire()
            c.init("copy", cb)
            c.copy(cb, 0, 0)
            c.copy(cb, 0, 1)
            c.reserve(cb_out, 1)
            c.pack(0, cb_out)
            c.pack(1, cb_out)
            c.release()
            c.push(cb_out, 1)
            c.pop(cb, 1)

    by_code(compute)


def cb_3_copy_past_the_waited_pages():
    def compute(c, p, cbs, count):
        c.startup(cbs[0], cbs[0], cbs[1])
        with c.loop(count):
            one_tile(c, cbs[0], cbs[1], index=1, waited=1, pages=2)

    by_code(compute, pages=2)


def lets_of_one_name(counts, same=False):
    """A copy program whose kernels each name a launch value `n`: a condition, or
    with `counts` a page count from 1 or 2. The reader takes it from value a (`same`:
    b), the others from value b."""
    p = km.Program("seed_lets", MACHINE)
    x, out = p.input("x", "float32"), p.output("out", "float32")
    start, count = p.split(p.runtime("tiles"))
    choices = (1, 2) if counts else None
    a, b = p.runtime("a", choices=choices), p.runtime("b", choices=choices)
    cb, cb_out = p.cb(0, like=x, pages=2), p.cb(16, like=out, pages=2)
    k, c, w = p.kernel("dm0"), p.kernel("compute"), p.kernel("dm1")
    took = {k: k.let("n", b if same else a), c: c.let("n", b), w: w.let("n", b)}
    pages = {kernel: (n if counts else 1) for kernel, n in took.items()}

    def under(kernel):
        n = took[kernel]
        return kernel.when(n.ne(0)) if not counts else contextlib.nullcontext()

    with k.loop(count) as i, under(k):
        slot = k.reserve(cb, pages[k])
        k.read(x.page(start + i), slot)
        k.reads_done()
        k.push(cb, pages[k])
    c.startup(cb, cb, cb_out)
    with c.loop(count), under(c):
        one_tile(c, cb, cb_out, waited=pages[c], pages=pages[c])
    with w.loop(count) as i, under(w):
        w.wait(cb_out, 1)
        w.write(cb_out, out.page(start + i))
        w.flushed()
        w.pop(cb_out, 1)
    w.writes_done()
    km.verify(p)
    return p


def cb_2_conditions_on_lets_of_one_name():
    lets_of_one_name(counts=False)


def cb_2_counts_through_lets_of_one_name():
    lets_of_one_name(counts=True)


def noc_3_barrier_only_in_a_loop():
    def reader(k, x, cb, start, count):
        with k.loop(count) as i:
            slot = k.reserve(cb, 1)
            k.read(x.page(start + i), slot)
            with k.loop(k.program.runtime("waits"), name="w"):
                k.reads_done()
            k.push(cb, 1)

    def compute(c, p, cbs, count):
        c.startup(cbs[0], cbs[0], cbs[1])
        with c.loop(count):
            one_tile(c, cbs[0], cbs[1])

    by_code(compute, reader=reader)


def unpack_1():
    def compute(c, p, cbs, count):
        cb_x, cb_y, cb_out = cbs
        c.startup(cb_x, cb_x, cb_out)
        with c.loop(count):
            c.wait(cb_x, 1)
            c.pop(cb_x, 1)
            one_tile(c, cb_y, cb_out)

    by_code(compute, dtype=("bfloat16", "float32"), inputs=("x", "y"))


def unpack_1_matmul_in_the_regular_order():
    def compute(c, p, cbs, count):
        cb_x, cb_y, cb_out = cbs
        c.startup(cb_x, cb_y, cb_out)
        product = c.matmul_init(cb_x, cb_y)
        with c.loop(count):
            c.wait(cb_x, 1)
            c.wait(cb_y, 1)
            c.acquire()
            c.matmul(product, 0, 0, 0)
            c.reserve(cb_out, 1)
            c.pack(0, cb_out)
            c.release()
            c.push(cb_out, 1)
            c.pop(cb_x, 1)
            c.pop(cb_y, 1)

    by_code(compute, dtype=("bfloat16", "float32"), inputs=("x", "y"))


def copy_at(index):
    """block_reads() copying one tile at page `index(c, p)` of four waited."""

    def read(c, p, cb):
        c.init("copy", cb)
        c.copy(cb, index(c, p), 0)

    return block_reads(read)


def cb_3_bounded_index_minus_one():
    by_code(copy_at(lambda c, p: p.runtime("j").bounded(0, 4) - 1), pages=4)


def cb_3_negative_low_bound():
    by_code(copy_at(lambda c, p: p.runtime("j").bounded(-3, 1)), pages=4)


def cb_3_loop_variable_minus_one():
    def read(c, p, cb):
        c.init("copy", cb)
        with c.loop(4, name="t") as t:
            c.copy(cb, t - 1, 0)

    by_code(block_reads(read), pages=4)


def declared_block(extent, limit=8):
    """A copy of `extent` tiles from an index declared inside a buffer of `block`
    pages, `block` a sized value up to `limit`."""
    p = km.Program("seed_declared_block", MACHINE)
    x, out = p.input("x", "float32"), p.output("out", "float32")
    start, count = p.split(p.runtime("tiles"))
    block = p.sized("block", of=p.runtime("n"), limit=limit)
    cb, cb_out = p.cb(0, like=x, pages=block), p.cb(16, like=out, pages=2)
    k = p.kernel("dm0")
    with k.loop(count) as i:
        slot = k.reserve(cb, block)
        k.read(x.page(start + i), slot)
        k.reads_done()
        k.push(cb, block)
    c = p.kernel("compute")
    c.startup(cb, cb, cb_out)
    with c.loop(count):
        c.wait(cb, block)
        c.acquire()
        c.init("copy", cb)
        c.copy_block(cb, p.runtime("j").bounded(0, block - extent + 1), 0, extent)
        c.reserve(cb_out, 1)
        c.pack(0, cb_out)
        c.release()
        c.push(cb_out, 1)
        c.pop(cb, block)
    w = p.kernel("dm1")
    with w.loop(count) as i:
        w.wait(cb_out, 1)
        w.write(cb_out, out.page(start + i))
        w.flushed()
        w.pop(cb_out, 1)
    w.writes_done()
    km.verify(p)


def cb_3_block_larger_than_a_sized_buffer():
    declared_block(4)


def two_pages_out(body):
    """A compute kernel copying two waited pages to slots 0, 1 and handing them to
    `body(c, p, cb_out)` inside the register scope after a reserve of two pages."""

    def compute(c, p, cbs, count):
        cb, cb_out = cbs
        c.startup(cb, cb, cb_out)
        with c.loop(count):
            c.wait(cb, 2)
            c.acquire()
            c.init("copy", cb)
            c.copy(cb, 0, 0)
            c.copy(cb, 1, 1)
            c.commit()
            c.reserve(cb_out, 2)
            body(c, p, cb_out)
            c.release()
            c.pop(cb, 2)

    return compute


def cb_3_pack_after_a_push():
    def body(c, p, cb_out):
        c.pack(0, cb_out, at=0)
        c.push(cb_out, 1)
        c.pack(1, cb_out, at=1)
        c.push(cb_out, 1)

    by_code(two_pages_out(body), pages=2, made=2)


def acc_1_after_a_push():
    def body(c, p, cb_out):
        c.pack(0, cb_out, at=0)
        c.push(cb_out, 1)
        c.pack_accumulate(1, cb_out)
        c.pack(1, cb_out, at=0)
        c.pack_accumulate(0)
        c.push(cb_out, 1)

    by_code(two_pages_out(body), pages=2, made=2)


def acc_1_page_of_a_variable_set_since():
    def body(c, p, cb_out):
        v = c.var("v", 0)
        c.pack(0, cb_out, at=v.bounded(0, 2))
        c.set(v, 1)
        c.pack_accumulate(1, cb_out)
        c.pack(1, cb_out, at=v.bounded(0, 2))
        c.pack_accumulate(0)
        c.push(cb_out, 2)

    by_code(two_pages_out(body), pages=2, made=2)


def ctl_1_ends_on_another_comparison():
    km.verify(rows_program(MACHINE, cut="exit"))


def ctl_1_word_put_after_the_exit():
    km.verify(rows_program(MACHINE, cut="put"))


def ctl_1_exit_outside_an_endless_loop():
    def compute(c, p, cbs, count):
        c.startup(cbs[0], cbs[0], cbs[1])
        with c.loop(count):
            one_tile(c, cbs[0], cbs[1])
            c.exit_when(count < 2)

    by_code(compute)


def signalling(flush=False, wait=False, op="inc", size=None, done=True, fetch=False):
    """A by_code() reader that sends each page (`size` bytes) to core 0 and signals
    core 0's semaphore by `op`: after the flush (`flush`); `done`: atomics_done at the
    end; `wait`: then waits on its own one; `fetch`: then fetches from core 0."""

    def reader(k, x, cb, start, count):
        p = k.program
        ready, done_ = p.semaphore(0, name="ready"), p.semaphore(0, name="done")
        first = p.core_ref(0)
        with k.loop(count) as i:
            slot = k.reserve(cb, 1)
            k.read(x.page(start + i), slot)
            k.reads_done()
            k.send(slot, to=first, size=size or cb.page_bytes)
            if flush:
                k.flushed()
            if op == "inc":
                k.sem_inc(ready, at=first)
            else:
                k.fetch_add(ready, at=first)
            if not flush:
                k.flushed()
            k.push(cb, 1)
        if fetch:
            k.fetch_add(ready, at=first)
        k.writes_done()
        if done:
            k.atomics_done()
        if wait:
            k.sem_wait_min(done_, 1)

    def compute(c, p, cbs, count):
        c.startup(cbs[0], cbs[0], cbs[1])
        with c.loop(count):
            one_tile(c, cbs[0], cbs[1])

    return by_code(compute, reader=reader)


def noc_4():
    signalling()


def noc_4_fetch_before_the_flush():
    signalling(op="fetch_add")


def noc_3_increments_not_waited_for():
    signalling(flush=True, done=False)


def cb_3_send_past_the_reserve():
    signalling(flush=True, size=10_000_000)


def sem_3_fetch_with_an_increment_in_flight():
    signalling(flush=True, fetch=True)


def sem_3_fetch_while_the_writer_increments():
    km.verify(rows_program(MACHINE, cut="inc"))


def sem_2():
    signalling(flush=True, wait=True)


def waiting(signal):
    """A by_code() reader that waits for semaphore `done` to reach 2 after `signal(k,
    w, done)` added the raises (k: the reader, w: the writer)."""
    p = km.Program("seed_waiting", MACHINE)
    x, out = p.input("x", "float32"), p.output("out", "float32")
    done = p.semaphore(0, name="done")
    cb_x, cb_out = p.cb(0, like=x), p.cb(16, like=out)
    k, w = p.kernel("dm0"), p.kernel("dm1")
    signal(k, w, done)
    k.sem_wait_min(done, 2)
    slot = k.reserve(cb_x, 1)
    k.read(x.page(0), slot)
    k.reads_done()
    k.push(cb_x, 1)
    c = p.kernel("compute")
    c.startup(cb_x, cb_x, cb_out)
    one_tile(c, cb_x, cb_out)
    w.wait(cb_out, 1)
    w.write(cb_out, out.page(0))
    w.flushed()
    w.pop(cb_out, 1)
    w.writes_done()
    km.verify(p)


def sem_2_own_set():
    waiting(lambda k, w, done: k.sem_set(done, 2))


def sem_2_raised_less_than_waited_for():
    def signal(k, w, done):
        w.sem_inc(done)
        w.atomics_done()

    waiting(signal)


def sem_2_raised_under_a_condition():
    def signal(k, w, done):
        with w.when(w.program.runtime("go").ne(0)), w.loop(2):
            w.sem_inc(done)
        w.atomics_done()

    waiting(signal)


def ctl_1_endless_loop_in_a_loop():
    def compute(c, p, cbs, count):
        c.startup(cbs[0], cbs[0], cbs[1])
        words = p.words(24, name="w")
        with c.loop(count):
            one_tile(c, cbs[0], cbs[1])
            with c.forever():
                c.exit_when(c.take(words, "v") >= count)

    by_code(compute)


def lay_1_core_that_may_not_run():
    def reader(k, x, cb, start, count):
        p = k.program
        k.sem_inc(p.semaphore(0, name="ready"), at=p.core_ref(1))
        k.atomics_done()
        with k.loop(count) as i:
            slot = k.reserve(cb, 1)
            k.read(x.page(start + i), slot)
            k.reads_done()
            k.push(cb, 1)

    def compute(c, p, cbs, count):
        c.startup(cbs[0], cbs[0], cbs[1])
        with c.loop(count):
            one_tile(c, cbs[0], cbs[1])

    by_code(compute, reader=reader)


def sem_1_signal_to_a_core_without_it():
    p = km.Program("seed_sem_target", MACHINE)
    p.layout(cores=40)
    row = p.cores.rows(2)
    sem = p.semaphore(0, cores=row, name="ready")
    k = p.kernel("dm0", cores=row)
    k.sem_inc(sem, at=p.core_ref(1))
    k.atomics_done()
    km.verify(p)


def ctl_1_word_changed_before_the_exit():
    km.verify(rows_program(MACHINE, cut="change"))


def ctl_1_ends_on_another_fetched_word():
    km.verify(rows_program(MACHINE, cut="fetch"))


def cb_2_two_streams_on_one_comparison():
    """The reader pushes x once per word of its stream (r += 1); the compute drains
    that stream, then pops x per word of the writer's own (s += 2)."""
    p = km.Program("seed_streams", MACHINE)
    x, out = p.input("x", "float32"), p.output("out", "float32")
    rows = p.runtime("rows")
    cb_x, cb_out = p.cb(0, like=x), p.cb(16, like=out)
    w1, w2 = p.words(24, name="w1"), p.words(25, name="w2")
    k = p.kernel("dm0")
    r = k.var("r", 0)
    with k.forever():
        k.put(w1, r)
        k.exit_when(r >= rows)
        slot = k.reserve(cb_x, 1)
        k.read(x.page(r), slot)
        k.reads_done()
        k.push(cb_x, 1)
        k.set(r, r + 1)
    c = p.kernel("compute", unpack_fp32=[cb_x])
    c.startup(cb_x, None, cb_out)
    c.init("copy", cb_x)
    with c.forever():
        c.exit_when(c.take(w1, "a") >= rows)
    with c.forever():
        c.exit_when(c.take(w2, "b") >= rows)
        c.wait(cb_x, 1)
        c.acquire()
        c.copy(cb_x, 0, 0)
        c.reserve(cb_out, 1)
        c.pack(0, cb_out)
        c.release()
        c.push(cb_out, 1)
        c.pop(cb_x, 1)
    w = p.kernel("dm1")
    s = w.var("s", 0)
    with w.forever():
        w.put(w2, s)
        w.exit_when(s >= rows)
        w.wait(cb_out, 1)
        w.write(cb_out, out.page(s))
        w.flushed()
        w.pop(cb_out, 1)
        w.set(s, s + 2)
    w.writes_done()
    km.verify(p)


def cb_5_page_never_packed():
    def compute(c, p, cbs, count):
        kept = p.cb(6, dtype="float32", pages=2, name="kept")
        c.startup(cbs[0], cbs[0], cbs[1])
        with c.loop(count):
            c.wait(cbs[0], 1)
            c.init("copy", cbs[0])
            c.acquire()
            c.copy(cbs[0], 0, 0)
            c.commit()
            c.reserve(kept, 2)
            c.pack(0, kept)
            c.release()
            c.push(kept, 2)
            c.wait(kept, 2)
            c.acquire()
            c.init("copy", kept)
            c.copy(kept, 1, 0)
            c.commit()
            c.reserve(cbs[1], 1)
            c.pack(0, cbs[1])
            c.release()
            c.push(cbs[1], 1)
            c.pop(kept, 2)
            c.pop(cbs[0], 1)

    by_code(compute)


def lane(cut):
    """km.verify of the lane program with bug `cut` (tests/km_lanes.py)."""
    km.verify(lane_program(cut))


def kern_1_variants_overlap():
    lane("overlap")


def kern_1_row_without_a_variant():
    lane("gap")


def mc_1_rectangle_holds_the_sender():
    lane("holds_sender")


def mc_1_rectangle_past_the_grid():
    lane("off_grid")


def mc_1_flag_set_outside_the_lane():
    lane("touch")


def mc_1_count_from_the_cores_place():
    lane("per_core")


def mc_1_count_changed_after_declaration():
    lane("changed")


def mc_1_opened_after_a_send():
    lane("late_open")


def mc_1_two_sending_kernels():
    lane("two_senders")


def noc_4_flag_before_the_flush():
    lane("unflushed")


def noc_5_ready_outside_a_reserve():
    lane("unreserved")


def noc_5_receiver_reads_into_the_slot():
    lane("reads_slot")


def noc_5_receiver_reserves_under_another_condition():
    lane("branch")


def lay_1_lane_with_a_layout():
    lane("layout")


def lay_1_relative_core_past_the_grid():
    lane("ref_off_grid")


def joint_block(table):
    """A by_code() matmul of an sh x sw block, (sh, sw) one row of `table`."""

    def compute(c, p, cbs, count):
        sh, sw = p.choose(("sh", "sw"), table)
        c.startup(cbs[0], cbs[1], cbs[2], reverse=True)
        product = c.matmul_init(cbs[0], cbs[1], ct=sw, rt=sh, kt=1)
        with c.loop(count):
            c.wait(cbs[0], 4)
            c.wait(cbs[1], 4)
            c.acquire()
            c.matmul(product, 0, 0, 0)
            c.commit()
            c.reserve(cbs[2], 1)
            c.pack(0, cbs[2])
            c.release()
            c.push(cbs[2], 1)
            c.pop(cbs[0], 4)
            c.pop(cbs[1], 4)

    by_code(compute, dtype="bfloat16", inputs=("a", "b"), pages=4)


def dst_1_joint_choice_past_the_registers():
    joint_block([(1, 4), (2, 4)])


def cb_2_reader_on_one_row():
    p = km.Program("seed_one_row", MACHINE)
    x, out = p.input("x", "float32"), p.output("out", "float32")
    n = p.runtime("n")
    cb_x, cb_out = p.cb(0, like=x), p.cb(16, like=out)
    k = p.kernel("dm0", cores=p.cores.rows(0))
    with k.loop(n) as i:
        slot = k.reserve(cb_x, 1)
        k.read(x.page(i), slot)
        k.reads_done()
        k.push(cb_x, 1)
    c = p.kernel("compute", unpack_fp32=[cb_x])
    c.startup(cb_x, None, cb_out)
    with c.loop(n):
        one_tile(c, cb_x, cb_out)
    w = p.kernel("dm1")
    with w.loop(n) as i:
        w.wait(cb_out, 1)
        w.write(cb_out, out.page(p.core * n + i))
        w.flushed()
        w.pop(cb_out, 1)
    w.writes_done()
    km.verify(p)


def cb_2_branches_with_different_counts():
    def compute(c, p, cbs, count):
        c.startup(cbs[0], cbs[0], cbs[1])
        with c.loop(count):
            with c.when(p.runtime("two").ne(0)):
                one_tile(c, cbs[0], cbs[1])
                one_tile(c, cbs[0], cbs[1])
            with c.otherwise():
                one_tile(c, cbs[0], cbs[1])

    by_code(compute)


def noc_5_two_sends_on_one_reserve():
    lane("double_send")


def noc_5_two_receives_on_one_reserve():
    lane("double_receive")


def noc_5_reserve_filled_after_its_send():
    lane("send_before_read")


def noc_5_receivers_out_of_step():
    lane("misaligned")


def loop_over_a_loop_variable(name, same=False):
    """A copy whose reader loops m x n (`same`: like the others) while the compute and
    the writer loop over their outer variable (named `name`) inside a loop over m."""
    p = km.Program("seed_keys", MACHINE)
    x, out = p.input("x", "float32"), p.output("out", "float32")
    n, m = p.runtime("n"), p.runtime("m")
    cb_x, cb_out = p.cb(0, like=x), p.cb(16, like=out)
    k = p.kernel("dm0")
    with (
        k.loop(m, name=name if same else "j") as v,
        k.loop(v if same else n, name="i") as i,
    ):
        slot = k.reserve(cb_x, 1)
        k.read(x.page(i), slot)
        k.reads_done()
        k.push(cb_x, 1)
    c = p.kernel("compute", unpack_fp32=[cb_x])
    c.startup(cb_x, None, cb_out)
    with c.loop(m, name=name) as v, c.loop(v, name="i"):
        one_tile(c, cb_x, cb_out)
    w = p.kernel("dm1")
    with w.loop(m, name=name) as v, w.loop(v, name="i") as i:
        w.wait(cb_out, 1)
        w.write(cb_out, out.page(i))
        w.flushed()
        w.pop(cb_out, 1)
    w.writes_done()
    km.verify(p)


def cb_2_loop_variable_named_like_a_value():
    loop_over_a_loop_variable("n")


def copy_of_sub_pages(rounded):
    """A copy of `sub` (sh * sw, exact or rounded up to 4) pages from a wait of
    `block` (exact bm * bn, bm a multiple of sh, bn of sw)."""
    p = km.Program("seed_sub", MACHINE)
    x, out = p.input("x", "float32"), p.output("out", "float32")
    sh, sw = p.runtime("sh", choices=[1, 2]), p.runtime("sw", choices=[1, 2])
    bm, bn = p.runtime("bm", multiple_of="sh"), p.runtime("bn", multiple_of="sw")
    block = p.sized("block", of=bm * bn, exact=True, limit=16)
    if rounded:
        sub = p.sized("sub", of=sh * sw, classes=(4,))
    else:
        sub = p.sized("sub", of=sh * sw, exact=True, limit=4)
    cb_x, cb_out = p.cb(0, like=x, pages=block), p.cb(16, like=out)
    n = p.runtime("n")
    k = p.kernel("dm0")
    with k.loop(n) as i:
        slot = k.reserve(cb_x, block)
        k.read(x.page(i), slot[0])
        k.reads_done()
        k.push(cb_x, block)
    c = p.kernel("compute", unpack_fp32=[cb_x])
    c.startup(cb_x, None, cb_out)
    c.init("copy", cb_x)
    with c.loop(n):
        c.wait(cb_x, block)
        c.acquire()
        c.copy_block(cb_x, 0, 0, sub)
        c.reserve(cb_out, 1)
        c.pack(0, cb_out)
        c.release()
        c.push(cb_out, 1)
        c.pop(cb_x, block)
    w = p.kernel("dm1")
    with w.loop(n) as i:
        w.wait(cb_out, 1)
        w.write(cb_out, out.page(i))
        w.flushed()
        w.pop(cb_out, 1)
    w.writes_done()
    km.verify(p)


def cb_3_rounded_sized_extent():
    copy_of_sub_pages(rounded=True)


def table(cut):
    """km.verify of the job-list program with bug `cut` (tests/km_tables.py)."""
    km.verify(table_program(cut))


def arg_2_row_past_the_table():
    table("past")


def arg_2_row_below_zero():
    table("below")


def arg_2_row_of_another_tables_loop():
    table("other_table")


def arg_2_row_variable_of_a_runtime_loop():
    table("same_name")


def arg_2_count_without_a_most():
    table("no_most")


def arg_2_page_index_from_an_undeclared_column():
    table("undeclared")


def arg_2_page_index_past_the_count():
    table("past_the_count")


def arg_2_page_index_declared_for_another_tensor():
    table("other_tensor")


def bind_1_page_count_from_a_table():
    table("pages")


def cb_2_counts_from_other_table_words():
    table("other_column")


SEEDS = {
    "SIZE-1": [size_1, size_1_register_slot],
    "LAY-1": [
        lay_1,
        lay_1_core_that_may_not_run,
        lay_1_lane_with_a_layout,
        lay_1_relative_core_past_the_grid,
    ],
    "KERN-1": [kern_1_variants_overlap, kern_1_row_without_a_variant],
    "ARG-2": [
        arg_2_row_past_the_table,
        arg_2_row_below_zero,
        arg_2_row_of_another_tables_loop,
        arg_2_row_variable_of_a_runtime_loop,
        arg_2_count_without_a_most,
        arg_2_page_index_from_an_undeclared_column,
        arg_2_page_index_past_the_count,
        arg_2_page_index_declared_for_another_tensor,
    ],
    "MC-1": [
        mc_1_rectangle_holds_the_sender,
        mc_1_rectangle_past_the_grid,
        mc_1_flag_set_outside_the_lane,
        mc_1_count_from_the_cores_place,
        mc_1_count_changed_after_declaration,
        mc_1_opened_after_a_send,
        mc_1_two_sending_kernels,
    ],
    "NOC-5": [
        noc_5_ready_outside_a_reserve,
        noc_5_receiver_reads_into_the_slot,
        noc_5_receiver_reserves_under_another_condition,
        noc_5_two_sends_on_one_reserve,
        noc_5_two_receives_on_one_reserve,
        noc_5_reserve_filled_after_its_send,
        noc_5_receivers_out_of_step,
    ],
    "INIT-1": [
        init_1,
        init_1_another_ops_setup,
        init_1_another_way_of_unpacking,
        init_1_setup_under_a_condition,
        init_1_setup_in_a_loop,
        init_1_generated_function,
        init_1_tiles_under_another_setup,
    ],
    "RED-1": [red_1, red_1_two_dims_in_one_scope, red_1_masks_left_in_effect],
    "MM-1": [mm_1, mm_1_setup_in_a_loop],
    "PACK-1": [pack_1, pack_1_format_set_in_a_loop],
    "ACC-1": [
        acc_1,
        acc_1_left_on,
        acc_1_runtime_switch,
        acc_1_onto_a_page_never_packed,
        acc_1_after_a_push,
        acc_1_page_of_a_variable_set_since,
    ],
    "UNPACK-1": [unpack_1, unpack_1_matmul_in_the_regular_order],
    "BIND-1": [
        bind_1_page_count_from_a_table,
        bind_1,
        bind_1_python_branch,
        bind_1_unbounded_count,
        bind_1_buffer_sized_by_a_runtime_value,
    ],
    "BIND-2": [bind_2],
    "DATA-1": [data_1_page, data_1_loop_bound],
    "ARG-1": [arg_1],
    "CB-4": [cb_4],
    "CB-1": [cb_1, cb_1_choice, cb_1_sized_buffer_not_a_multiple],
    "CB-3": [
        cb_3,
        cb_3_beyond_a_counted_reserve,
        cb_3_load_past_the_reserve,
        cb_3_congruent_slot_overruns,
        cb_3_pack_past_the_reserve,
        cb_3_matmul_past_the_buffer,
        cb_3_beyond_a_sized_reserve,
        cb_3_bounded_block_copy,
        cb_3_bounded_matmul_block,
        cb_3_span_past_the_last_page,
        cb_3_two_packs_into_one_page,
        cb_3_copy_past_the_waited_pages,
        cb_3_bounded_index_minus_one,
        cb_3_negative_low_bound,
        cb_3_loop_variable_minus_one,
        cb_3_block_larger_than_a_sized_buffer,
        cb_3_pack_after_a_push,
        cb_3_send_past_the_reserve,
        cb_3_rounded_sized_extent,
    ],
    "CB-2": [
        cb_2,
        cb_2_counted,
        cb_2_push_in_a_condition,
        cb_2_push_in_a_branch_the_consumer_lacks,
        cb_2_branches_on_different_values,
        cb_2_conditions_on_lets_of_one_name,
        cb_2_counts_through_lets_of_one_name,
        cb_2_two_streams_on_one_comparison,
        cb_2_branches_with_different_counts,
        cb_2_reader_on_one_row,
        cb_2_loop_variable_named_like_a_value,
        cb_2_counts_from_other_table_words,
    ],
    "LOOP-1": [loop_1],
    "CB-5": [cb_5, cb_5_page_never_packed],
    "NOC-1": [noc_1],
    "NOC-2": [noc_2, noc_2_value_offset],
    "NOC-3": [
        noc_3,
        noc_3_load_before_the_read_lands,
        noc_3_read_into_a_page_being_written,
        noc_3_write_before_the_read_lands,
        noc_3_barrier_only_in_a_loop,
        noc_3_increments_not_waited_for,
    ],
    "NOC-4": [noc_4, noc_4_fetch_before_the_flush, noc_4_flag_before_the_flush],
    "SEM-1": [sem_1, sem_1_signal_to_a_core_without_it],
    "SEM-2": [
        sem_2,
        sem_2_own_set,
        sem_2_raised_less_than_waited_for,
        sem_2_raised_under_a_condition,
    ],
    "SEM-3": [
        sem_3_fetch_with_an_increment_in_flight,
        sem_3_fetch_while_the_writer_increments,
    ],
    "CTL-1": [
        ctl_1_ends_on_another_comparison,
        ctl_1_word_put_after_the_exit,
        ctl_1_exit_outside_an_endless_loop,
        ctl_1_word_changed_before_the_exit,
        ctl_1_ends_on_another_fetched_word,
        ctl_1_endless_loop_in_a_loop,
    ],
    "SFPU-1": [sfpu_1],
    "SFPU-2": [
        sfpu_2,
        sfpu_2_shift_of_a_product,
        sfpu_2_table_lookup,
        sfpu_2_constant_from_a_sum,
    ],
    "DST-1": [
        dst_1,
        dst_1_value,
        dst_1_block,
        dst_1_function_operand,
        dst_1_joint_choice_past_the_registers,
    ],
    "REG-1": [reg_1, reg_1_commit_in_a_loop, reg_1_scope_across_else_branches],
    "REG-2": [reg_2, reg_2_statistics_slot_not_filled],
    "INL-1": [inl_1],
    "L1-1": [l1_1],
}


@pytest.mark.parametrize(
    ("rule", "seed"),
    [(rule, seed) for rule, seeds in SEEDS.items() for seed in seeds],
    ids=lambda v: v if isinstance(v, str) else v.__name__,
)
def test_seeded_bug_is_rejected(rule, seed):
    with pytest.raises(km.VerifyError) as error:
        seed()
    assert error.value.rule == rule


def test_programs_the_soundness_seeds_are_cut_from_verify():
    """The same programs written correctly: a scope inside one else branch, setups
    before a loop, reads inside the waited pages, lets holding equal values."""

    def scope_in_one_else(c, p, cbs, count):
        cb, cb_out = cbs
        c.startup(cb, cb, cb_out)
        with c.loop(count):
            c.wait(cb, 1)
            c.reserve(cb_out, 1)
            for branch in (c.when(p.runtime("a").eq(0)), c.otherwise()):
                with branch:
                    c.acquire()
                    c.init("copy", cb)
                    c.copy(cb, 0, 0)
                    c.pack(0, cb_out)
                    c.release()
            c.push(cb_out, 1)
            c.pop(cb, 1)

    def block_copy(c, p, cb):
        c.init("copy", cb)
        c.copy_block(cb, p.runtime("j").bounded(0, 1), 0, 4)

    def block_matmul(c, p, cb):
        j = p.runtime("j").bounded(0, 2)
        c.matmul(c.matmul_init(cb, cb, ct=2, rt=2, kt=1), j, j, 0)

    def reversed_matmul(c, p, cbs, count):
        cb_x, cb_y, cb_out = cbs
        c.startup(cb_x, cb_y, cb_out, reverse=True)
        product = c.matmul_init(cb_x, cb_y)
        with c.loop(count):
            c.wait(cb_x, 1)
            c.wait(cb_y, 1)
            c.acquire()
            c.matmul(product, 0, 0, 0)
            c.reserve(cb_out, 1)
            c.pack(0, cb_out)
            c.release()
            c.push(cb_out, 1)
            c.pop(cb_x, 1)
            c.pop(cb_y, 1)

    for _ in range(3):
        by_code(scope_in_one_else)
    by_code(block_reads(block_copy), pages=4)
    by_code(block_reads(block_matmul), dtype="bfloat16", pages=4)
    by_code(reversed_matmul, dtype=("bfloat16", "float32"), inputs=("x", "y"))
    lets_of_one_name(counts=False, same=True)
    lets_of_one_name(counts=True, same=True)


def test_programs_the_control_seeds_are_cut_from_verify(monkeypatch):
    """Control words with a counter, and a send flushed before its signal."""
    import kohakutt.dsl.metal.bridge as bridge

    monkeypatch.setattr(bridge, "_compute_options", lambda *_: {})
    program = rows_program(MACHINE)
    km.verify(program)
    signalling(flush=True)
    signalling(flush=True, op="fetch_add")

    def signal(k, w, done):
        with w.loop(2):
            w.sem_inc(done)
        w.atomics_done()

    waiting(signal)
    reader, compute = km.source(program, "dm0"), km.source(program, "compute")
    assert "for (uint32_t it = 0;; ++it) {" in reader
    assert "get_noc_addr(core0_x, core0_y, get_semaphore(sem_next))" in reader
    assert "MEM_NOC_ATOMIC_RET_VAL_ADDR" in reader
    assert "const uint32_t row = read_tile_value(cb_to_compute, 0, 0);" in compute
    spec = json.loads(spec_of(program))
    assert {"noc": [0, 0, 0]} in spec["kernels"][0]["common_args"]
    words = [cb for cb in spec["cbs"] if cb["index"] in (24, 25)]
    assert all(cb["page_bytes"] == 16 and cb["dtype"] == "uint32" for cb in words)


def spec_of(program):
    """The description of `program` on 16 cores with no sources written."""
    from kohakutt.dsl.metal.bridge import spec

    roles = {role: f"{role}.cpp" for role in program.kernels}
    return spec(program, roles, 16, None)


def test_programs_the_second_soundness_seeds_are_cut_from_verify():
    """Reads of two pages at slot[t] with t stepping by 2, pages packed in one loop
    and accumulated in another, packs before and after a push, an index t + 1."""

    def stepped(k, x, cb, start, count):
        with k.loop(count) as i:
            slot = k.reserve(cb, 4)
            with k.loop(4, step=2, name="t") as t:
                k.read(x.span(start + i, 0, 2 * cb.page_bytes), slot[t])
            k.reads_done()
            k.push(cb, 4)

    def first_page(c, p, cbs, count):
        c.startup(cbs[0], cbs[0], cbs[1])
        with c.loop(count):
            one_tile(c, cbs[0], cbs[1], waited=4, pages=4)

    def loops(c, p, cb_out):
        with c.loop(2, name="t") as t:
            c.pack(t, cb_out, at=t)
        c.pack_accumulate(1, cb_out)
        with c.loop(2, name="t") as t:
            c.pack(t, cb_out, at=t)
        c.pack_accumulate(0)
        c.push(cb_out, 2)

    def around_a_push(c, p, cb_out):
        c.pack(0, cb_out, at=0)
        c.pack(1, cb_out, at=1)
        c.push(cb_out, 1)
        c.pack_accumulate(1, cb_out)
        c.pack(1, cb_out, at=0)
        c.pack_accumulate(0)
        c.push(cb_out, 1)

    def shifted(c, p, cb):
        c.init("copy", cb)
        with c.loop(3, name="t") as t:
            c.copy(cb, t + 1, 0)

    by_code(first_page, pages=4, reader=stepped)
    by_code(two_pages_out(loops), pages=2, made=2)
    by_code(two_pages_out(around_a_push), pages=2, made=2)
    by_code(block_reads(shifted), pages=4)


def test_an_else_branch_is_one_entry_of_every_path_through_it():
    program = branched()
    walked = km_body.walk(program.kernels["dm0"].body)
    whens = [s for s, _ in walked if isinstance(s, km_body.When) and s.orelse]
    paths = [
        [path for _, path in km_body.walk(program.kernels["dm0"].body)]
        for _ in range(2)
    ]
    assert paths[0] == paths[1]
    assert whens[0].else_entry in paths[0][-1]


def test_every_checked_rule_has_a_seed():
    checked = [rule.id for rule in km.RULES if rule.check is not None]
    assert sorted(checked) == sorted(SEEDS)


def test_every_rule_is_checked():
    assert [rule.id for rule in km.RULES if rule.check is None] == []


def test_spaced_vector_unit_code_verifies():
    nop = "TTI_SFPNOP;"
    raw(f"{LOAD} TTI_SFPMAD(1, 2, 9, 5, 0); {nop} TTI_SFPSWAP(0, 5, 2, 1); {STORE}")
    # A plain swap, another register, and a stalled reader need no gap.
    raw(f"{LOAD} TTI_SFPMAD(1, 2, 9, 5, 0); TTI_SFPSWAP(0, 5, 2, 0); {STORE}")
    raw(f"{LOAD} TTI_SFPMAD(1, 2, 9, 5, 0); TTI_SFPSWAP(0, 1, 2, 1); {STORE}")
    raw(f"{LOAD} TTI_SFPMUL(1, 2, 9, 5, 0); TTI_SFPADD(10, 5, 2, 5, 0); {STORE}")
    raw(f"{LOAD} TTI_SFPMAD(1, 2, 9, 5, 0); // TTI_SFPSWAP(0, 5, 2, 1);\n {STORE}")


def test_the_unseeded_program_verifies():
    program = build()
    km.verify(program, l1_budget=program.footprint())


def test_counted_program_verifies():
    """The program the counted seeds are cut from: every choice divides the buffer."""
    program = build(
        reader=counted_reader((1, 2)),
        compute=counted_compute,
        writer=counted_writer,
        pages=4,
    )
    km.verify(program)


def test_programs_the_new_seeds_are_cut_from_verify():
    """A declared range, a congruent destination and a held page written per item."""
    km.verify(build(reader=looked_up_reader()))
    program = build(reader=half_page_reader(like=True))
    assert "+ (src & 63), 1024);" in km.source(program, "dm0")
    km.verify(build(reader=half_page_reader(staged=0, like=True)))
    program = moved()
    assert list(program.kernels) == ["dm0"]
    assert "noc_async_write_page(u, out, x_ptr);" in km.source(program, "dm0")
    moved(barrier="flushed")


def test_programs_the_compute_seeds_are_cut_from_verify():
    """Table ops under their setup, a block inside the registers, accumulation onto a
    packed page, and kernels branching alike on one launch value."""

    def ops(c, e):
        c.init("rsqrt")
        c.op("rsqrt", 0)
        c.init("copy", e.cb_in)
        c.copy(e.cb_in, 0, 1)
        c.init("transpose", e.cb_in)
        c.transpose(e.cb_in, 0, 2)
        c.init("max")
        c.op("max", 0, 1, 0)

    def block(c, e):
        product = c.matmul_init(e.cb_in, e.cb_in, ct=2, rt=2, kt=1)
        c.matmul(product, 0, 0, 0)

    def accumulated(c, e):
        e.cb_out.reserve(1)
        c.pack(0, e.cb_out, at=0)
        c.pack_accumulate(1, e.cb_out)
        c.pack(0, e.cb_out, at=0)
        c.pack_accumulate(0)
        return True

    for body, waited in ((ops, 1), (block, 2), (accumulated, 1)):
        km.verify(units(body, pages=4, waited=waited))
    km.verify(statistics(fill=True))
    km.verify(units(reduced))
    km.verify(
        units(lambda c, e: c.tiles(c.tiles_init("mul", e.cb_in, e.cb_in), 0, 0, 1))
    )
    program = branched()
    assert "} else {" in km.source(program, "dm0")
    km.verify(program)


def test_values_know_what_they_are_multiples_of():
    @km.program("multiples")
    def program(p, x: km.TensorRef, out: km.Out):
        n = p.runtime("n", multiple_of=6)
        block = p.runtime("block", choices=(1, 2, 3))
        nr = p.runtime("nr", multiple_of=block)
        assert n.multiple() == 6 and (n * 4 + 8).multiple() == 8
        assert ((n << 2) & 63).multiple() == 8 and (n % 4).multiple() == 2
        assert ((n & 15) * 16).upper() == 14 * 16 and (n >> 1).multiple() == 1
        assert nr.multiple() == 1 and nr.multiple_of == "block"
        good_program(p, x, out)

    traced = program.trace(MACHINE, DTYPES)
    with pytest.raises(ValueError, match="nr = 7 is not a multiple of block = 2"):
        traced.value_list({"tiles": 8, "n": 12, "block": 2, "nr": 7}, 8)
    with pytest.raises(ValueError, match="n = 7 is not a multiple of 6"):
        traced.value_list({"tiles": 8, "n": 7, "block": 2, "nr": 8}, 8)
    assert traced.value_list({"tiles": 8, "n": 12, "block": 2, "nr": 8}, 8)


def good_program(p, x, out):
    """The copy program of build() on an existing program `p`."""
    start, count = p.split(p.runtime("tiles"))
    e = SimpleNamespace(p=p, x=x, out=out, start=start, count=count)
    e.cb_in, e.cb_out = p.cb(0, like=x, pages=2), p.cb(16, like=out, pages=2)
    p.dm0(lambda k: good_reader(k, e))
    p.compute(lambda c: good_compute(c, e))
    p.dm1(lambda k: good_writer(k, e))


def test_examples_verify():
    for program in (rope(), swiglu()):
        km.verify(program, l1_budget=1 << 20)


def test_declared_constant_is_restored():
    def compute(c, e):
        c.inline("TTI_SFPCONFIG(0, 11, 1);", clobbers=["lreg11"])
        good_compute(c, e)

    text = km.source(build(compute=compute), "compute")
    assert text.rstrip().endswith("MATH((restore_constants()));\n}")


def test_constant_is_restored_after_the_last_vector_unit_work():
    code = "PACK((TTI_SFPCONFIG(0, 11, 1)));"

    def pack_only(c, e):
        c.inline(code, clobbers=["lreg11"], sfpu="pack")
        good_compute(c, e)

    def both_threads(c, e):
        c.inline(code, clobbers=["lreg11"], sfpu="pack")
        c.inline("MATH((TTI_SFPNOP));", sfpu="math")
        good_compute(c, e)

    text = km.source(build(compute=pack_only), "compute")
    assert text.rstrip().endswith("PACK((restore_constants()));\n}")
    assert "STALLWAIT" not in text
    text = km.source(build(compute=both_threads), "compute")
    assert text.rstrip().endswith(
        "PACK((TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::PACK)));\n"
        "    PACK((restore_constants()));\n}"
    )


def test_vector_unit_work_is_declared_on_a_compute_thread():
    def reader(k, e):
        k.inline("// no vector unit here", sfpu="math")
        good_reader(k, e)

    with pytest.raises(km.VerifyError) as error:
        build(reader=reader)
    assert error.value.rule == "SFPU-1"
