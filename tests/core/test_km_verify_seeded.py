"""Seeded bugs: one minimal failing program per verifier rule of the metal level; the
verifier must reject each with that rule, and accept the program they are cut from."""

from types import SimpleNamespace

import pytest
from km_programs import MACHINE, rope, swiglu

import kohakutt.dsl.metal as km

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


def units(body, before=None, pages=2):
    """build() with a compute kernel whose register scope per tile is `body(c, e)`
    after tile 0 is copied to slot 0 (reserve, pack and push of cb_out included when
    `body` returns None); `before(c, e)` runs once, ahead of the loop."""

    def setup(e):
        e.kept = e.p.cb(24, dtype="float32", pages=1)

    def compute(c, e):
        c.startup(e.cb_in, e.cb_in, e.cb_out)
        if before:
            before(c, e)
        for _ in c.range(e.count):
            e.cb_in.wait(1)
            with c.registers():
                c.init("copy", e.cb_in)
                c.copy(e.cb_in, 0, 0)
                if body(c, e) is None:
                    e.cb_out.reserve(1)
                    c.pack(0, e.cb_out)
                e.cb_out.push(1)
            e.cb_in.pop(1)

    return build(compute=compute, setup=setup, pages=pages)


def dst_1_block():
    def body(c, e):
        product = c.matmul_init(e.cb_in, e.cb_in, ct=4, rt=2, kt=1)
        c.matmul(product, 0, 0, 0)

    units(body, pages=8)


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


SEEDS = {
    "SIZE-1": [size_1, size_1_register_slot],
    "LAY-1": [lay_1],
    "INIT-1": [
        init_1,
        init_1_another_ops_setup,
        init_1_another_way_of_unpacking,
        init_1_setup_under_a_condition,
    ],
    "MM-1": [mm_1],
    "PACK-1": [pack_1],
    "ACC-1": [acc_1, acc_1_left_on, acc_1_runtime_switch],
    "BIND-1": [
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
    ],
    "CB-2": [
        cb_2,
        cb_2_counted,
        cb_2_push_in_a_condition,
        cb_2_push_in_a_branch_the_consumer_lacks,
        cb_2_branches_on_different_values,
    ],
    "LOOP-1": [loop_1],
    "CB-5": [cb_5],
    "NOC-1": [noc_1],
    "NOC-2": [noc_2, noc_2_value_offset],
    "NOC-3": [
        noc_3,
        noc_3_load_before_the_read_lands,
        noc_3_read_into_a_page_being_written,
        noc_3_write_before_the_read_lands,
    ],
    "SEM-1": [sem_1],
    "SFPU-1": [sfpu_1],
    "DST-1": [dst_1, dst_1_value, dst_1_block],
    "REG-1": [reg_1, reg_1_commit_in_a_loop],
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


def test_every_checked_rule_has_a_seed():
    checked = [rule.id for rule in km.RULES if rule.check is not None]
    assert sorted(checked) == sorted(SEEDS)


def test_reserved_rule_slot():
    reserved = [rule.id for rule in km.RULES if rule.check is None]
    assert reserved == ["SFPU-2"]


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

    for body, pages in ((ops, 2), (block, 4), (accumulated, 2)):
        km.verify(units(body, pages=pages))
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
