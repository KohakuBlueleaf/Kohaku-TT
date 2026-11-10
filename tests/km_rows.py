"""A copy program driven by control words, for tests: the reader hands each tile row's
index to the compute and the writer, which loop until an index past the rows; rows
are dealt out in turn or taken from a counter on core 0 (`dynamic`)."""

import math

import kohakutt.dsl.metal as km


def rows_program(machine, dtype="float32", cut=None):
    """out = x, tile row by tile row. Runtime values: `rows` (tile rows), `wt` (tiles
    per row), `groups` (cores with work), `dynamic` (1: rows from the counter).
    `cut` seeds a bug: "exit", "put", "change", "fetch", "inc" (see CUTS)."""
    p = km.Program("rows", machine)
    x, out = p.input("x", dtype), p.output("out", dtype)
    rows, wt = p.runtime("rows"), p.runtime("wt")
    groups, dynamic = p.runtime("groups"), p.runtime("dynamic")
    p.layout(cores=groups)
    counter = p.semaphore(0, name="next")
    first = p.core_ref(0)
    cb_x, cb_out = p.cb(0, like=x), p.cb(16, like=out)
    to_compute = p.words(24, name="to_compute")
    to_writer = p.words(25, name="to_writer")

    k = p.kernel("dm0")
    r = k.var("r", p.core)
    with k.forever():
        k.put(to_compute, r)
        if cut != "put":
            k.put(to_writer, r)
        if cut == "change":
            k.set(r, r + groups)
        if cut == "fetch":
            k.exit_when(k.fetch_add(counter, at=first) >= rows)
        else:
            k.exit_when(r >= rows)
        if cut == "put":
            k.put(to_writer, r)
        with k.loop(wt, name="j") as j:
            slot = k.reserve(cb_x, 1)
            k.read(x.page(r * wt + j), slot)
            k.reads_done()
            k.push(cb_x, 1)
        with k.when(dynamic.ne(0)):
            k.set(r, k.fetch_add(counter, at=first) + groups)
        with k.otherwise():
            k.set(r, r + groups)

    c = p.kernel("compute", unpack_fp32=[cb_x] if dtype == "float32" else [])
    c.startup(cb_x, None, cb_out)
    c.init("copy", cb_x)
    with c.forever():
        row = c.take(to_compute, "row")
        c.exit_when(row > rows if cut == "exit" else row >= rows)
        with c.loop(wt, name="j"):
            c.wait(cb_x, 1)
            c.acquire()
            c.copy(cb_x, 0, 0)
            c.reserve(cb_out, 1)
            c.pack(0, cb_out)
            c.release()
            c.push(cb_out, 1)
            c.pop(cb_x, 1)

    w = p.kernel("dm1")
    with w.forever():
        row = w.take(to_writer, "row")
        w.exit_when(row >= rows)
        with w.loop(wt, name="j") as j:
            w.wait(cb_out, 1)
            w.write(cb_out, out.page(row * wt + j))
            w.flushed()
            w.pop(cb_out, 1)
        if cut == "inc":
            w.sem_inc(counter, at=first)
    if cut == "inc":
        w.atomics_done()
    w.writes_done()
    return p


CUTS = {
    "exit": "the compute ends on another comparison",
    "put": "the writer's word is put after the reader's exit",
    "change": "the reader changes the word it put before its exit",
    "fetch": "the reader puts one word and ends on a newly fetched one",
    "inc": "the writer increments the counter the reader fetches from",
}


def runtime(shape, cores, dynamic):
    """The runtime values of rows_program for a tensor of device shape `shape` on
    `cores` cores."""
    rows = math.prod(shape[:-2]) * math.ceil(shape[-2] / 32)
    wt = math.ceil(shape[-1] / 32)
    return {"rows": rows, "wt": wt, "groups": min(rows, cores), "dynamic": dynamic}
