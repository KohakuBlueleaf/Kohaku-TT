"""The job-list copy of examples/kernels/dsl_metal/segments_km.py with seeded bugs, for
tests."""

import kohakutt.dsl.metal as km

MACHINE = km.Machine(4, 3)
ROWS, MOST = 4, 64

CUTS = {
    "past": "the reader reads a static row past the table",
    "below": "the reader reads row j - 1",
    "other_table": "the reader reads its jobs at the row of a loop over another table",
    "same_name": "the reader's row variable walks a runtime count under the name j",
    "no_most": "the reader counts with a column that declares no most",
    "pages": "the reader reserves a page count taken from the table",
    "other_column": "the writer counts with another column than the reader",
    "undeclared": "the table declares no page columns",
    "past_the_count": "the reader reads one page past each job",
    "other_tensor": "the writer writes out at pages declared for x",
}


def table_program(cut=None):
    """out pages = x pages by (source, destination, count) jobs. `cut`: CUTS."""
    p = km.Program("tables", MACHINE)
    x, out = p.input("x", "float32"), p.output("out", "float32")
    most = {1: 4 * MOST, 2: MOST} if cut == "other_column" else {2: MOST}
    pages = {0: (x, 2), 1: (out, 2)}
    if cut == "other_tensor":
        pages = {0: (x, 2), 1: (x, 2)}
    jobs = p.table(
        "jobs",
        width=3,
        rows=ROWS,
        most=most,
        pages=None if cut == "undeclared" else pages,
    )
    other = p.table("other", width=1, rows=ROWS)
    cb_x, cb_out = p.cb(0, like=x), p.cb(16, like=out)

    k = p.kernel("dm0")
    over = {"other_table": other.used, "same_name": p.runtime("n")}.get(cut, jobs.used)
    with k.loop(over, name="j") as j:
        row = {"past": ROWS, "below": j - 1}.get(cut, j)
        count = jobs[row, 0] if cut == "no_most" else jobs[row, 2]
        with k.loop(count, name="t") as t:
            pages = jobs[row, 2] if cut == "pages" else 1
            slot = k.reserve(cb_x, pages)
            k.read(
                x.page(jobs[row, 0] + (t + 1 if cut == "past_the_count" else t)), slot
            )
            k.reads_done()
            k.push(cb_x, 1)

    c = p.kernel("compute", unpack_fp32=[cb_x])
    c.startup(cb_x, None, cb_out)
    c.init("copy", cb_x)
    with c.loop(jobs.used, name="j") as j, c.loop(jobs[j, 2], name="t"):
        c.wait(cb_x, 1)
        c.acquire()
        c.copy(cb_x, 0, 0)
        c.reserve(cb_out, 1)
        c.pack(0, cb_out)
        c.release()
        c.push(cb_out, 1)
        c.pop(cb_x, 1)

    w = p.kernel("dm1")
    with w.loop(jobs.used, name="j") as j:
        count = jobs[j, 1] if cut == "other_column" else jobs[j, 2]
        with w.loop(count, name="t") as t:
            w.wait(cb_out, 1)
            # The miscounting writer writes each job's first page only.
            page = jobs[j, 1] if cut == "other_column" else jobs[j, 1] + t
            w.write(cb_out, out.page(page))
            w.flushed()
            w.pop(cb_out, 1)
    w.writes_done()
    return p
