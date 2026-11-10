"""A copy program whose top row reads each tile and multicasts it down its column (a
lane), for tests: every core then copies the tile to its own output pages."""

import kohakutt.dsl.metal as km
from kohakutt.dsl.metal.binding import as_expr
from kohakutt.dsl.metal.body import Mcast, McastSet, SemOp

MACHINE = km.Machine(4, 3)

CUTS = {
    "overlap": "the receiving variant also runs on the sender's row",
    "gap": "no variant runs on row 1",
    "holds_sender": "the lane's rectangle holds the sender's row",
    "off_grid": "the lane's rectangle runs past the grid",
    "touch": "the receiver sets the lane's flag outside the lane",
    "per_core": "the receiver's count is computed from the core's place",
    "changed": "the receiver's count is a variable set after its declaration",
    "late_open": "the sender opens the lane after its first send",
    "two_senders": "the writer also sends on the lane",
    "unflushed": "the flag is multicast before the data left",
    "unreserved": "the receiver signals ready outside a reserve",
    "reads_slot": "the receiver reads into the pages the lane fills",
    "branch": "the receiver reserves under a condition the sender does not have",
    "layout": "the program has a layout, so a receiver may not run",
    "ref_off_grid": "a relative core reference names a core past the last row",
    "double_send": "the sender sends twice on one reserve, the receivers once",
    "double_receive": "the receivers receive twice on one reserve",
    "send_before_read": "the sender reads into its reserve after sending it",
    "misaligned": "a reserve without a send comes first on the sender, last on the "
    "receivers",
}


def lane_program(cut=None):
    """out[core * n + i] = x[i] for i < n on every core. `cut` seeds a bug (CUTS)."""
    p = km.Program("lanes", MACHINE)
    x, out = p.input("x", "float32"), p.output("out", "float32")
    n = p.runtime("n")
    if cut == "layout":
        p.layout(cores=n)
    cb_x, cb_out = p.cb(0, like=x), p.cb(16, like=out)
    rows = MACHINE.rows
    top, below = p.cores.rows(0), p.cores.rows(1, rows - 1)
    wanted = {
        "holds_sender": p.cores.rows(0, rows - 1),
        "off_grid": p.cores.rows(1, rows),
    }.get(cut, below)
    lane = p.lane(top, wanted)

    s = p.kernel("dm0", cores=top, name="send")
    if cut != "late_open":
        lane.open(s)
    if cut == "misaligned":
        # One more page through the buffer, read here and not multicast.
        slot = s.reserve(cb_x, 1)
        s.read(x.page(0), slot)
        s.reads_done()
        s.push(cb_x, 1)
    with s.loop(n, name="i") as i:
        slot = s.reserve(cb_x, 1)
        if cut != "send_before_read":
            s.read(x.page(i), slot)
            s.reads_done()
        if cut == "unflushed":
            # lane.send without its flush.
            size = as_expr(slot.pages) * slot.cb.page_bytes
            s._lane(SemOp("wait_eq", lane.ready, lane.rect.count, lane=lane))
            s._lane(SemOp("set", lane.ready, as_expr(0), lane=lane))
            s._lane(Mcast(slot[0], lane.rect, size, lane))
            s._lane(McastSet(lane.flag, lane.rect, lane))
        else:
            lane.send(s, slot)
        if cut == "double_send":
            lane.send(s, slot)
        if cut == "send_before_read":
            s.read(x.page(i), slot)
            s.reads_done()
        s.push(cb_x, 1)
    if cut == "late_open":
        lane.open(s)
    s.writes_done()

    recv_rows = {
        "overlap": p.cores.rows(0, rows - 1),
        "gap": p.cores.rows(2, rows - 1),
    }.get(cut, below)
    r = p.kernel("dm0", cores=recv_rows, name="recv")
    count = km.select(p.core < 100, n, n) if cut == "per_core" else n
    if cut == "changed":
        count = r.var("count", n)
        r.set(count, n + 1)
    with r.loop(count, name="i"):

        def receive():
            if cut == "unreserved":
                lane.receive(r)
            slot = r.reserve(cb_x, 1)
            if cut == "reads_slot":
                r.read(x.page(0), slot)
                r.reads_done()
            if cut != "unreserved":
                lane.receive(r)
            if cut == "double_receive":
                lane.receive(r)
            if cut == "touch":
                r.sem_set(lane.flag, 1)
            r.push(cb_x, 1)

        if cut == "branch":
            with r.when(n.ne(1)):
                receive()
            with r.otherwise():
                receive()
        else:
            receive()
    if cut == "misaligned":
        r.reserve(cb_x, 1)
        r.push(cb_x, 1)
    r.atomics_done()

    c = p.kernel("compute", unpack_fp32=[cb_x])
    c.startup(cb_x, None, cb_out)
    c.init("copy", cb_x)
    with c.loop(n, name="i"):
        c.wait(cb_x, 1)
        c.acquire()
        c.copy(cb_x, 0, 0)
        c.reserve(cb_out, 1)
        c.pack(0, cb_out)
        c.release()
        c.push(cb_out, 1)
        c.pop(cb_x, 1)
    if cut == "misaligned":
        c.wait(cb_x, 1)
        c.pop(cb_x, 1)

    w = p.kernel("dm1")
    if cut == "ref_off_grid":
        below_me = p.core_ref(lambda x, y: (x, y + 1), "below")
        w.sem_inc(p.semaphore(0, name="poke"), at=below_me)
        w.atomics_done()
    if cut == "two_senders":
        lane.open(w)
    with w.loop(n, name="i") as i:
        w.wait(cb_out, 1)
        w.write(cb_out, out.page(p.core * n + i))
        w.flushed()
        w.pop(cb_out, 1)
    if cut == "two_senders":
        slot = w.reserve(cb_out, 1)
        lane.send(w, slot)
    w.writes_done()
    return p
