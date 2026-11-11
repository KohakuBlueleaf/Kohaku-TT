"""Row max and row sum of register slots on the vector unit (c.row_reduce): three
score tiles and the previous value (a column broadcast) go into slots 0 .. 3; the
reduce writes slot 3. `flash=True` then runs ktt_flash's step on the slots: b = 639 -
a m, P = flash_exp(S, a, b) over the three score tiles, packed with b."""

import kohakutt.dsl.metal as km

f = km.sfpu
A = 0.18033688011112042


def rowreduce_program(machine, op="max", fresh=False, flash=False):
    """Per item: s [32, 96] (three tiles) and prev [32, 32] -> out [32, 32] (the
    reduced slot, or b with `flash`) and, with `flash`, p [32, 96]. Runtime value
    `items`."""
    p = km.Program(
        f"row_{op}{'_fresh' if fresh else ''}{'_flash' if flash else ''}", machine
    )
    s, prev = p.input("s", "float32"), p.input("prev", "float32")
    out = p.output("out", "float32")
    probs = p.output("p", "float32") if flash else None
    items = p.runtime("items")
    cb_s, cb_prev = p.cb(0, like=s, pages=3), p.cb(1, like=prev)
    cb_out = p.cb(16, like=out)
    cb_p = p.cb(17, like=probs, pages=3) if flash else None

    k = p.kernel("dm0")
    with k.loop(items, name="i") as i:
        slot = k.reserve(cb_s, 3)
        for t in range(3):
            k.read(s.page(i * 3 + t), slot[t])
        held = k.reserve(cb_prev, 1)
        k.read(prev.page(i), held)
        k.reads_done()
        k.push(cb_s, 3)
        k.push(cb_prev, 1)

    c = p.kernel("compute", fp32_dest=True, unpack_fp32=[cb_s, cb_prev], approx=False)
    c.startup(cb_s, cb_prev, cb_out)
    m = f.operand(0)
    b_fn = km.sfpu_fn("b_fn", f.mad(m, -A, 639.0))
    x, b = f.operand(0), f.operand(1, along="col")
    exp = km.sfpu_fn("p_fn", f.flash_exp(x, A, b))
    with c.loop(items):
        c.wait(cb_s, 3)
        c.wait(cb_prev, 1)
        c.acquire()
        c.init("copy", cb_s)
        for t in range(3):
            c.copy(cb_s, t, t)
        c.init("copy", cb_prev)
        c.copy(cb_prev, 0, 3)
        c.init("mul")
        c.row_reduce(op, 0, 3, into=3, fresh=fresh)
        if flash:
            c.run(b_fn, 3, out=3)
            c.run(exp, 0, 3, out=0, tiles=3)
        c.reserve(cb_out, 1)
        c.pack(3, cb_out)
        if flash:
            c.reserve(cb_p, 3)
            for t in range(3):
                c.pack(t, cb_p)
        c.release()
        c.push(cb_out, 1)
        if flash:
            c.push(cb_p, 3)
        c.pop(cb_s, 3)
        c.pop(cb_prev, 1)

    w = p.kernel("dm1")
    with w.loop(items, name="i") as i:
        w.wait(cb_out, 1)
        w.write(cb_out, out.page(i))
        w.flushed()
        w.pop(cb_out, 1)
        if flash:
            w.wait(cb_p, 3)
            for t in range(3):
                w.write(cb_p, probs.page(i * 3 + t), t)
            w.flushed()
            w.pop(cb_p, 3)
    w.writes_done()
    return p
