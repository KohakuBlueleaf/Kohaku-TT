"""A tile carried in L1 and updated in place by the packer (c.hold, c.fence): per
tile of x, page 0 of a held buffer is written x, copied back and halved,
accumulated onto with 1.0 (L1 accumulation), and copied out: x / 2 + 1."""

import kohakutt.dsl.metal as km

f = km.sfpu


def hold_program(machine, cut=None):
    """out[i] = x[i] / 2 + 1 for `tiles` (runtime) tiles. `cut`: "unfenced" copies
    the held page after a pack with no fence; "late" holds after a use."""
    p = km.Program("hold", machine)
    x, out = p.input("x", "float32"), p.output("out", "float32")
    tiles = p.runtime("tiles")
    cb_x, cb_out = p.cb(0, like=x), p.cb(16, like=out)
    acc = p.cb(24, dtype="float32", pages=1, name="acc")
    token = p.words(pages=2, name="token")
    half = km.sfpu_fn("half_fn", f.operand(0) * 0.5)

    k = p.kernel("dm0")
    with k.loop(tiles, name="i") as i:
        slot = k.reserve(cb_x, 1)
        k.read(x.page(i), slot)
        k.reads_done()
        k.push(cb_x, 1)

    c = p.kernel("compute", fp32_dest=True, unpack_fp32=[cb_x, acc], approx=False)
    c.startup(cb_x, cb_x, cb_out)
    if cut != "late":
        c.hold(acc)
    with c.loop(tiles):
        c.wait(cb_x, 1)
        c.acquire()
        c.init("copy", cb_x)
        c.copy(cb_x, 0, 0)
        c.pack(0, acc, at=0)
        c.release()
        c.pop(cb_x, 1)
        if cut != "unfenced":
            c.fence(token)
        if cut == "late":
            c.hold(acc)
        c.acquire()
        c.init("copy", acc)
        c.copy(acc, 0, 0)
        c.init(half.init)
        c.run(half, 0, out=0)
        c.pack(0, acc, at=0)
        c.release()
        c.fence(token)
        c.acquire()
        c.init("fill")
        c.op("fill", 0, value=1.0)
        c.pack_accumulate(True, acc)
        c.pack(0, acc, at=0)
        c.release()
        c.pack_accumulate(False, acc)
        c.fence(token)
        c.acquire()
        c.init("copy", acc)
        c.copy(acc, 0, 0)
        c.reserve(cb_out, 1)
        c.pack(0, cb_out)
        c.release()
        c.push(cb_out, 1)

    w = p.kernel("dm1")
    with w.loop(tiles, name="i") as i:
        w.wait(cb_out, 1)
        w.write(cb_out, out.page(i))
        w.flushed()
        w.pop(cb_out, 1)
    w.writes_done()
    return p
