"""A branch of the compute kernel on a value it computed: per tile of x, the flag
"row 0 holds an element above 3" is computed in registers (a select, then a row
max), packed into a buffer the compute kernel itself waits on, read back as a word
on all three threads, and the tile is doubled only where it is set."""

import kohakutt.dsl.metal as km

f = km.sfpu


def flag_program(machine, cut=None):
    """out[i] = 2 x[i] where row 0 of tile x[i] has an element above 3, else x[i];
    runtime value `tiles`. `cut` "unwaited" seeds a read of the flag before its
    wait."""
    p = km.Program("flag", machine)
    x, out = p.input("x", "float32"), p.output("out", "float32")
    tiles = p.runtime("tiles")
    cb_x, cb_out = p.cb(0, like=x), p.cb(16, like=out)
    cb_flag = p.cb(24, like=out, name="flag")

    k = p.kernel("dm0")
    with k.loop(tiles, name="i") as i:
        slot = k.reserve(cb_x, 1)
        k.read(x.page(i), slot)
        k.reads_done()
        k.push(cb_x, 1)

    a = f.operand(0)
    above = km.sfpu_fn("above_fn", f.select(a > 3.0, 1.0, 0.0))
    double = km.sfpu_fn("double_fn", a * 2.0)
    c = p.kernel("compute", fp32_dest=True, unpack_fp32=[cb_x, cb_flag], approx=False)
    c.startup(cb_x, cb_x, cb_out)
    with c.loop(tiles):
        c.wait(cb_x, 1)
        c.acquire()
        c.init("copy", cb_x)
        c.copy(cb_x, 0, 0)
        c.init("mul")
        c.run(above, 0, out=1)
        c.row_reduce("max", 1, 1, into=1, fresh=True)
        c.reserve(cb_flag, 1)
        c.pack(1, cb_flag)
        c.release()
        c.push(cb_flag, 1)
        if cut != "unwaited":
            c.wait(cb_flag, 1)
        flag = c.read_word(cb_flag, 0)
        if cut == "unwaited":
            c.wait(cb_flag, 1)
        c.pop(cb_flag, 1)
        c.acquire()
        c.init("copy", cb_x)
        c.copy(cb_x, 0, 0)
        with c.when(flag.ne(0)):
            c.init("mul")
            c.run(double, 0, out=0)
        c.reserve(cb_out, 1)
        c.pack(0, cb_out)
        c.release()
        c.push(cb_out, 1)
        c.pop(cb_x, 1)

    w = p.kernel("dm1")
    with w.loop(tiles, name="i") as i:
        w.wait(cb_out, 1)
        w.write(cb_out, out.page(i))
        w.flushed()
        w.pop(cb_out, 1)
    w.writes_done()
    return p
