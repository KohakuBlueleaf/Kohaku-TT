"""A 2-D matmul with both operands shared: core (x, y) computes output tile (y, x) of
a @ w; the left column reads a's tiles and multicasts them along each row (a row
lane), the top row reads w's tiles and multicasts them down each column."""

import kohakutt.dsl.metal as km

CUTS = {
    "holds_sender": "the row lane's rectangle holds the sender's column",
    "idle_column": "a receiving column runs a variant that does not receive",
    "column_gap": "the dm1 variants leave the last column out",
}


def grid2d_program(machine, cut=None):
    """The program above on every core of `machine`; runtime value `kt` (a's column
    tiles). `cut` seeds a bug (CUTS)."""
    p = km.Program("grid2d", machine)
    a, w = p.input("a", "bfloat16"), p.input("w", "bfloat16")
    out = p.output("out", "bfloat16")
    kt, cols, rows = p.runtime("kt"), machine.cols, machine.rows
    cb_a, cb_w = p.cb(0, like=a, pages=2), p.cb(1, like=w, pages=2)
    cb_out = p.cb(16, like=out, pages=1)
    x, y = p.core % cols, p.core // cols

    top, below = p.cores.rows(0), p.cores.rows(1, rows - 1)
    w_lane = p.lane(top, below, name="w")
    left = p.cores.cols(0)
    right = p.cores.cols(0 if cut == "holds_sender" else 1, cols - 1)
    a_lane = p.lane(left, right, name="a")

    def sender(k, lane, cb, page):
        lane.open(k)
        with k.loop(kt, name="k") as i:
            slot = k.reserve(cb, 1)
            k.read(page(i), slot)
            k.reads_done()
            lane.send(k, slot)
            k.push(cb, 1)

    def receiver(k, lane, cb):
        with k.loop(kt, name="k"):
            k.reserve(cb, 1)
            lane.receive(k)
            k.push(cb, 1)
        k.atomics_done()

    def write(k):
        k.wait(cb_out, 1)
        k.write(cb_out, out.page(y * cols + x))
        k.flushed()
        k.pop(cb_out, 1)
        k.writes_done()

    w_send = p.kernel("dm0", cores=top, noc=0, name="w_send")
    sender(w_send, w_lane, cb_w, lambda i: w.page(i * cols + x))
    w_send.writes_done()
    receiver(p.kernel("dm0", cores=below, noc=0, name="w_recv"), w_lane, cb_w)

    a_send = p.kernel("dm1", cores=left, noc=1, name="a_send")
    sender(a_send, a_lane, cb_a, lambda i: a.page(y * kt + i))
    write(a_send)
    last = cols - 1 if cut == "column_gap" else cols
    if cut == "idle_column":
        receiving = p.cores.cols(2, cols - 1)
        idle = p.kernel("dm1", cores=p.cores.cols(1), noc=1, name="a_idle")
        write(idle)
    else:
        receiving = p.cores.cols(1, last - 1)
    a_recv = p.kernel("dm1", cores=receiving, noc=1, name="a_recv")
    receiver(a_recv, a_lane, cb_a)
    write(a_recv)

    c = p.kernel("compute", fp32_dest=True, approx=False)
    c.startup(cb_a, cb_w, cb_out)
    product = c.matmul_init(cb_a, cb_w)
    c.acquire()
    with c.loop(kt, name="k"):
        c.wait(cb_a, 1)
        c.wait(cb_w, 1)
        c.matmul(product, 0, 0, 0)
        c.pop(cb_a, 1)
        c.pop(cb_w, 1)
    c.commit()
    c.reserve(cb_out, 1)
    c.pack(0, cb_out)
    c.release()
    c.push(cb_out, 1)
    return p


def run(launcher, a, w, cols):
    """out [rows * 32, cols * 32] of the launch for tt a [rows * 32, kt * 32] and w
    [kt * 32, cols * 32]."""
    import torch

    (out,) = launcher(
        {"a": a, "w": w},
        outputs={"out": ([a.shape[0], cols * 32], torch.bfloat16)},
        runtime={"kt": a.shape[1] // 32},
    )
    return out
