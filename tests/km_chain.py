"""A generator for tests: an elementwise chain given as a list of nodes becomes one
metal-level program, built through the explicit builder and run with km.Launcher
(no decorator, no torch op)."""

import math

import kohakutt.dsl.metal as km


def chain_program(machine, nodes, inputs, dtype="float32"):
    """out = (((x0 op x_i) op x_j) ...) for `nodes` [(op, input index), ...] over
    `inputs` tensors of one shape in format `dtype`; op is "add", "sub", "mul" or
    "div". Runtime value: `tiles`."""
    p = km.Program(f"chain{len(nodes)}", machine)
    xs = [p.input(f"x{i}", dtype) for i in range(inputs)]
    out = p.output("out", dtype)
    start, count = p.split(p.runtime("tiles"))
    cbs = [p.cb(i, like=x) for i, x in enumerate(xs)]
    cb_out = p.cb(16, like=out)

    k = p.kernel("dm0")
    with k.loop(count) as i:
        tile = k.let("tile", start + i)
        slots = [k.reserve(cb, 1) for cb in cbs]
        for x, slot in zip(xs, slots):
            k.read(x.page(tile), slot)
        k.reads_done()
        for cb in cbs:
            k.push(cb, 1)

    fp32 = cbs if dtype == "float32" else []
    c = p.kernel("compute", fp32_dest=True, unpack_fp32=fp32, approx=False)
    c.startup(cbs[0], cbs[min(1, inputs - 1)], cb_out)
    with c.loop(count):
        for cb in cbs:
            c.wait(cb, 1)
        c.acquire()
        c.init("copy", cbs[0])
        c.copy(cbs[0], 0, 0)
        for op, index in nodes:
            c.init("copy", cbs[index])
            c.copy(cbs[index], 0, 1)
            c.init(op)
            c.binary(op, 0, 1, 0)
        c.reserve(cb_out, 1)
        c.pack(0, cb_out)
        c.release()
        c.push(cb_out, 1)
        for cb in cbs:
            c.pop(cb, 1)

    w = p.kernel("dm1")
    with w.loop(count) as i:
        w.wait(cb_out, 1)
        w.write(cb_out, out.page(start + i))
        w.flushed()
        w.pop(cb_out, 1)
    w.writes_done()
    return p


def run(launcher, tensors):
    """The chain of `launcher` over tt `tensors` (in input order)."""
    first = tensors[0]
    tiles = math.prod(first.shape[:-2]) * math.ceil(first.shape[-2] / 32)
    tiles *= math.ceil(first.shape[-1] / 32)
    (out,) = launcher(
        {f"x{i}": t for i, t in enumerate(tensors)},
        outputs={"out": (list(first.shape), first.dtype)},
        runtime={"tiles": tiles},
    )
    return out
