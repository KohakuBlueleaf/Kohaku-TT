"""A metal-level program for tests of the matrix unit, the op table, a broadcast
unpack, a runtime branch, packs at a page and packer accumulation: y = relu(x @ w) +
bias; out = y * g's first row (flag != 0) or y - 1; acc = out's two column tiles
summed in fp32. Built through the explicit builder; run with km.Launcher."""

import struct

import kohakutt.dsl.metal as km

DTYPES = {"x": "bfloat16", "w": "bfloat16", "g": "bfloat16"}
# Inner (K) and column (N) tiles of the product.
KT = NT = 2


def units_program(machine):
    """The program above. Runtime values: `rows` (row tiles of x), `flag`, and
    `bias` (the bits of an fp32 value)."""
    p = km.Program("units", machine)
    x, w, g = (p.input(name, dtype) for name, dtype in DTYPES.items())
    out, acc = p.output("out", "bfloat16"), p.output("acc", "float32")
    flag, bias = p.runtime("flag"), p.runtime("bias")
    first, count = p.split(p.runtime("rows"))
    cb_x, cb_w = p.cb(0, like=x, pages=2 * KT), p.cb(1, like=w, pages=KT * NT)
    cb_g = p.cb(2, like=g, pages=NT)
    cb_out, cb_acc = p.cb(16, like=out, pages=2 * NT), p.cb(17, like=acc, pages=2)

    k = p.kernel("dm0")
    for cb, ref, pages in ((cb_w, w, KT * NT), (cb_g, g, NT)):
        slot = k.reserve(cb, pages)
        with k.loop(pages, name="t") as t:
            k.read(ref.page(t), slot[t])
        k.reads_done()
        k.push(cb, pages)
    with k.loop(count, name="m") as m:
        slot = k.reserve(cb_x, KT)
        with k.loop(KT, name="k") as i:
            k.read(x.page((first + m) * KT + i), slot[i])
        k.reads_done()
        k.push(cb_x, KT)

    c = p.kernel("compute", fp32_dest=True)
    c.startup(cb_x, cb_w, cb_out)
    c.wait(cb_w, KT * NT)
    c.wait(cb_g, NT)
    with c.loop(count, name="m"):
        c.wait(cb_x, KT)
        c.acquire()
        product = c.matmul_init(cb_x, cb_w, transpose=0, ct=NT, rt=1, kt=KT)
        with c.loop(KT, name="k") as i:
            c.matmul(product, i, i * NT, 0)
        c.init("relu")
        c.op("relu", 0)
        c.op("relu", 1)
        c.init("add_scalar")
        c.op("add_scalar", 0, value=bias)
        c.op("add_scalar", 1, value=bias)
        with c.when(flag.ne(0)):
            c.format(srca=cb_g, srcb=cb_g)
            c.init("bcast_row", cb_g)
            c.bcast(cb_g, 0, 2, "row")
            c.bcast(cb_g, 1, 3, "row")
            c.init("mul")
            c.op("mul", 0, 2, 0)
            c.op("mul", 1, 3, 1)
        with c.otherwise():
            c.init("sub_scalar")
            c.op("sub_scalar", 0, value=1.0)
            c.op("sub_scalar", 1, value=1.0)
        c.commit()
        c.reserve(cb_out, NT)
        c.pack(0, cb_out, at=0)
        c.pack(1, cb_out, at=1)
        c.pack_format(cb_acc)
        c.reserve(cb_acc, 1)
        c.pack(0, cb_acc, at=0)
        c.pack_accumulate(1, cb_acc)
        c.pack(1, cb_acc, at=0)
        c.pack_accumulate(0)
        c.pack_format(cb_out)
        c.release()
        c.push(cb_out, NT)
        c.push(cb_acc, 1)
        c.pop(cb_x, KT)
    c.pop(cb_w, KT * NT)
    c.pop(cb_g, NT)

    d = p.kernel("dm1")
    with d.loop(count, name="m") as m:
        d.wait(cb_out, NT)
        with d.loop(NT, name="j") as j:
            d.write(cb_out, out.page((first + m) * NT + j), index=j)
        d.wait(cb_acc, 1)
        d.write(cb_acc, acc.page(first + m))
        d.flushed()
        d.pop(cb_out, NT)
        d.pop(cb_acc, 1)
    d.writes_done()
    return p


def float_bits(value):
    """The bit pattern of fp32 `value`, as a runtime value."""
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def run(launcher, x, w, g, flag, bias):
    """(out, acc) of the launch for tt tensors x [R * 32, 64], w [64, 64] and
    g [32, 64]."""
    import torch

    rows = x.shape[0] // 32
    return launcher(
        {"x": x, "w": w, "g": g},
        outputs={
            "out": ([rows * 32, 64], torch.bfloat16),
            "acc": ([rows * 32, 32], torch.float32),
        },
        runtime={"rows": rows, "flag": int(flag), "bias": float_bits(bias)},
    )
