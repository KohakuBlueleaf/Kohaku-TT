"""a * s + b at the metal level, one tile per step (torch op `scaled_add_km`); prints
its error against float64 next to the CPU's, and its time against aten's (s: --scale).
Usage: python examples/kernels/dsl_metal/scaled_add_km.py [--rows R] [--cols C]
"""

import argparse
import struct
import time

import torch

import kohakutt
import kohakutt.dsl.metal as km
import kohakutt.kernels as kk


@km.program("scaled_add")
def scaled_add_program(p, a: km.TensorRef, b: km.TensorRef, out: km.Out):
    """a * s + b, one tile per step. Runtime values: `tiles` of a, `s` (fp32 bits)."""
    start, count = p.split(p.runtime("tiles"))  # each core's range of tiles
    s = p.runtime("s")  # fp32 bits of the scale
    cb_a, cb_b, cb_out = p.cb(0, like=a), p.cb(1, like=b), p.cb(16, like=out)

    @p.dm0
    def reader(k):
        for i in k.range(count):
            a_slot, b_slot = cb_a.reserve(1), cb_b.reserve(1)
            k.read(a.page(start + i), a_slot)
            k.read(b.page(start + i), b_slot)
            k.reads_done()
            cb_a.push(1)
            cb_b.push(1)

    @p.compute(fp32_dest=True)
    def compute(c):
        c.startup(cb_a, cb_b, cb_out)
        for _ in c.range(count):
            cb_a.wait(1)
            cb_b.wait(1)
            with c.registers():
                c.init("copy", cb_a)
                c.copy(cb_a, 0, 0)  # a in register slot 0
                c.init("copy", cb_b)
                c.copy(cb_b, 0, 1)  # b in register slot 1
                c.init("mul_scalar")
                c.op("mul_scalar", 0, value=s)  # slot 0 = slot 0 * s
                c.init("add")
                c.op("add", 0, 1, 0)  # slot 0 = slot 0 + slot 1
                cb_out.reserve(1)
                c.pack(0, cb_out)
                cb_out.push(1)
            cb_a.pop(1)
            cb_b.pop(1)

    @p.dm1
    def writer(k):
        for i in k.range(count):
            cb_out.wait(1)
            k.write(cb_out, out.page(start + i))
            k.flushed()
            cb_out.pop(1)
        k.writes_done()


@km.op(
    "kohakutt_example::scaled_add_km(Tensor a, Tensor b, float s) -> Tensor",
    scaled_add_program,
)
def scaled_add(a, b, s):
    """The launch computing a * s + b for tt tensors a, b of the same sizes."""
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError("scaled_add: a and b must have the same sizes and dtype")
    return km.Call(
        inputs={"a": a, "b": b},
        outputs={"out": (list(a.shape), a.dtype)},
        runtime={"tiles": kk.tile_count(a), "s": bits(s)},
    )


def bits(value):
    """The fp32 bits of `value`, as a runtime value."""
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def timed(fn, reps=20):
    """Mean seconds per call of `fn` after one warm-up call."""
    fn()
    kohakutt.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    kohakutt.synchronize()
    return (time.perf_counter() - start) / reps


def main():
    """Runs the program on tt tensors; prints its error and time."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--scale", type=float, default=0.5)
    args = parser.parse_args()

    torch.manual_seed(0)
    a = torch.randn(args.rows, args.cols).bfloat16()
    b = torch.randn(args.rows, args.cols).bfloat16()
    ruler = a.double() * args.scale + b.double()
    cpu = a * args.scale + b
    dev = kohakutt.device(0)
    try:
        a_tt, b_tt = a.to(dev), b.to(dev)
        op = torch.ops.kohakutt_example.scaled_add_km
        got = op(a_tt, b_tt, args.scale).cpu()
        print(f"scaled_add_km {args.rows}x{args.cols}, s = {args.scale}")
        for name, result in (("tt", got), ("CPU", cpu)):
            stats = kohakutt.diff_stats(result, ruler)
            print(f"  {name} vs float64: {kohakutt.format_stats(stats)}")
        own = timed(lambda: op(a_tt, b_tt, args.scale))
        aten = timed(lambda: a_tt * args.scale + b_tt)
        print(f"  time: program {own * 1e3:.3f} ms, aten a * s + b {aten * 1e3:.3f} ms")
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
