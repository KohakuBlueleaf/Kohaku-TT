"""swiglu(a, b) = silu(a) * b written at the metal level (`kohakutt.dsl.metal`): the
program of swiglu.py as a model, with generated kernel sources (torch op
``kohakutt_example::swiglu_km``). Prints whether it equals swiglu.py and both times.
Usage: python examples/kernels/dsl_metal/swiglu_km.py [--rows R] [--cols C]
"""

import argparse
import importlib.util
import time
from pathlib import Path

import torch

import kohakutt
import kohakutt.dsl.metal as km
import kohakutt.kernels as kk


@km.program("swiglu")
def swiglu_program(p, a: km.TensorRef, b: km.TensorRef, out: km.Out):
    """silu(a) * b, one tile per step. Runtime value: `tiles` of a."""
    start, count = p.split(p.runtime("tiles"))
    cb_a, cb_b, cb_out = p.cb(0, like=a), p.cb(1, like=b), p.cb(16, like=out)

    @p.dm0
    def reader(k):
        for i in k.range(count):
            a_slot, b_slot = cb_a.reserve(1), cb_b.reserve(1)
            t = k.let("t", start + i)
            k.read(a.page(t), a_slot)
            k.read(b.page(t), b_slot)
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
                # a in slot 0, b in slot 1; slot 0 = silu(slot 0) * slot 1.
                c.init("copy", cb_a)
                c.copy(cb_a, 0, 0)
                c.init("copy", cb_b)
                c.copy(cb_b, 0, 1)
                c.init("silu")
                c.silu(0)
                c.init("mul")
                c.mul(0, 1, 0)
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


@km.op("kohakutt_example::swiglu_km(Tensor a, Tensor b) -> Tensor", swiglu_program)
def swiglu(a, b):
    """The launch computing silu(a) * b for tt tensors a, b of the same sizes."""
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError("swiglu: a and b must have the same sizes and dtype")
    return km.Call(
        inputs={"a": a, "b": b},
        outputs={"out": (list(a.shape), a.dtype)},
        runtime={"tiles": kk.tile_count(a)},
    )


def hand_written():
    """The torch op of swiglu.py (the same program with hand-written kernels)."""
    path = Path(__file__).parents[1] / "metalium" / "swiglu.py"
    spec = importlib.util.spec_from_file_location("swiglu_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.swiglu.op


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
    """Runs both programs on tt tensors; prints their agreement and times."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--cols", type=int, default=4096)
    args = parser.parse_args()

    torch.manual_seed(0)
    a = torch.randn(args.rows, args.cols).bfloat16()
    b = torch.randn(args.rows, args.cols).bfloat16()
    dev = kohakutt.device(0)
    try:
        a_tt, b_tt = a.to(dev), b.to(dev)
        reference = hand_written()
        got = torch.ops.kohakutt_example.swiglu_km(a_tt, b_tt).cpu()
        want = reference(a_tt, b_tt).cpu()
        same = torch.equal(got, want)
        print(f"swiglu_km {args.rows}x{args.cols}: equal to swiglu.py: {same}")
        own = timed(lambda: torch.ops.kohakutt_example.swiglu_km(a_tt, b_tt))
        hand = timed(lambda: reference(a_tt, b_tt))
        print(f"  time: metal level {own * 1e3:.3f} ms, swiglu.py {hand * 1e3:.3f} ms")
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
