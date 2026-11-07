"""Loads the C++ SwiGLU extension and runs torch.ops.kohakutt_example_cpp.swiglu on tt
tensors; prints its error against float64 and an fp32-math simulation, and its time.
Usage: python examples/custom_kernel_cpp/swiglu.py --library <path to swiglu_ext.so>
"""

import argparse
import time

import torch
import torch.nn.functional as F

import kohakutt
import kohakutt.extension


def stats(got, ref):
    """Max / median / mean of the absolute and relative error of `got` against `ref`."""
    err = (got.double() - ref).abs()
    rel = err / ref.abs().clamp_min(1e-30)
    return " ".join(
        f"{name} max {e.max():.2e} med {e.median():.2e} mean {e.mean():.2e}"
        for name, e in (("abs", err), ("rel", rel))
    )


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
    """Runs the extension op on tt tensors and prints its error and time."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", default="build/swiglu_ext/swiglu_ext.so")
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--cols", type=int, default=4096)
    args = parser.parse_args()
    kohakutt.extension.load(args.library)

    dev = kohakutt.device(0)
    a = torch.randn(args.rows, args.cols).bfloat16()
    b = torch.randn(args.rows, args.cols).bfloat16()
    a_tt, b_tt = a.to(dev), b.to(dev)
    swiglu = torch.ops.kohakutt_example_cpp.swiglu

    got = swiglu(a_tt, b_tt).cpu()
    ruler = F.silu(a.double()) * b.double()
    sim = (F.silu(a.float()) * b.float()).bfloat16()
    print(f"swiglu (C++ extension) {args.rows}x{args.cols}")
    print(f"  vs float64:     {stats(got, ruler)}")
    print(f"  sim vs float64: {stats(sim, ruler)}")

    own = timed(lambda: swiglu(a_tt, b_tt))
    aten = timed(lambda: F.silu(a_tt) * b_tt)
    print(f"  time: program {own * 1e3:.3f} ms, aten silu * b {aten * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
