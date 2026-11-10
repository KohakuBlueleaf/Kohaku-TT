"""swiglu(a, b) = silu(a) * b as a user Metalium program registered as torch op
``kohakutt_example::swiglu`` and called on tt tensors; prints its error against float64
and an fp32-math simulation, and its time against the aten composition.
Usage: python examples/kernels/metalium/swiglu.py [--rows R] [--cols C]
"""

import argparse
import time

import torch
import torch.nn.functional as F

import kohakutt
import kohakutt.kernels as kk


@kk.op("kohakutt_example::swiglu(Tensor a, Tensor b) -> Tensor")
def swiglu(a, b):
    """The program computing silu(a) * b for tt tensors a, b of the same sizes."""
    out = kk.empty_like(a)
    split = kk.split_tiles(a)
    return kk.Program(
        cbs=[kk.CB(0, a), kk.CB(1, b), kk.CB(16, out)],
        reader=kk.Kernel("kernels/reader.cpp", [a, b], lambda c: [a, b, *split[c]]),
        compute=kk.Kernel(
            "kernels/compute.cpp", runtime_args=lambda c: [split[c][1]], fp32_dest=True
        ),
        writer=kk.Kernel("kernels/writer.cpp", [out], lambda c: [out, *split[c]]),
        outputs=[out],
        cores=split,
    )


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
    """Runs swiglu on tt tensors and prints its error and time."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--cols", type=int, default=4096)
    args = parser.parse_args()

    a = torch.randn(args.rows, args.cols).bfloat16()
    b = torch.randn(args.rows, args.cols).bfloat16()
    dev = kohakutt.device(0)
    a_tt, b_tt = a.to(dev), b.to(dev)

    got = torch.ops.kohakutt_example.swiglu(a_tt, b_tt).cpu()
    ruler = F.silu(a.double()) * b.double()
    sim = (F.silu(a.float()) * b.float()).bfloat16()
    print(f"swiglu {args.rows}x{args.cols}")
    print(f"  vs float64:    {stats(got, ruler)}")
    print(f"  sim vs float64: {stats(sim, ruler)}")

    own = timed(lambda: torch.ops.kohakutt_example.swiglu(a_tt, b_tt))
    aten = timed(lambda: F.silu(a_tt) * b_tt)
    print(f"  time: program {own * 1e3:.3f} ms, aten silu * b {aten * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
