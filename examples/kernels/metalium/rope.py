"""Rotary position embedding as one user Metalium program (torch op
``kohakutt_example::rope``); prints its error and its time against the aten ops.
Usage: python examples/kernels/metalium/rope.py [--batch B] [--heads H] [--tokens S]
    [--dim D]
"""

import argparse
import time

import torch

import kohakutt
import kohakutt.kernels as kk

# Tiles per NoC barrier (kernels/rope_reader.cpp, rope_compute.cpp).
BLOCK = 4


@kk.op("kohakutt_example::rope(Tensor x, Tensor table) -> Tensor")
def rope(x, table):
    """The program rotating x [..., S, D] (bf16, D even) by table [S, D] (fp32, cos at
    even columns and sin at odd ones); the table repeats over x's leading dims."""
    if x.dtype != torch.bfloat16 or table.dtype != torch.float32:
        raise TypeError("rope: x must be bfloat16 and the table float32")
    if tuple(table.shape) != tuple(x.shape[-2:]) or x.shape[-1] % 2:
        raise ValueError("rope: the table must be [S, D] of x [..., S, D] with D even")
    out = kk.empty_like(x)
    split = kk.split_tiles(x)
    period = kk.tile_count(table)
    return kk.Program(
        cbs=[
            kk.CB(0, x, tiles=2 * BLOCK),
            kk.CB(1, table, tiles=2 * BLOCK),
            kk.CB(16, out),
        ],
        reader=kk.Kernel(
            "kernels/rope_reader.cpp",
            [x, table],
            lambda c: [x, table, *split[c], period],
            defines={"BLOCK": BLOCK},
        ),
        compute=kk.Kernel(
            "kernels/rope_compute.cpp",
            runtime_args=lambda c: [split[c][1]],
            defines={"BLOCK": BLOCK},
            fp32_dest=True,
            unpack_fp32=[1],
        ),
        writer=kk.Kernel("kernels/writer.cpp", [out], lambda c: [out, *split[c]]),
        outputs=[out],
        cores=split,
    )


def as_the_model(x, freqs):
    """RoPE as model code writes it: x [..., S, D] times complex freqs [S, D / 2]."""
    pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(pairs * freqs).flatten(-2).to(x.dtype)


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
    """Runs rope on tt tensors and prints its error and time."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=9)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=576)
    parser.add_argument("--dim", type=int, default=64)
    args = parser.parse_args()

    torch.manual_seed(0)
    x = torch.randn(args.batch, args.heads, args.tokens, args.dim).bfloat16()
    angles = torch.rand(args.tokens, args.dim // 2) * 6.2831853
    freqs = torch.polar(torch.ones_like(angles), angles)
    table = torch.view_as_real(freqs).reshape(args.tokens, args.dim).contiguous()

    dev = kohakutt.device(0)
    try:
        x_tt, table_tt, freqs_tt = x.to(dev), table.to(dev), freqs.to(dev)
        got = torch.ops.kohakutt_example.rope(x_tt, table_tt).cpu()
        ruler = torch.view_as_real(
            torch.view_as_complex(x.double().reshape(*x.shape[:-1], -1, 2))
            * freqs.to(torch.complex128)
        ).flatten(-2)
        sim = as_the_model(x, freqs)
        print(f"rope {tuple(x.shape)}")
        print(f"  vs float64:     {stats(got, ruler)}")
        print(f"  sim vs float64: {stats(sim, ruler)}")
        print(
            f"  differs from the CPU sim in {(got != sim).float().mean():.2%} of values"
        )

        own = timed(lambda: torch.ops.kohakutt_example.rope(x_tt, table_tt))
        aten = timed(lambda: as_the_model(x_tt, freqs_tt), reps=5)
        print(f"  time: program {own * 1e3:.3f} ms, aten ops {aten * 1e3:.3f} ms")
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
