"""Achieved DRAM bandwidth of tt memory and vector ops (bytes read + written per call
vs 512 GB/s), median of synchronized calls; one JSON line per op.
Usage: python -m kohakutt.bench.perf.memops [--rows 8192] [--cols 4096]
"""

import argparse
import json
import statistics
import time

import torch

import kohakutt

DRAM_GBPS = 512.0


def timed(fn, dev, calls):
    """Median seconds per call of fn() over `calls` synchronized calls (after 2)."""
    for _ in range(2):
        fn()
    kohakutt.synchronize(dev)
    samples = []
    for _ in range(calls):
        start = time.perf_counter()
        fn()
        kohakutt.synchronize(dev)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def cases(x, y, idx):
    """{name: (call, bytes moved)} over bf16 [r, c] tensors x, y and an index."""
    r, c = x.shape
    e = x.numel() * x.element_size()
    half = e // 2
    return {
        "clone": (lambda: x.clone(), 2 * e),
        "copy_ (into y)": (lambda: y.copy_(x), 2 * e),
        "add": (lambda: x + y, 3 * e),
        "add_ (in place)": (lambda: y.add_(x), 3 * e),
        "mul by scalar": (lambda: x * 2.0, 2 * e),
        "typecast bf16->fp32": (lambda: x.float(), e + 2 * e),
        "transpose(0,1).contiguous": (lambda: x.t().contiguous(), 2 * e),
        "permute 3-D (0,2,1)": (
            lambda: x.view(r // 32, 32, c).permute(0, 2, 1).contiguous(),
            2 * e,
        ),
        "reshape (merge leading)": (
            lambda: x.view(r // 2, 2, c).reshape(r, c) + 0,
            3 * e,
        ),
        "reshape (split last dim)": (
            lambda: x.view(r, c // 64, 64).contiguous(),
            2 * e,
        ),
        "slice rows [:r/2]": (lambda: x[: r // 2].contiguous(), 2 * half),
        "slice cols [:, :c/2]": (lambda: x[:, : c // 2].contiguous(), 2 * half),
        "narrow row block, middle": (
            lambda: x.narrow(0, r // 4, r // 2).contiguous(),
            2 * half,
        ),
        "cat dim 0": (lambda: torch.cat([x, y], 0), 4 * e),
        "cat dim 1": (lambda: torch.cat([x, y], 1), 4 * e),
        "repeat (2, 1)": (lambda: x[: r // 2].repeat(2, 1), e + half),
        "expand + add (broadcast row)": (lambda: x + y[:1], 2 * e),
        "index_select rows": (lambda: x.index_select(0, idx), 2 * e),
        "write slice x[:r/2] = y[:r/2]": (
            lambda: x[: r // 2].copy_(y[: r // 2]),
            2 * half,
        ),
        "zero_": (lambda: y.zero_(), e),
    }


def main():
    """Measures every case on device 0 and prints one JSON line each."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--calls", type=int, default=5)
    args = parser.parse_args()

    dev = kohakutt.device(0)
    x = torch.randn(args.rows, args.cols).bfloat16().to(dev)
    y = torch.randn(args.rows, args.cols).bfloat16().to(dev)
    idx = torch.randperm(args.rows).to(dev)
    for name, (fn, nbytes) in cases(x, y, idx).items():
        try:
            seconds = timed(fn, dev, args.calls)
            gbps = nbytes / seconds / 1e9
            record = {
                "op": name,
                "ms": round(seconds * 1e3, 3),
                "GB/s": round(gbps, 1),
                "of_peak": round(gbps / DRAM_GBPS, 3),
                "tt_programs": None,
            }
            before = kohakutt.counters()["ops"]
            fn()
            record["tt_programs"] = kohakutt.counters()["ops"] - before
        except Exception as error:  # noqa: BLE001  (a failing case is reported)
            record = {"op": name, "error": str(error).splitlines()[0][:160]}
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
