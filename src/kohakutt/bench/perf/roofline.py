"""Roofline of one tt chip: matmul TFLOPS vs the FPU peak at the active fidelity
(KTT_MATMUL_FIDELITY), elementwise / reduction streams vs the DRAM peak; one JSON
record. Usage: python -m kohakutt.bench.perf.roofline [--sizes] [--elements]
"""

import argparse
import json
import os
import statistics
import time

import torch

import kohakutt

# P150A peaks: bf16 FPU FLOP / cycle / core at LoFi, fidelity passes.
FLOP_PER_CYCLE_LOFI = 4096
PHASES = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}
DRAM_GBPS = 512.0


def timed(fn, dev, calls):
    """Median device-synced seconds per call of fn() over `calls` calls (after 2)."""
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


def matmul_rows(dev, sizes, peak_tflops, calls):
    """[n, ms, TFLOPS, fraction of peak] for bf16 n x n @ n x n."""
    rows = []
    for n in sizes:
        a = torch.randn(n, n).bfloat16().to(dev)
        b = torch.randn(n, n).bfloat16().to(dev)
        seconds = timed(lambda a=a, b=b: a @ b, dev, calls)
        tflops = 2 * n**3 / seconds / 1e12
        rows.append(
            [
                n,
                round(seconds * 1e3, 3),
                round(tflops, 1),
                round(tflops / peak_tflops, 3),
            ]
        )
    return rows


def stream_rows(dev, elements, calls):
    """{op: [GB/s, fraction of DRAM peak, Gelem/s]} for memory-bound streams."""
    # Tile-aligned 2-D tensors: a 1-D [n] tensor is stored [1, n], padded to 32 rows.
    shape = (elements // 4096, 4096)
    a = torch.randn(shape).bfloat16().to(dev)
    b = torch.randn(shape).bfloat16().to(dev)
    f = torch.randn(shape).to(dev)
    g = torch.randn(shape).to(dev)
    # name: (call, bytes moved per call)
    cases = {
        "copy bf16 (clone)": (lambda: a.clone(), 2 * 2 * elements),
        "add bf16 (a + b)": (lambda: a + b, 3 * 2 * elements),
        "mul bf16 (a * b)": (lambda: a * b, 3 * 2 * elements),
        "exp bf16 (SFPU)": (lambda: a.exp(), 2 * 2 * elements),
        "gelu bf16 (SFPU)": (lambda: torch.nn.functional.gelu(a), 2 * 2 * elements),
        "add fp32": (lambda: f + g, 3 * 4 * elements),
        "exp fp32 (SFPU)": (lambda: f.exp(), 2 * 4 * elements),
        "sum bf16 (reduce)": (lambda: a.sum(-1), 2 * elements),
    }
    rows = {}
    for name, (fn, nbytes) in cases.items():
        seconds = timed(fn, dev, calls)
        gbps = nbytes / seconds / 1e9
        rows[name] = [
            round(gbps, 1),
            round(gbps / DRAM_GBPS, 3),
            round(elements / seconds / 1e9, 2),
        ]
    return rows


def main():
    """Measures matmul and stream throughput on device 0 and prints JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[1024, 2048, 4096, 8192]
    )
    parser.add_argument("--elements", type=int, default=64 << 20)
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--clock-ghz", type=float, default=1.35)
    args = parser.parse_args()

    dev = kohakutt.device(0)
    cols, rows = kohakutt.compute_grid(dev)
    cores = cols * rows
    fidelity = os.environ.get("KTT_MATMUL_FIDELITY", "HiFi4")
    peak = FLOP_PER_CYCLE_LOFI * cores * args.clock_ghz * 1e9 / PHASES[fidelity] / 1e12
    record = {
        "grid": [cols, rows],
        "cores": cores,
        "clock_ghz": args.clock_ghz,
        "fidelity": fidelity,
        "peak_tflops": round(peak, 1),
        "matmul [n, ms, TFLOPS, of peak]": matmul_rows(
            dev, args.sizes, peak, args.calls
        ),
        "streams {op: [GB/s, of 512 GB/s, Gelem/s]}": stream_rows(
            dev, args.elements, args.calls
        ),
    }
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
