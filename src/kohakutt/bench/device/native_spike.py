"""Experiment: the native tt device (step 6 spike): load order, correctness, host work.

Per size: .to("tt"), a + b, a @ b against fp64, with bridge upload/download/op counts.
    python -m kohakutt.bench.device.native_spike [--sizes 64 512 2048] [--load-order]
"""

import argparse
import json
import subprocess
import sys
import time

import torch

import kohakutt
from kohakutt.core.stats import diff_stats

# Each import order runs in its own process; all must open and close the device.
LOAD_ORDERS = {
    "torch_first": "import torch, kohakutt as n",
    "ttnn_first": "import ttnn, torch, kohakutt as n",
    "kohakutt_then_ttnn": "import torch, kohakutt as n, ttnn",
}


def load_order():
    """{order: ok or the last error line} for every LOAD_ORDERS entry."""
    results = {}
    for name, imports in LOAD_ORDERS.items():
        code = f"{imports}; n.open(0); n.close(); print('ok')"
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
        )
        lines = (proc.stdout + proc.stderr).strip().splitlines()
        ok = proc.returncode == 0 and "ok" in proc.stdout.splitlines()
        results[name] = "ok" if ok else lines[-1:]
    return results


def run(native, n):
    """One record at size n x n: tt and CPU bf16 error stats vs fp64 and the bridge
    counter deltas."""
    gen = torch.Generator().manual_seed(n)
    a = torch.randn(n, n, generator=gen).bfloat16()
    b = (torch.randn(n, n, generator=gen) / n**0.5).bfloat16()

    start = native.counters()
    at, bt = a.to("tt"), b.to("tt")
    after_upload = native.counters()
    tic = time.perf_counter()
    s, p = at + bt, at @ bt
    dispatch_us = (time.perf_counter() - tic) * 1e6
    after_ops = native.counters()
    s_cpu, p_cpu = s.cpu(), p.cpu()
    end = native.counters()

    def delta(x, y):
        return {k: y[k] - x[k] for k in x}

    ad, bd = a.double(), b.double()
    return {
        "n": n,
        "device": str(s.device),
        "add": diff_stats(s_cpu, ad + bd),
        "cpu_bf16_add": diff_stats(a + b, ad + bd),
        "mm": diff_stats(p_cpu, ad @ bd),
        "cpu_bf16_mm": diff_stats(a @ b, ad @ bd),
        "to_tt": delta(start, after_upload),
        "ops": delta(after_upload, after_ops),
        "cpu": delta(after_ops, end),
        "two_ops_host_us": round(dispatch_us, 1),
    }


def main():
    """Runs the load-order check (optional) and every size; prints one record each."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 512, 2048])
    parser.add_argument("--load-order", action="store_true")
    args = parser.parse_args()
    if args.load_order:
        print(json.dumps({"load_order": load_order()}), flush=True)

    kohakutt.open(0)
    try:
        for n in args.sizes:
            print(json.dumps(run(kohakutt, n)), flush=True)
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
