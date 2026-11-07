"""Hardware utilization per op class on one chip, on Kohaku-TT or on raw TT-NN (the
baseline: same ops, no torch / kohakutt layers). Compute-bound matmuls report TFLOPS vs
the FPU peak at the measured AICLK; bandwidth-bound matmuls (decode shapes), vector and
memory ops report GB/s vs 512 GB/s DRAM; "launch" reports the per-program cost of a
stream of tiny ops. One JSON line per case. Run as a file (the TT-NN venv has no
kohakutt): python src/kohakutt/bench/perf/utilization.py --stack kohakutt|ttnn
"""

import argparse
import glob
import json
import statistics
import threading
import time

import torch

DRAM_GBPS = 512.0
# bf16 FPU FLOP / cycle / core at LoFi; fidelity = passes per multiply.
FLOP_PER_CYCLE_LOFI = 4096
PASSES = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}
FIDELITY = "HiFi4"


class ClockSampler:
    """Mean AICLK (MHz) of chip 0 over a with-block, sampled from the driver's sysfs."""

    def __init__(self):
        paths = sorted(glob.glob("/sys/class/tenstorrent/*/tt_aiclk"))
        self.path = paths[0] if paths else None
        self.samples = []

    def _run(self):
        while not self.stop.is_set():
            with open(self.path) as f:
                self.samples.append(int(f.read()))
            self.stop.wait(0.01)

    def __enter__(self):
        self.stop = threading.Event()
        if self.path:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        if self.path:
            self.thread.join()

    def mean(self):
        return round(statistics.mean(self.samples)) if self.samples else None


def cases(n_vec, dims):
    """[(class, name, work, kind)]: work is FLOPs (kind 'flops') or bytes moved."""
    r, c = n_vec // 4096, 4096
    e = r * c * 2
    out = [("matmul", f"mm {n}x{n}x{n}", 2 * n**3, "flops") for n in dims]
    out += [
        ("matmul", f"decode linear [{m},{k}]x[{n},{k}]^T", 2 * k * n, "bytes")
        for m, k, n in [(32, 4096, 16384), (1, 4096, 16384), (32, 896, 4864)]
    ]
    for name, mult in [
        ("add", 3),
        ("mul", 3),
        ("exp", 2),
        ("gelu", 2),
        ("silu", 2),
        ("softmax(-1)", 2),
        ("layer_norm(-1)", 2),
        ("sum(-1)", 1),
        ("cast bf16->fp32", 3),
    ]:
        out.append(("vector", name, mult * e, "bytes"))
    for name, mult in [
        ("clone", 2),
        ("transpose(-2,-1)", 2),
        ("permute (0,2,1) 3-D", 2),
        ("slice rows [:r/2]", 1),
        ("slice cols [:, :c/2]", 1),
        ("cat dim 0", 4),
        ("cat dim 1", 4),
        ("repeat (2,1) of half", 1.5),
        ("gather rows (index_select)", 2),
        ("broadcast add (row)", 2),
        ("reshape split last dim", 2),
    ]:
        out.append(("memory", name, mult * e, "bytes"))
    out += [("launch", name, 1, "launch") for name in LAUNCH]
    return out


# Back-to-back streams: us per program (tiny = 1 tile, mid = 1024 tiles).
LAUNCH = [
    "tiny add 32x32 (per program)",
    "tiny exp 32x32 (per program)",
    "mid add 1024 tiles (per program)",
    "mid exp 1024 tiles (per program)",
]


def kohakutt_stack():
    """(ops {name: fn}, sync, programs) on Kohaku-TT device 0."""
    # Optional dependency: Kohaku-TT itself (this repo's venv).
    import torch.nn.functional as F

    import kohakutt

    dev = kohakutt.device(0)

    def sync():
        kohakutt.synchronize(dev)

    def make(n_vec, dims):
        r, c = n_vec // 4096, 4096
        x = torch.randn(r, c).bfloat16().to(dev)
        y = torch.randn(r, c).bfloat16().to(dev)
        idx = torch.randperm(r).to(dev)
        w_ln = torch.ones(c).bfloat16().to(dev)
        tiny = torch.randn(32, 32).bfloat16().to(dev)
        mid = torch.randn(1024, 1024).bfloat16().to(dev)
        ops = {
            "tiny exp 32x32 (per program)": lambda: tiny.exp(),
            "mid add 1024 tiles (per program)": lambda: mid + mid,
            "mid exp 1024 tiles (per program)": lambda: mid.exp(),
        }
        for n in dims:
            a = torch.randn(n, n).bfloat16().to(dev)
            b = torch.randn(n, n).bfloat16().to(dev)
            ops[f"mm {n}x{n}x{n}"] = lambda a=a, b=b: a @ b
        for m, k, n in [(32, 4096, 16384), (1, 4096, 16384), (32, 896, 4864)]:
            xi = torch.randn(m, k).bfloat16().to(dev)
            w = torch.randn(n, k).bfloat16().to(dev)
            ops[f"decode linear [{m},{k}]x[{n},{k}]^T"] = lambda xi=xi, w=w: F.linear(
                xi, w
            )
        ops.update(
            {
                "add": lambda: x + y,
                "mul": lambda: x * y,
                "exp": lambda: x.exp(),
                "gelu": lambda: F.gelu(x),
                "silu": lambda: F.silu(x),
                "softmax(-1)": lambda: x.softmax(-1),
                "layer_norm(-1)": lambda: F.layer_norm(x, (c,), w_ln, w_ln),
                "sum(-1)": lambda: x.sum(-1),
                "cast bf16->fp32": lambda: x.float(),
                "clone": lambda: x.clone(),
                "transpose(-2,-1)": lambda: x.t().contiguous(),
                "permute (0,2,1) 3-D": lambda: x.view(r // 32, 32, c)
                .permute(0, 2, 1)
                .contiguous(),
                "slice rows [:r/2]": lambda: x[: r // 2].contiguous(),
                "slice cols [:, :c/2]": lambda: x[:, : c // 2].contiguous(),
                "cat dim 0": lambda: torch.cat([x, y], 0),
                "cat dim 1": lambda: torch.cat([x, y], 1),
                "repeat (2,1) of half": lambda: x[: r // 2].repeat(2, 1),
                "gather rows (index_select)": lambda: x.index_select(0, idx),
                "broadcast add (row)": lambda: x + y[:1],
                "reshape split last dim": lambda: x.view(r, c // 64, 64).contiguous(),
                "tiny add 32x32 (per program)": lambda: tiny + tiny,
            }
        )
        return ops

    def programs():
        return kohakutt.counters()["ops"]

    return make, sync, programs


def ttnn_stack():
    """(ops {name: fn}, sync, programs) on raw TT-NN device 0, same configs as ours."""
    # Optional dependency: TT-NN's Python package (the ttsrc venv).
    import ttnn

    dev = ttnn.open_device(device_id=0)
    hifi4 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    def up(t):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev
        )

    def sync():
        ttnn.synchronize_device(dev)

    def make(n_vec, dims):
        r, c = n_vec // 4096, 4096
        x, y = up(torch.randn(r, c)), up(torch.randn(r, c))
        idx = ttnn.from_torch(
            torch.randperm(r).reshape(1, r).int(), dtype=ttnn.uint32, device=dev
        )
        x_rm = ttnn.from_torch(torch.randn(r, c), dtype=ttnn.bfloat16, device=dev)
        w_ln = up(torch.ones(1, c))
        tiny = up(torch.randn(32, 32))
        mid = up(torch.randn(1024, 1024))
        x3 = ttnn.reshape(x, (r // 32, 32, c))
        half = ttnn.slice(x, (0, 0), (r // 2, c))
        row = ttnn.slice(y, (0, 0), (1, c))
        ops = {
            "tiny exp 32x32 (per program)": lambda: ttnn.exp(tiny),
            "mid add 1024 tiles (per program)": lambda: ttnn.add(mid, mid),
            "mid exp 1024 tiles (per program)": lambda: ttnn.exp(mid),
        }
        for n in dims:
            a, b = up(torch.randn(n, n)), up(torch.randn(n, n))
            ops[f"mm {n}x{n}x{n}"] = lambda a=a, b=b: ttnn.matmul(
                a, b, compute_kernel_config=hifi4
            )
        for m, k, n in [(32, 4096, 16384), (1, 4096, 16384), (32, 896, 4864)]:
            xi, w = up(torch.randn(m, k)), up(torch.randn(n, k))
            ops[f"decode linear [{m},{k}]x[{n},{k}]^T"] = (
                lambda xi=xi, w=w: ttnn.linear(
                    xi, w, transpose_b=True, compute_kernel_config=hifi4
                )
            )
        ops.update(
            {
                "add": lambda: ttnn.add(x, y),
                "mul": lambda: ttnn.multiply(x, y),
                "exp": lambda: ttnn.exp(x),
                "gelu": lambda: ttnn.gelu(x, fast_and_approximate_mode=False),
                "silu": lambda: ttnn.silu(x),
                "softmax(-1)": lambda: ttnn.softmax(x, -1),
                "layer_norm(-1)": lambda: ttnn.layer_norm(x, weight=w_ln, bias=w_ln),
                "sum(-1)": lambda: ttnn.sum(x, -1),
                "cast bf16->fp32": lambda: ttnn.typecast(x, ttnn.float32),
                "clone": lambda: ttnn.clone(x),
                "transpose(-2,-1)": lambda: ttnn.transpose(x, -2, -1),
                "permute (0,2,1) 3-D": lambda: ttnn.permute(x3, (0, 2, 1)),
                "slice rows [:r/2]": lambda: ttnn.slice(x, (0, 0), (r // 2, c)),
                "slice cols [:, :c/2]": lambda: ttnn.slice(x, (0, 0), (r, c // 2)),
                "cat dim 0": lambda: ttnn.concat([x, y], 0),
                "cat dim 1": lambda: ttnn.concat([x, y], 1),
                "repeat (2,1) of half": lambda: ttnn.repeat(half, (2, 1)),
                "gather rows (index_select)": lambda: ttnn.embedding(idx, x_rm),
                "broadcast add (row)": lambda: ttnn.add(x, row),
                "reshape split last dim": lambda: ttnn.reshape(x, (r, c // 64, 64)),
                "tiny add 32x32 (per program)": lambda: ttnn.add(tiny, tiny),
            }
        )
        return ops

    def programs():
        return None

    return make, sync, programs


def measure(fn, sync, calls):
    """(median seconds per synced call over `calls` calls after 2 warm-ups, AICLK)."""
    for _ in range(2):
        fn()
    sync()
    samples = []
    with ClockSampler() as clock:
        for _ in range(calls):
            start = time.perf_counter()
            fn()
            sync()
            samples.append(time.perf_counter() - start)
    return statistics.median(samples), clock.mean()


def launch_cost(fn, sync, count):
    """(seconds per program over `count` back-to-back calls synced once, AICLK)."""
    for _ in range(20):
        fn()
    sync()
    with ClockSampler() as clock:
        start = time.perf_counter()
        for _ in range(count):
            fn()
        sync()
        seconds = (time.perf_counter() - start) / count
    return seconds, clock.mean()


def main():
    """Measures every case on one stack; prints one JSON line per case."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", choices=["kohakutt", "ttnn"], required=True)
    parser.add_argument("--elements", type=int, default=32 << 20)
    parser.add_argument("--dims", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--cores", type=int, default=110)
    args = parser.parse_args()

    make, sync, programs = (
        kohakutt_stack() if args.stack == "kohakutt" else ttnn_stack()
    )
    ops = make(args.elements, args.dims)
    for cls, name, work, kind in cases(args.elements, args.dims):
        record = {"stack": args.stack, "class": cls, "op": name}
        try:
            fn = ops[name]
            if kind == "launch":
                seconds, mhz = launch_cost(fn, sync, 1000)
                record.update(us_per_program=round(seconds * 1e6, 1), aiclk_mhz=mhz)
            else:
                seconds, mhz = measure(fn, sync, args.calls)
                record.update(ms=round(seconds * 1e3, 3), aiclk_mhz=mhz)
                if kind == "flops":
                    clock = (mhz or 1350) * 1e6
                    peak = FLOP_PER_CYCLE_LOFI * args.cores * clock / PASSES[FIDELITY]
                    record.update(
                        tflops=round(work / seconds / 1e12, 1),
                        of_peak=round(work / seconds / peak, 3),
                    )
                else:
                    gbps = work / seconds / 1e9
                    record.update(
                        gbps=round(gbps, 1), of_peak=round(gbps / DRAM_GBPS, 3)
                    )
            before = programs()
            if before is not None:
                fn()
                record["programs"] = programs() - before
        except Exception as error:  # noqa: BLE001  (a failing case is reported)
            record["error"] = str(error).splitlines()[0][:200]
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
