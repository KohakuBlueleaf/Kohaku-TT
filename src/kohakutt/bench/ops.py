"""Per-op benchmark of aten ops on tt: for each case (an op at a shape and dtype) the
cold time (first call at the shape), warm time (median of repeat calls), host CPU time
of the cold call, error statistics against float64 on the CPU next to the CPU
same-dtype result's (the precision baseline), and, for cases with a leading batch dim,
batch invariance (each row of the batched call against the row alone).

    python -m kohakutt.bench.ops                     # every case
    python -m kohakutt.bench.ops --ops add softmax   # cases whose name starts so
    python -m kohakutt.bench.ops --json out.json     # also write the records

Statistics are reported, not judged: abs and rel error max / median / mean and cosine
similarity (kohakutt.core.stats.diff_stats).
"""

import argparse
import json
import resource
import statistics
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

import kohakutt
from kohakutt.core.stats import diff_stats, format_stats


@dataclass
class Case:
    """An op at one shape: fn(*inputs) -> tensor; inputs from make(generator, dtype);
    `batched`: inputs[0] (and every input with the same leading size) carry a batch dim
    that the output keeps."""

    name: str
    fn: object
    make: object
    dtypes: tuple = (torch.float32, torch.bfloat16)
    batched: bool = False
    tags: tuple = field(default_factory=tuple)


def _rand(*shape, scale=1.0, positive=False):
    def make(gen, dtype):
        t = torch.randn(*shape, generator=gen) * scale
        return (t.abs() + 0.1 if positive else t).to(dtype)

    return make


def _inputs(*makers):
    return lambda gen, dtype: [m(gen, dtype) for m in makers]


def _cases():
    """The benchmark cases: elementwise, reductions, norms, matmuls, attention,
    convolution, pooling, data movement; small aligned, ragged and image-sized
    shapes."""
    cases = []
    shapes = {
        "aligned": (4, 256, 1024),
        "ragged": (3, 1043, 1000),
        "large": (1, 4096, 3072),
    }
    unary = {
        "exp": torch.exp,
        "log": torch.log,
        "sqrt": torch.sqrt,
        "rsqrt": torch.rsqrt,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "gelu": F.gelu,
        "silu": F.silu,
        "relu": F.relu,
        "abs": torch.abs,
        "neg": torch.neg,
        "sin": torch.sin,
        "cos": torch.cos,
        "reciprocal": torch.reciprocal,
        "square": torch.square,
    }
    positive = {"log", "sqrt", "rsqrt", "reciprocal"}
    for label, shape in shapes.items():
        for name, fn in unary.items():
            cases.append(
                Case(
                    f"{name}/{label}",
                    fn,
                    _inputs(_rand(*shape, positive=name in positive)),
                    batched=True,
                )
            )
        binary = {
            "add": torch.add,
            "sub": torch.sub,
            "mul": torch.mul,
            "div": torch.div,
            "maximum": torch.maximum,
            "minimum": torch.minimum,
        }
        for name, fn in binary.items():
            second = _rand(*shape, positive=name == "div")
            cases.append(
                Case(
                    f"{name}/{label}", fn, _inputs(_rand(*shape), second), batched=True
                )
            )
        row = _rand(1, 1, shape[-1])
        cases.append(
            Case(
                f"mul_row_bcast/{label}",
                torch.mul,
                _inputs(_rand(*shape), row),
                batched=True,
            )
        )
        cases.append(
            Case(
                f"addcmul/{label}",
                torch.addcmul,
                _inputs(_rand(*shape), _rand(*shape), row),
                batched=True,
            )
        )
        cases.append(
            Case(
                f"where/{label}",
                lambda a, b: torch.where(a > 0, a, b),
                _inputs(_rand(*shape), _rand(*shape)),
                batched=True,
            )
        )
        reductions = {
            "sum_last": lambda x: x.sum(-1),
            "mean_last": lambda x: x.mean(-1),
            "amax_last": lambda x: x.amax(-1),
            "var_last": lambda x: x.var(-1),
            "softmax": lambda x: F.softmax(x, -1),
            "log_softmax": lambda x: F.log_softmax(x, -1),
            "sum_rows": lambda x: x.sum(-2),
            "cumsum_last": lambda x: x.cumsum(-1),
            "argmax_last": lambda x: x.argmax(-1),
        }
        for name, fn in reductions.items():
            cases.append(
                Case(f"{name}/{label}", fn, _inputs(_rand(*shape)), batched=True)
            )
        d = shape[-1]
        cases.append(
            Case(
                f"layer_norm/{label}",
                lambda x, w, b: F.layer_norm(x, (x.shape[-1],), w, b),
                _inputs(_rand(*shape), _rand(d), _rand(d)),
                batched=True,
            )
        )
        cases.append(
            Case(
                f"rms_norm/{label}",
                lambda x, w: F.rms_norm(x, (x.shape[-1],), w),
                _inputs(_rand(*shape), _rand(d)),
                batched=True,
            )
        )
        cases.append(
            Case(
                f"linear/{label}",
                lambda x, w, b: F.linear(x, w, b),
                _inputs(_rand(*shape), _rand(1024, d, scale=d**-0.5), _rand(1024)),
                batched=True,
            )
        )
    for m, k, n in ((256, 256, 256), (1043, 1000, 777), (4096, 3072, 3072)):
        cases.append(
            Case(
                f"mm/{m}x{k}x{n}",
                torch.mm,
                _inputs(_rand(m, k), _rand(k, n, scale=k**-0.5)),
            )
        )
    cases.append(
        Case(
            "bmm/8x512x256x512",
            torch.bmm,
            _inputs(_rand(8, 512, 256), _rand(8, 256, 512, scale=1 / 16)),
            batched=True,
        )
    )
    for s in (279, 1043, 1615, 4379):
        qkv = [_rand(3, 24, s, 128) for _ in range(3)]
        cases.append(
            Case(
                f"sdpa/{s}",
                F.scaled_dot_product_attention,
                _inputs(*qkv),
                dtypes=(torch.bfloat16,),
                batched=True,
            )
        )
        cases.append(
            Case(
                f"sdpa_causal/{s}",
                lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True),
                _inputs(*qkv),
                dtypes=(torch.bfloat16,),
                batched=True,
            )
        )
    for c, h, w, o, k, s, p in (
        (32, 64, 64, 64, 3, 1, 1),
        (128, 100, 164, 128, 3, 1, 1),
        (64, 128, 128, 128, 3, 2, 1),
        (256, 50, 82, 256, 1, 1, 0),
    ):
        cases.append(
            Case(
                f"conv2d/c{c}_{h}x{w}_o{o}_k{k}s{s}",
                lambda x, wt, b, s=s, p=p: F.conv2d(x, wt, b, s, p),
                _inputs(
                    _rand(2, c, h, w),
                    _rand(o, c, k, k, scale=(c * k * k) ** -0.5),
                    _rand(o),
                ),
                dtypes=(torch.bfloat16,),
                batched=True,
            )
        )
    cases.append(
        Case(
            "group_norm/c192_200x328",
            lambda x, w, b: F.group_norm(x, 32, w, b),
            _inputs(_rand(2, 192, 200, 328), _rand(192), _rand(192)),
            batched=True,
        )
    )
    cases.append(
        Case(
            "max_pool2d/64x128x128",
            lambda x: F.max_pool2d(x, 2),
            _inputs(_rand(2, 64, 128, 128)),
            batched=True,
        )
    )
    cases.append(
        Case(
            "avg_pool2d/64x128x128",
            lambda x: F.avg_pool2d(x, 2),
            _inputs(_rand(2, 64, 128, 128)),
            batched=True,
        )
    )
    cases.append(
        Case(
            "interpolate_nearest/64x100x164",
            lambda x: F.interpolate(x, scale_factor=2.0),
            _inputs(_rand(2, 64, 100, 164)),
            batched=True,
        )
    )
    cases.append(
        Case(
            "pixel_shuffle/256x50x82",
            lambda x: F.pixel_shuffle(x, 2),
            _inputs(_rand(2, 256, 50, 82)),
            batched=True,
        )
    )
    movement = {
        "transpose_contiguous": lambda x: x.transpose(-1, -2).contiguous(),
        "permute_0213": lambda x: x.unsqueeze(1).permute(0, 2, 1, 3).contiguous(),
        "cat_last": lambda x: torch.cat((x, x), -1),
        "slice_ragged": lambda x: x[:, 7:-9, 3:-5].contiguous(),
        "repeat": lambda x: x.repeat(1, 2, 1),
        "pad": lambda x: F.pad(x, (3, 5, 1, 2)),
        "flip": lambda x: x.flip(-1),
    }
    for name, fn in movement.items():
        cases.append(
            Case(f"{name}/ragged", fn, _inputs(_rand(3, 1043, 1000)), batched=True)
        )
    cases.append(
        Case(
            "embedding/32000x1024",
            lambda i, w: F.embedding(i, w),
            lambda gen, dtype: [
                torch.randint(0, 32000, (4, 517), generator=gen),
                torch.randn(32000, 1024, generator=gen).to(dtype),
            ],
            batched=True,
        )
    )
    cases.append(
        Case(
            "index_select/rows",
            lambda x, i: x.index_select(1, i),
            lambda gen, dtype: [
                torch.randn(3, 1043, 1000, generator=gen).to(dtype),
                torch.randint(0, 1043, (300,), generator=gen),
            ],
            batched=True,
        )
    )
    cases.append(
        Case(
            "topk/1000",
            lambda x: x.topk(16, -1).values,
            _inputs(_rand(3, 517, 1000)),
            batched=True,
        )
    )
    cases.append(
        Case(
            "sort/1000",
            lambda x: x.sort(-1).values,
            _inputs(_rand(3, 517, 1000)),
            batched=True,
        )
    )
    return cases


def _cpu_s():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def _first(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def _run(case, dtype, dev, gen, warm_calls):
    """The record of one case at one dtype."""
    inputs = case.make(gen, dtype)
    ref64 = _first(
        case.fn(*[t.double() if t.is_floating_point() else t for t in inputs])
    )
    cpu = _first(case.fn(*inputs))
    device_inputs = [t.to(dev) for t in inputs]
    with torch.no_grad():
        kohakutt.synchronize(dev)
        wall, cpu_start = time.perf_counter(), _cpu_s()
        out = _first(case.fn(*device_inputs))
        kohakutt.synchronize(dev)
        cold, cold_cpu = time.perf_counter() - wall, _cpu_s() - cpu_start
        warm = []
        for _ in range(warm_calls):
            start = time.perf_counter()
            _first(case.fn(*device_inputs))
            kohakutt.synchronize(dev)
            warm.append(time.perf_counter() - start)
        result = out.cpu()
        record = {
            "case": case.name,
            "dtype": str(dtype).removeprefix("torch."),
            "cold_ms": cold * 1e3,
            "cold_host_cpu_ms": cold_cpu * 1e3,
            "warm_ms": statistics.median(warm) * 1e3,
            "tt_vs_f64": diff_stats(result, ref64),
            "cpu_vs_f64": diff_stats(cpu, ref64),
        }
        if case.batched and inputs[0].shape[0] > 1:
            lead = inputs[0].shape[0]
            worst = None
            for r in range(lead):
                rows = [
                    t[r : r + 1] if t.dim() and t.shape[0] == lead else t
                    for t in device_inputs
                ]
                single = _first(case.fn(*rows)).cpu()
                stats = diff_stats(result[r : r + 1], single)
                if worst is None or stats["abs_max"] > worst["abs_max"]:
                    worst = stats
            record["batch_vs_rows"] = worst
    return record


def main():
    """Runs the selected cases on device 0 and prints one block per case."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ops", nargs="*", help="case-name prefixes to run")
    parser.add_argument("--dtypes", nargs="*", help="dtype names to run")
    parser.add_argument("--warm-calls", type=int, default=5)
    parser.add_argument("--json", help="write the records here")
    args = parser.parse_args()
    dev = kohakutt.device(0)
    gen = torch.Generator().manual_seed(0)
    records = []
    for case in _cases():
        if args.ops and not any(case.name.startswith(p) for p in args.ops):
            continue
        for dtype in case.dtypes:
            name = str(dtype).removeprefix("torch.")
            if args.dtypes and name not in args.dtypes:
                continue
            try:
                record = _run(case, dtype, dev, gen, args.warm_calls)
            except Exception as e:  # noqa: BLE001 (reported per case)
                record = {
                    "case": case.name,
                    "dtype": name,
                    "error": str(e).splitlines()[0][:200],
                }
            records.append(record)
            if "error" in record:
                print(f"{case.name} [{name}] ERROR {record['error']}", flush=True)
                continue
            print(
                f"{case.name} [{name}] cold {record['cold_ms']:.2f} ms (host cpu "
                f"{record['cold_host_cpu_ms']:.2f}) warm {record['warm_ms']:.3f} ms\n"
                f"  tt  vs f64 {format_stats(record['tt_vs_f64'])}\n"
                f"  cpu vs f64 {format_stats(record['cpu_vs_f64'])}"
                + (
                    f"\n  batch vs rows {format_stats(record['batch_vs_rows'])}"
                    if "batch_vs_rows" in record
                    else ""
                ),
                flush=True,
            )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(records, f, indent=1)


if __name__ == "__main__":
    main()
