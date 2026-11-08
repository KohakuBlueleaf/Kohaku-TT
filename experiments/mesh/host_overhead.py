"""Host time against device time of Kohaku-TT's own programs.

For each case (an aten call routed to one or more of Kohaku-TT's own programs, at a
size), after warm-up:

- host µs per program by phase, from kohakutt.counters(): output allocation, runtime
  arguments, the rest of the descriptor build (or the descriptor-cache lookup and
  patch), and generic_op's enqueue; also op_ns per aten call (all of the worker's
  TT-NN calls for it) and the Python thread's submission time per call;
- eager µs per call: `--calls` async calls, then one synchronize;
- device µs per call: a trace of `--trace-calls` calls replayed `--replays` times
  (trace activations live in DRAM; includes trace dispatch per program).

    python experiments/mesh/host_overhead.py                 # every case
    python experiments/mesh/host_overhead.py --cases unary   # cases starting so
    python experiments/mesh/host_overhead.py --json out.json --save outs.pt

--save writes each case's outputs (CPU; also for a second set of inputs in other
buffers) for a bit-identity check between processes, e.g. KTT_DESC_CACHE=0 against the
default: --compare a.pt b.pt.
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch.nn.functional as F

import kohakutt
from kohakutt import fusion


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def _randn(*shape, dtype=torch.bfloat16, seed=0):
    return torch.randn(*shape, generator=_gen(seed)).to(dtype)


def cases():
    """{name: (fn, make_inputs)}: make_inputs() returns CPU inputs; fn runs on tt
    tensors."""
    out = {}
    for tag, shape in (
        ("512t", (1, 512, 1024)),
        ("4kt", (4, 1024, 1024)),
        ("12kt", (1, 4096, 3072)),
    ):
        out[f"unary_exp_{tag}"] = (torch.exp, lambda s=shape: [_randn(*s)])
        out[f"cast_bf16_fp32_{tag}"] = (
            lambda x: x.float(),
            lambda s=shape: [_randn(*s)],
        )
        out[f"binary_row_add_{tag}"] = (
            torch.add,
            lambda s=shape: [_randn(*s), _randn(s[-1], seed=1)],
        )
        out[f"binary_head_mul_{tag}"] = (
            torch.mul,
            lambda s=shape: [_randn(4, *s[1:]), _randn(*s[1:], seed=1)],
        )
        out[f"fusion_chain_{tag}"] = (
            lambda x, b: fusion.run(
                [x, b],
                ["full", "row"],
                [["add", 0, 0, 1], ["silu", 0, 0, None]],
                0,
                list(x.shape),
                x.dtype,
            ),
            lambda s=shape: [_randn(*s), _randn(1, s[-1], seed=1)],
        )
    for s in (256, 1024, 4096):
        out[f"attention_causal_s{s}"] = (
            lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True),
            lambda s=s: [_randn(1, 8, s, 64, seed=i) for i in range(3)],
        )
    for m in (1024, 4096, 16384):
        out[f"matmul_m{m}_k1024_n1024"] = (
            F.linear,
            lambda m=m: [_randn(m, 1024), _randn(1024, 1024, seed=1) * 0.03],
        )
    for hw in (32, 64, 128):
        out[f"conv3x3_c64_hw{hw}"] = (
            lambda x, w: F.conv2d(x, w, padding=1),
            lambda hw=hw: [_randn(1, 64, hw, hw), _randn(64, 64, 3, 3, seed=1) * 0.05],
        )
        out[f"group_norm_c128_hw{hw}"] = (
            lambda x: F.group_norm(x, 32),
            lambda hw=hw: [_randn(1, 128, hw, hw)],
        )
        out[f"upsample_nearest2_c64_hw{hw}"] = (
            lambda x: F.interpolate(x, scale_factor=2, mode="nearest"),
            lambda hw=hw: [_randn(1, 64, hw, hw)],
        )
        out[f"pixel_shuffle2_c64_hw{hw}"] = (
            lambda x: F.pixel_shuffle(x, 2),
            lambda hw=hw: [_randn(1, 256, hw, hw)],
        )
    for n in (32, 512, 4096):
        out[f"embedding_fp32_n{n}"] = (
            F.embedding,
            lambda n=n: [
                torch.randint(0, 1000, (n,), generator=_gen(2)),
                _randn(1000, 512, dtype=torch.float32),
            ],
        )
    for r, w in ((128, 1024), (1024, 1024), (4096, 4096)):
        out[f"argmax_fp32_{r}x{w}"] = (
            lambda x: torch.argmax(x, dim=-1),
            lambda r=r, w=w: [_randn(r, w, dtype=torch.float32)],
        )
        out[f"cumsum_bf16_{r}x{w}"] = (
            lambda x: torch.cumsum(x, dim=-1),
            lambda r=r, w=w: [_randn(r, w)],
        )
    return out


def _delta(before, after):
    return {k: after[k] - before[k] for k in after}


def measure(fn, inputs, calls, trace_calls, replays):
    """Per-call and per-program numbers of fn(*inputs) on tt tensors."""
    for _ in range(3):
        fn(*inputs)
    kohakutt.synchronize()

    # Host phases: async calls, counters around them, then one synchronize.
    c0 = kohakutt.counters()
    t0 = time.perf_counter()
    for _ in range(calls):
        y = fn(*inputs)
    submit = time.perf_counter() - t0
    kohakutt.synchronize()
    wall = time.perf_counter() - t0
    c = _delta(c0, kohakutt.counters())
    programs = max(c["programs"], 1)
    rec = {
        "programs_per_call": c["programs"] / calls,
        "desc_hits_per_call": c["desc_hits"] / calls,
        "op_us_per_call": c["op_ns"] / calls / 1e3,
        "submit_us_per_call": submit / calls * 1e6,
        "eager_us_per_call": wall / calls * 1e6,
    }
    for phase in ("alloc", "args", "desc", "enqueue"):
        rec[f"{phase}_us_per_program"] = c[f"{phase}_ns"] / programs / 1e3
    rec["host_us_per_program"] = sum(
        rec[f"{p}_us_per_program"] for p in ("alloc", "args", "desc", "enqueue")
    )

    # Device time: a trace of trace_calls calls, replayed.
    def chain(*xs):
        r = None
        for _ in range(trace_calls):
            r = fn(*xs)
        return r

    step = kohakutt.capture(chain, *inputs)
    step(*inputs)
    kohakutt.synchronize()
    times = []
    for _ in range(replays):
        t0 = time.perf_counter()
        torch.ops.ktt.replay(step.trace_id)
        kohakutt.synchronize()
        times.append(time.perf_counter() - t0)
    step.release()
    rec["device_us_per_call"] = statistics.median(times) / trace_calls * 1e6
    return rec, y


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cases", nargs="*", default=None)
    p.add_argument("--calls", type=int, default=50)
    p.add_argument("--trace-calls", type=int, default=10)
    p.add_argument("--replays", type=int, default=10)
    p.add_argument("--json", default=None)
    p.add_argument("--save", default=None)
    p.add_argument("--compare", nargs=2, default=None)
    args = p.parse_args()

    if args.compare:
        a, b = (torch.load(f) for f in args.compare)
        for name in sorted(set(a) & set(b)):
            same = a[name].dtype == b[name].dtype and torch.equal(a[name], b[name])
            print(f"{name:34s} {'bit-identical' if same else 'DIFFERENT'}")
        return

    os.environ.setdefault("KTT_OWN_MATMUL_MIN_ROWS", "1")
    dev = kohakutt.device(0)
    print(f"KTT_DESC_CACHE={os.environ.get('KTT_DESC_CACHE', '1')}")
    records, outputs = {}, {}
    for name, (fn, make) in cases().items():
        if args.cases and not any(name.startswith(c) for c in args.cases):
            continue
        cpu = make()
        inputs = [x.to(dev) for x in cpu]
        rec, y = measure(fn, inputs, args.calls, args.trace_calls, args.replays)
        records[name] = rec
        outputs[name] = y.cpu()
        # Other values in other buffers through the same cached program.
        other = [(x * -0.5 if x.is_floating_point() else x).to(dev) for x in cpu]
        outputs[name + "/other"] = fn(*other).cpu()
        del other
        print(
            f"{name:34s} progs {rec['programs_per_call']:.0f} "
            f"host/prog {rec['host_us_per_program']:7.1f} us "
            f"(alloc {rec['alloc_us_per_program']:5.1f} "
            f"args {rec['args_us_per_program']:5.1f} "
            f"desc {rec['desc_us_per_program']:5.1f} "
            f"enq {rec['enqueue_us_per_program']:6.1f}) "
            f"op {rec['op_us_per_call']:7.1f} submit {rec['submit_us_per_call']:6.1f} "
            f"eager {rec['eager_us_per_call']:8.1f} "
            f"device {rec['device_us_per_call']:8.1f} us/call",
            flush=True,
        )
        del inputs, y
    if args.json:
        with open(args.json, "w") as f:
            json.dump(records, f, indent=1)
    if args.save:
        torch.save(outputs, args.save)


if __name__ == "__main__":
    main()
