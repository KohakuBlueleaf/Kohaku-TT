"""Experiment: torch.compile on tt: inference and training, compiled vs eager.

Inference ("ktt_tt"): outputs on new inputs vs eager, ms per call. Training
("ktt_tt_aot" + eager AdamW): loss curve vs eager training, ms per step.
    python -m kohakutt.bench.capture.compile_tt_check [--steps 10]
"""

import argparse
import json
import time

import torch
import torch._dynamo

import kohakutt
from kohakutt.bench.models import train_check
from kohakutt.torch_backend import compiler as tt_compile


def make_model(kind="mlp"):
    """Seeded bf16 model on tt: "mlp" (64 -> 128 -> 10) or "block" (transformer)."""
    return train_check.make_model(kind).to("tt")


def timed_ms(fn, repeat):
    """Mean ms of fn() over `repeat` calls, device-synchronized."""
    torch.tt.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.tt.synchronize()
    return (time.perf_counter() - start) * 1e3 / repeat


def inference():
    """Record for a compiled forward vs eager on new inputs."""
    model = make_model()
    compiled = torch.compile(model, backend="ktt_tt")
    gen = torch.Generator().manual_seed(1)
    # The first EAGER_CALLS calls run eagerly; the rest replay the trace.
    xs = [torch.randn(32, 64, generator=gen).bfloat16().to("tt") for _ in range(5)]
    with torch.no_grad():
        got = [compiled(x).cpu() for x in xs]
        ref = [model(x).cpu() for x in xs]
        eager_ms = timed_ms(lambda: model(xs[0]), 20)
        compiled_ms = timed_ms(lambda: compiled(xs[0]), 20)
    return {
        "eq_eager": all(torch.equal(a, b) for a, b in zip(got, ref)),
        "max_abs_diff": max(
            (a.float() - b.float()).abs().max().item() for a, b in zip(got, ref)
        ),
        "eager_ms": round(eager_ms, 3),
        "compiled_ms": round(compiled_ms, 3),
    }


def train(compile_backend, steps, kind="mlp"):
    """(losses, ms per step) of AdamW training, the model optionally compiled."""
    model = make_model(kind)
    run = torch.compile(model, backend=compile_backend) if compile_backend else model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x, y = (
        t.to("tt") for t in train_check.batch(kind, torch.Generator().manual_seed(2))
    )
    losses = []

    def step():
        opt.zero_grad()
        logits = run(x)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, -2), y.flatten())
        loss.backward()
        opt.step()
        losses.append(loss.detach())

    step()
    ms = timed_ms(step, steps)
    return [v.item() for v in losses], ms


def main():
    """Runs inference and training checks on device 0; prints one JSON record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--model", choices=["mlp", "block", "cnn"], default="mlp")
    args = parser.parse_args()
    tt_compile.register()
    kohakutt.open(0)
    try:
        record = {"inference": inference()}
        torch._dynamo.reset()
        eager, eager_ms = train(None, args.steps, args.model)
        compiled, compiled_ms = train("ktt_tt_aot", args.steps, args.model)
        dev = [abs(a - b) / max(abs(b), 1e-6) for a, b in zip(compiled, eager)]
        record["training"] = {
            "eager_loss": [round(v, 4) for v in eager[:: max(1, len(eager) // 5)]],
            "compiled_loss": [
                round(v, 4) for v in compiled[:: max(1, len(compiled) // 5)]
            ],
            "max_rel_dev": round(max(dev), 5),
            "eager_ms_per_step": round(eager_ms, 2),
            "compiled_ms_per_step": round(compiled_ms, 2),
            "fallbacks": kohakutt.fallback_counts(),
            "unsafe_buffers_max": max(
                (max(t.unsafe, default=0) for t in tt_compile.CAPTURED), default=0
            ),
        }
        tt_compile.release()
    finally:
        kohakutt.close()
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
