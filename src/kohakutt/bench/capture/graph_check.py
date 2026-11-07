"""Experiment: native trace: replay vs eager for silu(x @ w) + x, and a whole
SGD-momentum training step (MLP or CNN): loss curves and ms per step.
    python -m kohakutt.bench.capture.graph_check [--steps 20] [--model mlp|cnn]
"""

import argparse
import json
import time

import torch

import kohakutt
from kohakutt import capture
from kohakutt.bench.models.train_check import batch, make_cnn

WARMUP = 3


def small():
    """Record for a captured silu(x @ w) + x on two new inputs."""
    gen = torch.Generator().manual_seed(0)
    w = torch.randn(64, 64, generator=gen).bfloat16().to("tt")
    xs = [torch.randn(32, 64, generator=gen).bfloat16().to("tt") for _ in range(3)]

    def fn(x):
        return torch.nn.functional.silu(x @ w) + x

    step = capture(fn, xs[0])
    replayed = [step(x).cpu() for x in xs[1:]]
    eager = [fn(x).cpu() for x in xs[1:]]
    return {
        "replay_eq_eager": all(torch.equal(a, b) for a, b in zip(replayed, eager)),
        "held": step.held,
    }


def make_training(kind="mlp"):
    """(model, optimizer, x, y, step fn) of a seeded MLP or CNN with SGD momentum on
    tt."""
    torch.manual_seed(0)
    if kind == "cnn":
        model = make_cnn().bfloat16().to("tt")
    else:
        model = (
            torch.nn.Sequential(
                torch.nn.Linear(64, 128), torch.nn.SiLU(), torch.nn.Linear(128, 10)
            )
            .bfloat16()
            .to("tt")
        )
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, foreach=False)
    x, y = (t.to("tt") for t in batch(kind, torch.Generator().manual_seed(1)))

    def step(x, y):
        opt.zero_grad(set_to_none=False)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        return loss.detach()

    return model, opt, x, y, step


def training(steps, kind="mlp"):
    """Record comparing `steps` captured training steps of `kind` with the same steps
    eager."""
    # Eager: WARMUP steps + `steps` steps (capture() runs WARMUP eager steps first).
    _, _, x, y, eager_step = make_training(kind)
    for _ in range(WARMUP):
        eager_step(x, y)
    torch.tt.synchronize()
    start = time.perf_counter()
    eager = [eager_step(x, y).item() for _ in range(steps)]
    torch.tt.synchronize()
    eager_ms = (time.perf_counter() - start) * 1e3 / steps

    # Captured: same seed; capture's warm-up steps match the eager ones.
    _, _, x, y, graph_step = make_training(kind)
    step = capture(graph_step, x, y, warmup=WARMUP)
    torch.tt.synchronize()
    start = time.perf_counter()
    # Each loss is read right after its replay (outputs live until the next replay).
    replayed = [step(x, y).item() for _ in range(steps)]
    torch.tt.synchronize()
    replay_ms = (time.perf_counter() - start) * 1e3 / steps

    dev = [abs(a - b) / max(abs(b), 1e-6) for a, b in zip(replayed, eager)]
    return {
        "steps": steps,
        "eager_loss": [round(v, 4) for v in eager[:: max(1, steps // 5)]],
        "replay_loss": [round(v, 4) for v in replayed[:: max(1, steps // 5)]],
        "max_rel_dev": round(max(dev), 5),
        "loss_decreases": replayed[-1] < replayed[0],
        "eager_ms_per_step": round(eager_ms, 2),
        "replay_ms_per_step": round(replay_ms, 2),
        "held": step.held,
    }


def main():
    """Runs both checks on device 0 and prints one JSON record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--part", choices=["small", "training", "all"], default="all")
    parser.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    args = parser.parse_args()
    kohakutt.open(0)
    try:
        record = {}
        if args.part in ("small", "all"):
            record["small"] = small()
        if args.part in ("training", "all"):
            record["training"] = training(args.steps, args.model)
    finally:
        kohakutt.close()
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
