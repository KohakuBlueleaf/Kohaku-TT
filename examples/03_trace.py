"""Capture a whole training step once and replay it as one device trace (the
torch.cuda graphs counterpart; contract in docs/design/trace.md).
Usage: python examples/03_trace.py
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import kohakutt


def make_step(dev):
    """A seeded MLP on `dev` with SGD + momentum, and its training step (returns the
    loss)."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(256, 512), nn.SiLU(), nn.Linear(512, 10)).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    def step(x, y):
        opt.zero_grad(
            set_to_none=False
        )  # keep grad buffers: a trace needs fixed tensors
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        return loss.detach()

    return step


def timed(step, x, y, steps):
    """(losses, ms per step); each loss is read right after its step."""
    kohakutt.synchronize(x.device)
    start = time.perf_counter()
    losses = [step(x, y).item() for _ in range(steps)]
    return losses, (time.perf_counter() - start) * 1e3 / steps


def main():
    """Runs the same steps eagerly and by trace replay; both lose the same way."""
    dev = kohakutt.device(0)
    x = torch.randn(128, 256).to(dev)
    y = torch.randint(0, 10, (128,)).to(dev)

    eager_step = make_step(dev)
    for _ in range(3):  # the same 3 warm-up steps capture() runs
        eager_step(x, y)
    eager, eager_ms = timed(eager_step, x, y, 20)

    replay = kohakutt.capture(make_step(dev), x, y, warmup=3)
    traced, trace_ms = timed(replay, x, y, 20)

    print(f"eager  {eager_ms:.2f} ms/step  losses {eager[0]:.4f} -> {eager[-1]:.4f}")
    print(f"traced {trace_ms:.2f} ms/step  losses {traced[0]:.4f} -> {traced[-1]:.4f}")
    print("same losses:", all(abs(a - b) < 1e-6 for a, b in zip(eager, traced)))
    replay.release()


if __name__ == "__main__":
    main()
