"""torch.compile on a Tenstorrent chip: the same training step eager and compiled;
each compiled graph is captured once and replayed as a device trace.
"""

import time

import torch
import torch.nn as nn

import kohakutt


def step_time(step, dev, calls=20):
    """Mean seconds per call of step() after warm-up, synchronized on `dev`."""
    for _ in range(5):
        step()
    kohakutt.synchronize(dev)
    start = time.perf_counter()
    for _ in range(calls):
        step()
    kohakutt.synchronize(dev)
    return (time.perf_counter() - start) / calls


def main():
    """Times an MLP training step eager and with backend="ktt_tt_aot"."""
    dev = kohakutt.device(0)
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.GELU(),
        nn.Linear(1024, 1024),
        nn.GELU(),
        nn.Linear(1024, 10),
    ).to(dev)
    x = torch.randn(64, 512, device=dev)
    y = torch.randint(0, 10, (64,)).to(dev)
    loss_fn = nn.CrossEntropyLoss()

    # "ktt_tt" compiles Dynamo graphs (inference); "ktt_tt_aot" also the backward.
    compiled = torch.compile(model, backend="ktt_tt_aot")

    def eager_step():
        model.zero_grad()
        loss_fn(model(x), y).backward()

    def compiled_step():
        model.zero_grad()
        loss_fn(compiled(x), y).backward()

    eager = step_time(eager_step, dev)
    fast = step_time(compiled_step, dev)
    print(
        f"eager {eager * 1e3:.2f} ms/step, compiled {fast * 1e3:.2f} ms/step "
        f"({eager / fast:.1f}x)"
    )


if __name__ == "__main__":
    main()
