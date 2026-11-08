"""Where the time goes in MLP stacks (4 blocks of Linear d -> 4d -> d, GELU, residual,
bf16) on 1 card or a 1x2 mesh: each aten op of one forward timed with a synchronize
after it (median of 3 forwards, after 3 untimed), grouped by op, with the mesh
conversions each op triggered; the async warm forward beside it.

    python experiments/mesh/mesh_profile.py --cards 1
    python experiments/mesh/mesh_profile.py --cards 2
"""

import argparse
import collections
import statistics
import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import kohakutt

CONVERSIONS = (
    "all_gathers",
    "reduce_scatters",
    "all_reduces",
    "partitions",
    "all_to_alls",
)


class MLP(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = torch.nn.Linear(dim, 4 * dim)
        self.down = torch.nn.Linear(4 * dim, dim)

    def forward(self, x):
        return x + self.down(torch.nn.functional.gelu(self.up(x)))


class OpTimer(TorchDispatchMode):
    """Synchronous time (ms) and mesh conversions per aten op name."""

    def __init__(self, mesh):
        super().__init__()
        self.mesh = mesh
        self.ms = collections.defaultdict(float)
        self.calls = collections.Counter()
        self.conv = collections.defaultdict(collections.Counter)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        before = torch.tt.mesh_stats() if self.mesh else {}
        torch.tt.synchronize()
        start = time.perf_counter()
        out = func(*args, **(kwargs or {}))
        torch.tt.synchronize()
        name = str(func).removeprefix("aten.")
        self.ms[name] += (time.perf_counter() - start) * 1e3
        self.calls[name] += 1
        if self.mesh:
            after = torch.tt.mesh_stats()
            for k in CONVERSIONS:
                if after[k] - before[k]:
                    self.conv[name][k] += after[k] - before[k]
        return out


def profile(model, x, mesh):
    runs = []
    with torch.no_grad():
        for _ in range(3):
            model(x)
        torch.tt.synchronize()
        for _ in range(3):
            timer = OpTimer(mesh)
            with timer:
                model(x)
            runs.append(timer)
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            model(x)
            torch.tt.synchronize()
            samples.append((time.perf_counter() - start) * 1e3)
    return runs, statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    mesh = args.cards == 2
    dev = kohakutt.mesh([0, 1], shape=(1, 2)) if mesh else kohakutt.device(0)
    try:
        g = torch.Generator().manual_seed(0)
        for dim, rows in ((1024, 256), (2048, 1024), (4096, 2048)):
            torch.manual_seed(0)
            model = torch.nn.Sequential(*[MLP(dim) for _ in range(4)])
            model = model.to(torch.bfloat16).to(dev)
            x = torch.randn(rows, dim, generator=g, dtype=torch.bfloat16).to(dev)
            runs, wall = profile(model, x, mesh)
            names = runs[0].ms.keys()
            per_op = {n: statistics.median(r.ms[n] for r in runs) for n in names}
            total = sum(per_op.values())
            print(
                f"{args.cards} card(s) d={dim} rows={rows}: "
                f"async warm forward {wall:.2f} ms; "
                f"synchronous per-op sum {total:.2f} ms",
                flush=True,
            )
            for n, ms in sorted(per_op.items(), key=lambda kv: -kv[1]):
                conv = dict(runs[0].conv.get(n, {}))
                print(
                    f"    {n:42s} {runs[0].calls[n]:3d} calls {ms:8.2f} ms"
                    f" ({100 * ms / total:4.1f} %) {conv if conv else ''}",
                    flush=True,
                )
            del model, x
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
