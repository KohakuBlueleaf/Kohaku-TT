"""Per-aten-op profile of tt work: each op synchronized and timed, with the TT-NN
programs it launched; aggregated by overload.
    with OpProfile() as prof: step()
    print(prof.table())
"""

import time
from collections import defaultdict

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from kohakutt.device import counters, synchronize


class OpProfile(TorchDispatchMode):
    """Records {overload: [calls, wall ms, TT-NN ops, shapes, host transfers]} for tt
    aten calls; host transfers are uploads + downloads (each one blocks the host)."""

    def __init__(self):
        super().__init__()
        self.stats = defaultdict(lambda: [0, 0.0, 0, set(), 0])

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        tensors = [a for a in args if isinstance(a, torch.Tensor)]
        on_tt = any(t.device.type == "tt" for t in tensors)
        if not on_tt:
            return func(*args, **kwargs)
        synchronize()
        before = counters()
        start = time.perf_counter()
        out = func(*args, **kwargs)
        synchronize()
        elapsed = (time.perf_counter() - start) * 1e3
        after = counters()
        entry = self.stats[str(func)]
        entry[0] += 1
        entry[1] += elapsed
        entry[2] += after["ops"] - before["ops"]
        entry[4] += (after["uploads"] - before["uploads"]) + (
            after["downloads"] - before["downloads"]
        )
        if len(entry[3]) < 4:
            entry[3].add(tuple(tuple(t.shape) for t in tensors[:2]))
        return out

    def table(self, top=25):
        """The `top` overloads by total wall time, as aligned text rows."""
        rows = sorted(self.stats.items(), key=lambda kv: -kv[1][1])
        total = sum(v[1] for v in self.stats.values())
        lines = [
            f"total {total:.1f} ms over {sum(v[0] for v in self.stats.values())} calls"
        ]
        lines.append(
            f"{'overload':42s} {'calls':>6s} {'ms':>9s} {'%':>5s} {'ms/call':>8s} "
            f"{'progs/call':>10s} {'host xfer':>9s}"
        )
        for name, (calls, ms, ops, shapes, xfer) in rows[:top]:
            lines.append(
                f"{name[:42]:42s} {calls:6d} {ms:9.2f} {100 * ms / total:5.1f} "
                f"{ms / calls:8.3f} {ops / calls:10.1f} {xfer:9d}  {sorted(shapes)[:2]}"
            )
        return "\n".join(lines)
