"""Helpers of the tile-level examples: timing, counter deltas, and loading the
metal-level examples they are compared with."""

import functools
import importlib.util
import time
from pathlib import Path

import kohakutt

HERE = Path(__file__).parent


@functools.cache
def metal_example(name):
    """The module `examples/kernels/dsl_metal/<name>.py` (loaded once)."""
    path = HERE.parent / "dsl_metal" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"km_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timed(fn, reps=20, rounds=7):
    """The best over `rounds` of the mean seconds per call of `fn` over `reps` calls,
    after one warm-up call."""
    fn()
    kohakutt.synchronize()
    best = float("inf")
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(reps):
            fn()
        kohakutt.synchronize()
        best = min(best, (time.perf_counter() - start) / reps)
    return best


def counted(fn):
    """(fn(), {counter: increase}) with the stream drained before and after."""
    kohakutt.synchronize()
    before = kohakutt.counters()
    out = fn()
    kohakutt.synchronize()
    after = kohakutt.counters()
    return out, {k: after[k] - before[k] for k in after}


def builds(grown):
    """(kernel compiles, program builds) of a counter delta."""
    return grown["kernel_compiles"], grown["program_entries"]
