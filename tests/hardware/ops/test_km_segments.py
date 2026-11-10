"""The job-list program at the metal level (examples/kernels/dsl_metal/segments_km.py)
on the card: equal bit for bit to the backend's cat of the same slices and to the
host's, with a new job list per launch rewriting the per-core tables and building
nothing."""

import functools
import importlib.util
from pathlib import Path

import pytest
import torch

EXAMPLE = (
    Path(__file__).resolve().parents[3] / "examples/kernels/dsl_metal/segments_km.py"
)


@functools.cache
def _example():
    spec = importlib.util.spec_from_file_location("segments_km_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _segments(seed, count, rows, total=None):
    """`count` random (first tile row, tile rows) of a tensor of `rows` tile rows;
    with `total`, sizes summing to it."""
    g = torch.Generator().manual_seed(seed)
    if total is None:
        sizes = torch.randint(1, 40, (count,), generator=g).tolist()
    else:
        cuts = sorted(
            torch.randperm(total - 1, generator=g)[: count - 1].add(1).tolist()
        )
        sizes = [b - a for a, b in zip([0, *cuts], [*cuts, total], strict=True)]
    firsts = [int(torch.randint(0, rows - n + 1, (1,), generator=g)) for n in sizes]
    return list(zip(firsts, sizes, strict=True))


def _counted(native, fn):
    native.synchronize()
    before = native.counters()
    out = fn()
    native.synchronize()
    after = native.counters()
    return out, {name: after[name] - before[name] for name in after}


@pytest.mark.hw
def test_segments_equal_cat_and_new_job_lists_build_nothing(native):
    _example()
    dev = native.device(0)
    op = torch.ops.kohakutt_example.segments_km
    rows = 600
    x = torch.randn(
        rows * 32, 32, generator=torch.Generator().manual_seed(9)
    ).bfloat16()
    xt = x.to(dev)
    cases = [
        _segments(0, 7, rows),
        _segments(1, 300, rows),
        _segments(2, 1600, rows, total=4000),
        _segments(3, 1500, rows, total=4000),
        [(0, rows), (0, 1)],
    ]
    for i, segs in enumerate(cases):
        flat = [v for pair in segs for v in pair]
        got, grown = _counted(native, lambda flat=flat: op(xt, flat))
        if i == 3:
            # The same sizes as the case before, other jobs on every core.
            assert grown["desc_hits"] == 1 and grown["kernel_compiles"] == 0
        want = torch.cat([xt[32 * f : 32 * (f + n)] for f, n in segs]).cpu()
        host = torch.cat([x[32 * f : 32 * (f + n)] for f, n in segs])
        got = got.cpu()
        assert torch.equal(got, want), (i, len(segs))
        assert torch.equal(got, host), (i, len(segs))
