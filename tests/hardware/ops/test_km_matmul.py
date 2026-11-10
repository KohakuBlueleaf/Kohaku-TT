"""The matmul written at the metal level (examples/kernels/dsl_metal/matmul_km.py) on
the card: equal bit for bit to the backend's own matmul program over a sweep of
shapes, with no kernel compiled after the first call and one program build per class
of core count and buffer sizes."""

import functools
import importlib.util
import os
from pathlib import Path

import pytest
import torch

# The backend runs its own matmul program from this many rows on (read at its first
# matmul; without it only the shapes of at least 16384 rows below take that program).
os.environ.setdefault("KTT_OWN_MATMUL_MIN_ROWS", "1")

EXAMPLE = (
    Path(__file__).resolve().parents[3] / "examples/kernels/dsl_metal/matmul_km.py"
)
# (a's shape, N, w stored [N, K]).
SWEEP = [
    ((16384, 768), 1024, False),
    ((16384, 384), 768, True),
    ((4096, 256), 512, False),
    ((4096, 256), 512, True),
    ((2, 2048, 512), 768, False),
    ((8192, 1024), 1024, True),
    ((16384, 384), 96, False),
    ((6000, 64), 1000, False),
    ((3, 1000, 320), 40, True),
    ((16384, 1024), 256, False),
    ((131072, 64), 32, False),
]


@functools.cache
def _example():
    spec = importlib.util.spec_from_file_location("matmul_km_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _counted(native, fn):
    """(result of fn(), counter increases over it), with the stream drained."""
    native.synchronize()
    before = native.counters()
    out = fn()
    native.synchronize()
    after = native.counters()
    return out, {name: after[name] - before[name] for name in after}


def _operands(a_shape, n, w_nk, seed):
    g = torch.Generator().manual_seed(seed)
    k = a_shape[-1]
    a = torch.randn(*a_shape, generator=g).bfloat16()
    w = (torch.randn((n, k) if w_nk else (k, n), generator=g) / 16).bfloat16()
    return a, w


@pytest.mark.hw
def test_matmul_equals_the_backend_and_sizes_compile_nothing(native):
    module, dev = _example(), native.device(0)
    op = torch.ops.kohakutt_example.matmul_km
    program = module.traced(*native.compute_grid())
    seen, columns = set(), set()
    for seed, (a_shape, n, w_nk) in enumerate(SWEEP):
        a, w = (t.to(dev) for t in _operands(a_shape, n, w_nk, seed))
        values = module.values_for(a, w, w_nk)
        key = (program.core_count(values), *program.sizes(values).values())
        got, grown = _counted(native, lambda a=a, w=w, w_nk=w_nk: op(a, w, w_nk))
        assert grown["programs"] == 1, a_shape
        if seen:
            assert grown["kernel_compiles"] == 0, a_shape
            # Not on the first launch: its count also sees what earlier tests freed.
            assert grown["program_entries"] == (key not in seen), a_shape
        seen.add(key)
        columns.add(values["nt"])
        want = torch.nn.functional.linear(a, w) if w_nk else a @ w
        assert got.shape == want.shape and got.dtype == torch.bfloat16
        assert torch.equal(got.cpu(), want.cpu()), (a_shape, n, w_nk)
        again, grown = _counted(native, lambda a=a, w=w, w_nk=w_nk: op(a, w, w_nk))
        assert grown["program_entries"] == grown["kernel_compiles"] == 0, a_shape
        assert torch.equal(again.cpu(), want.cpu()), a_shape
    assert columns == {1, 2, 4} and len(seen) == 9


@pytest.mark.hw
def test_matmul_refuses_what_the_program_does_not_take(native):
    _example()
    dev = native.device(0)
    op = torch.ops.kohakutt_example.matmul_km
    a = torch.randn(64, 100).bfloat16().to(dev)
    w = torch.randn(100, 64).bfloat16().to(dev)
    with pytest.raises(ValueError, match="K a multiple of 32"):
        op(a, w, False)
    with pytest.raises(TypeError, match="must be bfloat16"):
        op(a.float(), w.float(), False)
