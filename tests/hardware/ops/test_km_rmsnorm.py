"""RMS norm written at the metal level (examples/kernels/dsl_metal/rmsnorm_km.py) on
the card: equal bit for bit to the backend's own norm forward with one core per tile
row, with rows dealt out in turn and taken from the counter, with and without gamma."""

import functools
import importlib.util
import os
from pathlib import Path

import pytest
import torch

# The backend's norm with one core per tile row, as the example lays it out (read at
# the backend's first norm).
os.environ["KTT_NORM_GROUP"] = "1"

EXAMPLE = (
    Path(__file__).resolve().parents[3] / "examples/kernels/dsl_metal/rmsnorm_km.py"
)
# (x's shape, dtype, gamma's dtype or None): fewer and more tile rows than cores,
# ragged widths and rows.
CASES = [
    ((256, 768), torch.bfloat16, torch.bfloat16),
    ((4096, 768), torch.bfloat16, torch.bfloat16),
    ((2, 1000, 320), torch.bfloat16, None),
    ((300, 1000), torch.bfloat16, torch.bfloat16),
    ((4096, 2048), torch.bfloat16, None),
    ((512, 256), torch.float32, torch.float32),
    ((3, 77, 96), torch.float32, None),
]
NAMES = {torch.bfloat16: "bfloat16", torch.float32: "float32"}


@functools.cache
def _example():
    spec = importlib.util.spec_from_file_location("rmsnorm_km_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.hw
def test_rmsnorm_equals_the_backend(native):
    import kohakutt.dsl.metal as km

    module, dev = _example(), native.device(0)
    machine = km.Machine(*native.compute_grid())
    launchers, seen = {}, set()
    for seed, (shape, dtype, gamma_dtype) in enumerate(CASES):
        key = (NAMES[dtype], gamma_dtype and NAMES[gamma_dtype])
        if key not in launchers:
            launchers[key] = km.Launcher(module.rmsnorm_program(machine, *key))
        g = torch.Generator().manual_seed(seed)
        x = (torch.randn(*shape, generator=g) * 3).to(dtype).to(dev)
        gamma = None
        if gamma_dtype is not None:
            gamma = torch.randn(shape[-1], generator=g).to(gamma_dtype).to(dev)
        with torch.no_grad():
            want = torch.nn.functional.rms_norm(x, (shape[-1],), gamma, 1e-6)
        native.synchronize()
        before = native.counters()["kernel_compiles"]
        got = module.rmsnorm(launchers[key], x, 1e-6, gamma)
        native.synchronize()
        compiled = native.counters()["kernel_compiles"] - before
        # A new shape of a variant already run compiles nothing.
        assert compiled == 0 or key not in seen, (shape, compiled)
        seen.add(key)
        assert got.shape == want.shape and got.dtype == want.dtype
        got, want = got.cpu(), want.cpu()
        assert torch.equal(got, want), (shape, dtype, (got - want).abs().max())
