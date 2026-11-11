"""Flash attention written at the metal level (examples/kernels/dsl_metal/flash_km.py)
on the card: equal bit for bit to the backend's attention (its flash kernel route),
non-causal and causal, head dims 64 and 128, ragged and whole tile lengths."""

import functools
import importlib.util
from pathlib import Path

import pytest
import torch

EXAMPLE = Path(__file__).resolve().parents[3] / "examples/kernels/dsl_metal/flash_km.py"
# (B, H, S, D, causal): one and several key blocks, jobs and lanes, a ragged last tile.
CASES = [
    (1, 2, 96, 64, False),
    (1, 2, 300, 64, True),
    (2, 3, 1000, 64, True),
    (1, 2, 300, 128, False),
    (1, 4, 1000, 128, True),
    (1, 5, 777, 64, False),
]


@functools.cache
def _example():
    spec = importlib.util.spec_from_file_location("flash_km_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.hw
def test_flash_equals_the_backends_flash_kernel(native):
    module, dev = _example(), native.device(0)
    for seed, (b, h, s, d, causal) in enumerate(CASES):
        g = torch.Generator().manual_seed(seed)
        q, k, v = (torch.randn(b, h, s, d, generator=g).bfloat16() for _ in range(3))
        qt, kt, vt = (t.to(dev) for t in (q, k, v))
        before = native.counters()["attention_flash"]
        want = torch.nn.functional.scaled_dot_product_attention(
            qt, kt, vt, is_causal=causal
        )
        native.synchronize()
        assert native.counters()["attention_flash"] == before + 1
        got = module.flash.op(qt, kt, vt, causal)
        assert got.shape == want.shape and got.dtype == want.dtype
        got, want = got.cpu(), want.cpu()
        assert torch.equal(got, want), ((b, h, s, d, causal), (got - want).abs().max())
