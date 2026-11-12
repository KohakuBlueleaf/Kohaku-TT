"""The GDN decode step at the metal level (examples/kernels/dsl_metal/gdn_decode_km.py)
on the card, four steps with the state carried: the state within fp32 rounding of
float64, o within bf16 rounding, both no worse than transformers' step on the CPU."""

import functools
import importlib.util
from pathlib import Path

import pytest

EXAMPLE = (
    Path(__file__).resolve().parents[3] / "examples/kernels/dsl_metal/gdn_decode_km.py"
)


@functools.cache
def _example():
    spec = importlib.util.spec_from_file_location("gdn_decode_km_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.hw
@pytest.mark.parametrize(
    "seqs, heads", [(1, (16, 32)), (1, (16, 48)), (2, (16, 32))], ids=str
)
def test_steps_match_float64(native, seqs, heads):
    errors = _example().compare((seqs, *heads, 128, 128, 4))
    for step in range(4):
        # fp32 state: a few roundings per step; o: one bf16 rounding (2^-8).
        assert errors["state"][step] < 1e-6, errors
        assert errors["o"][step] <= 2**-8, errors
        assert errors["state"][step] <= errors["hf state"][step], errors
        assert errors["o"][step] <= errors["hf o"][step] + 2**-9, errors
