"""Regression: the tt tensor model (gate checks, view chains, dtypes)."""

import random

import pytest
import torch

from kohakutt.bench.device.tensor_gate import checks
from kohakutt.bench.device.view_chains import run_chain


@pytest.mark.hw
@pytest.mark.parametrize("index", range(9))
def test_gate_check(native, index):
    name, fn = checks()[index]
    assert fn(), name


@pytest.mark.hw
def test_view_chains_never_wrong(native):
    rng, gen = random.Random(1), torch.Generator().manual_seed(1)
    outcomes = []
    for _ in range(60):
        shape = [rng.choice([1, 3, 5, 32, 40, 64]) for _ in range(rng.randint(1, 4))]
        outcomes.append(run_chain(rng, torch.randn(*shape, generator=gen).bfloat16()))
    assert not [r for r in outcomes if r["outcome"] == "wrong"]
    assert sum(r["outcome"] == "ok" for r in outcomes) >= 55
