"""Regression: numeric kernels reproduce their recorded golden outputs bit for bit (any
change is a numerics change to re-check as stats before regenerating the goldens)."""

from pathlib import Path

import pytest
import torch
from golden_cases import CASES

HERE = Path(__file__).parent


@pytest.mark.hw
@pytest.mark.parametrize("name", sorted(CASES))
def test_golden(native, name):
    path = HERE / f"{name}.pt"
    if not path.exists():
        pytest.skip(f"no golden for {name} (run make_goldens.py)")
    want = torch.load(path)
    got = CASES[name]()
    assert want.keys() == got.keys()
    changed = [key for key in want if not torch.equal(want[key], got[key])]
    assert not changed, f"{name}: outputs changed: {changed}"
