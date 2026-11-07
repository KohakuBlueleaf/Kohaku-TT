"""Writes tests/hardware/golden/<case>.pt from the current build (device 0). Run only
after the numerics of the build were checked as stats against the CPU baseline:
    python tests/hardware/golden/make_goldens.py [case ...]
"""

import sys
from pathlib import Path

import torch
from golden_cases import CASES

import kohakutt

HERE = Path(__file__).parent


def main():
    """Runs the named (default: all) cases and saves their outputs."""
    kohakutt.device(0)
    for name in sys.argv[1:] or CASES:
        torch.save(CASES[name](), HERE / f"{name}.pt")
        print("wrote", name, flush=True)


if __name__ == "__main__":
    main()
