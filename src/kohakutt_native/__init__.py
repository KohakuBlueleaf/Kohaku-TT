"""Kohaku-TT's compute / system layer: the compiled extension (TT-NN bridge, tt tensor
model, PrivateUse1 aten kernels, trace runtime). Loaded only by kohakutt; no user API.
"""

from pathlib import Path

import torch

LIBRARY = Path(__file__).with_name("_ktt_native.so")


def load():
    """Loads the extension into torch (idempotent); False when it is not built."""
    if not LIBRARY.exists():
        return False
    if not hasattr(torch.ops.ktt, "open"):
        torch.ops.load_library(str(LIBRARY))
    return True
