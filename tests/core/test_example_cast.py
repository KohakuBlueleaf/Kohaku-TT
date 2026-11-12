"""The bf16 cast of the transformers examples (examples/08_hf_models/common.py) casts
floating tensors and leaves complex buffers and parameters as they were registered."""

import importlib.util
import warnings
from pathlib import Path

import torch
from torch import nn

COMMON = Path(__file__).parents[2] / "examples/08_hf_models/common.py"


def _to_bf16():
    spec = importlib.util.spec_from_file_location("hf_common", COMMON)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.to_bf16


class Rotary(nn.Module):
    def __init__(self, freqs=None):
        super().__init__()
        self.proj = nn.Linear(8, 8)
        angles = torch.arange(16.0).view(4, 4)
        if freqs is None:
            freqs = torch.polar(torch.ones(4, 4), angles)
        self.register_buffer("freqs", freqs)
        self.register_buffer("table", torch.polar(torch.ones(4), angles[0]), False)
        self.register_buffer("scale", torch.ones(4))
        self.phase = nn.Parameter(torch.polar(torch.ones(2), torch.ones(2)))


def test_to_bf16_keeps_complex_tensors():
    first = Rotary()
    model = nn.Sequential(first, Rotary(first.freqs))
    want = first.freqs.clone()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = _to_bf16()(model)
    a, b = model[0], model[1]
    assert a.freqs.dtype == torch.complex64 and torch.equal(a.freqs, want)
    assert b.freqs is a.freqs
    assert a.phase.dtype == torch.complex64 and isinstance(a.phase, nn.Parameter)
    assert a.proj.weight.dtype == torch.bfloat16 and a.scale.dtype == torch.bfloat16
    keys = model.state_dict().keys()
    assert "0.freqs" in keys and "0.table" not in keys
    assert a.table.dtype == torch.complex64
