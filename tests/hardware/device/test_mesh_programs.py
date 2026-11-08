"""Regression: own and user Metalium programs on a 1x2 mesh run on both chips (the two
replicas of each output are equal)."""

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.hw
@pytest.mark.dual
def test_mesh_replicas_equal(native_module):
    kohakutt = native_module
    if "kohakutt_example::swiglu" not in torch._C._dispatch_get_all_op_names():
        path = ROOT / "examples" / "custom_kernel" / "swiglu.py"
        spec = importlib.util.spec_from_file_location("swiglu_example_mesh", path)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
    mesh = kohakutt.mesh([0, 1], shape=(1, 2))
    try:
        gen = torch.Generator().manual_seed(0)
        a, b = (torch.randn(1024, 4096, generator=gen).bfloat16() for _ in range(2))
        x = torch.randn(2048, 4096, generator=gen)
        a, b, x = (kohakutt.replicate(t.to(mesh)) for t in (a, b, x))
        outs = {
            "kohakutt.kernels swiglu": torch.ops.kohakutt_example.swiglu(a, b),
            "ktt_unary exp": torch.exp(x),
            "ktt_binary row broadcast": x * x[:1],
        }
        for name, out in outs.items():
            first, second = (torch.ops.ktt.replica(out, chip) for chip in (0, 1))
            assert torch.equal(first, second), name
    finally:
        kohakutt.close()
