"""Regression: user Metalium programs (kohakutt.kernels, C++ extensions) on tt tensors.
The Python and C++ SwiGLU examples run the same kernels, so their outputs are equal."""

import functools
import importlib.util
import os
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
EXTENSION = Path(
    os.environ.get("KTT_SWIGLU_EXT", ROOT / "build" / "swiglu_ext" / "swiglu_ext.so")
)


@functools.cache
def _python_swiglu():
    """The kohakutt.kernels SwiGLU example module, loaded unless another test already
    registered its op."""
    if "kohakutt_example::swiglu" in torch._C._dispatch_get_all_op_names():
        return None
    path = ROOT / "examples" / "custom_kernel" / "swiglu.py"
    spec = importlib.util.spec_from_file_location("swiglu_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(dev, rows=1024, cols=4096):
    gen = torch.Generator().manual_seed(0)
    a = torch.randn(rows, cols, generator=gen).bfloat16()
    b = torch.randn(rows, cols, generator=gen).bfloat16()
    return a.to(dev), b.to(dev)


@pytest.mark.hw
def test_python_program_runs(native):
    _python_swiglu()
    a, b = _inputs(native.device(0))
    out = torch.ops.kohakutt_example.swiglu(a, b)
    assert out.shape == a.shape and out.dtype == torch.bfloat16
    again = torch.ops.kohakutt_example.swiglu(a, b)  # cached description
    assert torch.equal(out.cpu(), again.cpu())


@pytest.mark.hw
def test_cpp_extension_equals_python(native):
    if not EXTENSION.exists():
        pytest.skip("C++ example extension not built")
    import kohakutt.extension

    _python_swiglu()
    kohakutt.extension.load(EXTENSION)
    a, b = _inputs(native.device(0))
    python = torch.ops.kohakutt_example.swiglu(a, b).cpu()
    cpp = torch.ops.kohakutt_example_cpp.swiglu(a, b).cpu()
    assert torch.equal(python, cpp)
