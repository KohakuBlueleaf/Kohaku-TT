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


def _launched(native, fn):
    """(output, counters grown by fn())."""
    native.synchronize()
    before = native.counters()
    out = fn()
    native.synchronize()
    after = native.counters()
    return out, {name: after[name] - before[name] for name in after}


@pytest.mark.hw
def test_remembered_call_follows_the_numerics_scope(native):
    _python_swiglu()
    a, b = _inputs(native.device(0), rows=64, cols=64)
    swiglu = torch.ops.kohakutt_example.swiglu
    first = swiglu(a, b).cpu()
    again, grown = _launched(native, lambda: swiglu(a, b))
    assert (grown["programs"], grown["desc_hits"]) == (1, 1)
    assert grown["alloc_ns"] > 0
    assert torch.equal(first, again.cpu())
    with native.numerics(fidelity="HiFi4"):
        _, grown = _launched(native, lambda: swiglu(a, b))
        assert (grown["programs"], grown["desc_hits"]) == (1, 0)
        _, grown = _launched(native, lambda: swiglu(a, b))
        assert (grown["programs"], grown["desc_hits"]) == (1, 1)
    out, grown = _launched(native, lambda: swiglu(a, b))
    assert (grown["programs"], grown["desc_hits"]) == (1, 1)
    assert torch.equal(first, out.cpu())
    # Other sizes are another call of the same op.
    c, d = _inputs(native.device(0), rows=96, cols=64)
    assert swiglu(c, d).shape == c.shape
    assert torch.equal(first, swiglu(a, b).cpu())


@pytest.mark.hw
def test_only_the_ops_own_launch_is_remembered(native):
    module = _python_swiglu()
    if module is None:
        pytest.skip("the SwiGLU example was loaded by another test module")
    import kohakutt.kernels as kk

    @kk.op("kohakutt_test::swiglu_after_another(Tensor a, Tensor b) -> Tensor")
    def after_another(a, b):
        module.swiglu(b, a).run()  # a raw program before the op's own
        return module.swiglu(a, b)

    a, b = _inputs(native.device(0), rows=64, cols=64)
    expected = torch.ops.kohakutt_example.swiglu(a, b).cpu()
    assert not torch.equal(expected, torch.ops.kohakutt_example.swiglu(b, a).cpu())
    op = torch.ops.kohakutt_test.swiglu_after_another
    assert torch.equal(op(a, b).cpu(), expected)
    out, grown = _launched(native, lambda: op(a, b))
    assert grown["programs"] == 1
    assert torch.equal(out.cpu(), expected)


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
