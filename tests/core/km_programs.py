"""The example metal-level programs (examples/kernels/dsl_metal/*_km.py), loaded once
for the CPU tests."""

import functools
import importlib.util
from pathlib import Path

import kohakutt.dsl.metal as km

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "kernels" / "dsl_metal"
MACHINE = km.Machine(cols=11, rows=10)
ROPE_DTYPES = {"x": "bfloat16", "table": "float32", "out": "bfloat16"}
SWIGLU_DTYPES = {"a": "bfloat16", "b": "bfloat16", "out": "bfloat16"}


@functools.cache
def example(name):
    """The example module `name` (its torch op is registered on first load)."""
    spec = importlib.util.spec_from_file_location(f"km_{name}", EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rope():
    """The traced RoPE program."""
    return example("rope_km").rope_program.trace(MACHINE, ROPE_DTYPES)


def swiglu():
    """The traced SwiGLU program."""
    return example("swiglu_km").swiglu_program.trace(MACHINE, SWIGLU_DTYPES)
