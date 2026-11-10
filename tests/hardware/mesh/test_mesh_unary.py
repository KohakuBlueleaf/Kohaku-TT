"""Mesh: elementwise unary ops, casts and scalar ops: every placement of the
input (replicated, sharded on each dim, partial sum), odd sizes, a size-1 dim, an empty
tensor, bf16 and fp32 and int32. Each result is compared with the CPU fp32 result and
with one card's error for the same op, and its placement must be a legal one."""

import pytest
import torch
from mesh_support import SPECS_2D, host, run_case

pytestmark = [pytest.mark.hw, pytest.mark.dual]

# Cases known to fail.
KNOWN = {}

SHAPES = [(64, 64), (63, 65), (1, 64), (0, 64)]
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}

FLOAT_OPS = {
    "exp": torch.exp,
    "tanh": torch.tanh,
    "relu": torch.relu,
    "gelu": torch.nn.functional.gelu,
    "neg": torch.neg,
    "abs": torch.abs,
    "square": lambda x: x * x,
    "mul_scalar": lambda x: x * 2.5,
    "add_scalar": lambda x: x + 1.0,
    "rsub_scalar": lambda x: 3.0 - x,
    "div_scalar": lambda x: x / 4.0,
    "clamp": lambda x: x.clamp(-0.5, 0.5),
}
EXACT = {"relu", "neg", "abs", "clamp"}


def _specs_for(shape):
    """Placements used for `shape`: no shard on a size-1 dim, and none of an empty
    tensor (test_mesh_empty.py covers that)."""
    if 0 in shape:
        return ["R"]
    return [s for s in SPECS_2D if not (s[0] == "S" and shape[int(s[1:])] < 2)]


def _cases():
    for op in FLOAT_OPS:
        for shape in SHAPES:
            for spec in _specs_for(shape):
                for dt in DTYPES:
                    # bf16 on every placement; fp32 on the even shape only, to keep
                    # the file inside one card hold.
                    if dt == "fp32" and shape != (64, 64):
                        continue
                    key = (op, spec, shape, dt)
                    marks = []
                    if key in KNOWN:
                        marks.append(pytest.mark.xfail(strict=True, reason=KNOWN[key]))
                    yield pytest.param(
                        *key,
                        marks=marks,
                        id=f"{op}-{spec}-{'x'.join(map(str, shape))}-{dt}",
                    )


@pytest.mark.parametrize(("op", "spec", "shape", "dt"), list(_cases()))
def test_unary_float(native_module, mesh, op, spec, shape, dt):
    """A unary or scalar op gives the CPU result in a legal placement."""
    x = host(shape, DTYPES[dt], seed=1)
    run_case(
        native_module,
        mesh,
        "unary",
        f"{op}|{spec}|{shape}|{dt}",
        FLOAT_OPS[op],
        [x],
        [spec],
        exact=op in EXACT,
    )


@pytest.mark.parametrize("spec", ["R", "S0", "S1"])
@pytest.mark.parametrize("shape", [(64, 64), (63, 65)])
def test_int32_scalar_ops(native_module, mesh, spec, shape):
    """int32 add / mul by a scalar and negation are exact."""
    x = host(shape, torch.int32, seed=2)
    run_case(
        native_module,
        mesh,
        "unary",
        f"int_ops|{spec}|{shape}",
        lambda t: (t + 3) * 2 - t,
        [x],
        [spec],
        exact=True,
    )


@pytest.mark.parametrize("spec", SPECS_2D)
@pytest.mark.parametrize("target", ["fp32", "bf16", "int32"])
def test_casts(native_module, mesh, spec, target):
    """A cast keeps the values (bf16 / fp32 exactly, to int32 by truncation) in a legal
    placement."""
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "int32": torch.int32}[
        target
    ]
    src = torch.bfloat16
    x = host((64, 64), src, seed=3)
    run_case(
        native_module,
        mesh,
        "unary",
        f"cast_{target}|{spec}",
        lambda t: t.to(dtype),
        [x],
        [spec],
        exact=True,
    )
