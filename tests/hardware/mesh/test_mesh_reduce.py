"""Mesh: reductions and normalisations: sum / mean / amax over each dim, softmax,
layer norm and RMS norm, with the input in every placement (replicated, sharded on each
dim, partial sum), odd sizes and a size-1 dim. Each result is compared with the CPU fp32
result and with one card's error for the same op, and its placement must be a legal
one."""

import itertools

import pytest
import torch
import torch.nn.functional as F
from mesh_support import SPECS_2D, host, run_case

pytestmark = [pytest.mark.hw, pytest.mark.dual]

REASON_SUM = "a bf16 partial sum is rounded on each card before the sum (sum, mean)"
REASON_SOFTMAX = "a bf16 partial sum is rounded on each card before the sum (softmax)"
REASON_RMS = "a bf16 partial sum is rounded on each card before the sum (rms norm)"

# Case ids known to fail, with the reason.
KNOWN_IDS = {
    **dict.fromkeys(
        [
            "sum0-S0-64x64-bf16",
            "sum0-S0-63x65-bf16",
            "sum0-S0-64x1-bf16",
            "sum1-S1-64x64-bf16",
            "sum1-S1-64x64-fp32",
            "sum1-S1-63x65-bf16",
            "sum1-S1-1x64-bf16",
            "sum_all-S1-64x64-fp32",
            "sum_all-S0-63x65-bf16",
            "sum_all-S1-63x65-bf16",
            "sum_all-S1-1x64-bf16",
            "sum_all-S0-64x1-bf16",
            "sum1_keep-S1-64x64-bf16",
            "sum1_keep-S1-64x64-fp32",
            "sum1_keep-S1-63x65-bf16",
            "sum1_keep-S1-1x64-bf16",
            "mean0-S0-64x64-bf16",
            "mean0-P-63x65-bf16",
            "mean0-S0-64x1-bf16",
            "mean1-S1-64x64-bf16",
            "mean1-S1-64x64-fp32",
            "mean1-P-63x65-bf16",
            "mean1-S1-1x64-bf16",
        ],
        REASON_SUM,
    ),
    **dict.fromkeys(
        [
            "softmax0-S0-64x64-bf16",
            "softmax0-S0-63x65-bf16",
            "softmax0-S0-64x1-bf16",
            "softmax1-S1-64x64-bf16",
            "softmax1-S1-63x65-bf16",
            "softmax1-S1-1x64-bf16",
        ],
        REASON_SOFTMAX,
    ),
    **dict.fromkeys(
        [
            "rms_norm-xS0-wR",
            "rms_norm-xS0-wS0",
            "rms_norm-xS1-wR",
            "rms_norm-xS1-wS0",
            "rms_norm-xP-wR",
            "rms_norm-xP-wS0",
        ],
        REASON_RMS,
    ),
}

SHAPES = [(64, 64), (63, 65), (1, 64), (64, 1)]
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}

REDUCE = {
    "sum0": lambda x: x.sum(0),
    "sum1": lambda x: x.sum(1),
    "sum_all": lambda x: x.sum(),
    "sum1_keep": lambda x: x.sum(1, keepdim=True),
    "mean0": lambda x: x.mean(0),
    "mean1": lambda x: x.mean(1),
    "amax0": lambda x: x.amax(0),
    "amax1": lambda x: x.amax(1),
    "softmax0": lambda x: torch.softmax(x, 0),
    "softmax1": lambda x: torch.softmax(x, 1),
    "logsoftmax1": lambda x: torch.log_softmax(x, 1),
}
EXACT = {"amax0", "amax1"}


def _valid(spec, shape):
    return not (spec[0] == "S" and shape[int(spec[1:])] < 2)


def _mark(cid):
    return (
        [pytest.mark.xfail(strict=True, reason=KNOWN_IDS[cid])]
        if cid in KNOWN_IDS
        else []
    )


def _cases():
    for op in REDUCE:
        for shape in SHAPES:
            for spec in (s for s in SPECS_2D if _valid(s, shape)):
                for dt in DTYPES:
                    if dt == "fp32" and shape != (64, 64):
                        continue
                    cid = f"{op}-{spec}-{'x'.join(map(str, shape))}-{dt}"
                    yield pytest.param(op, spec, shape, dt, id=cid, marks=_mark(cid))


@pytest.mark.parametrize(("op", "spec", "shape", "dt"), list(_cases()))
def test_reduce(native_module, mesh, op, spec, shape, dt):
    """A reduction or softmax gives the CPU result in a legal placement."""
    x = host(shape, DTYPES[dt], seed=60)
    run_case(
        native_module,
        mesh,
        "reduce",
        f"{op}|{spec}|{shape}|{dt}",
        REDUCE[op],
        [x],
        [spec],
        exact=op in EXACT,
    )


def _norm_cases():
    for kind, xs_, ws_ in itertools.product(
        ["layer_norm", "rms_norm"], SPECS_2D, ["R", "S0"]
    ):
        cid = f"{kind}-x{xs_}-w{ws_}"
        yield pytest.param(kind, xs_, ws_, id=cid, marks=_mark(cid))


@pytest.mark.parametrize(("kind", "xs_", "ws_"), list(_norm_cases()))
def test_norms(native_module, mesh, kind, xs_, ws_):
    """layer_norm and rms_norm of a [64, 64] input over its last dim, with the weight
    replicated or sharded, give the CPU result in a legal placement."""
    x = host((64, 64), torch.bfloat16, seed=61)
    w = host((64,), torch.bfloat16, seed=62)
    if kind == "layer_norm":

        def fn(a, b):
            return F.layer_norm(a, (64,), b, None)

    else:

        def fn(a, b):
            return F.rms_norm(a, (64,), b)

    run_case(
        native_module, mesh, "reduce", f"{kind}|{xs_}|{ws_}", fn, [x, w], [xs_, ws_]
    )


@pytest.mark.parametrize("spec", ["R", "S0", "S1"])
def test_norm_odd_width(native_module, mesh, spec):
    """layer_norm over a width of 65 that does not split into equal halves."""
    x = host((64, 65), torch.bfloat16, seed=63)
    run_case(
        native_module,
        mesh,
        "reduce",
        f"layer_norm_odd|{spec}",
        lambda a: F.layer_norm(a, (65,)),
        [x],
        [spec],
    )
