"""Mesh: matmul and Linear: every pair of placements of the two operands
(replicated, sharded on each dim, partial sum), odd sizes, a batched left operand, a
bias in each placement. Each result is compared with the CPU fp32 result and with one
card's error for the same op, and its placement must be a legal one."""

import itertools

import pytest
import torch
from mesh_support import SPECS_2D, host, run_case

pytestmark = [pytest.mark.hw, pytest.mark.dual]

REASON_KSPLIT = (
    "a bf16 partial sum is rounded on each card before the sum (split matmul)"
)

# Case ids known to fail, with the reason.
KNOWN_IDS = dict.fromkeys(
    [
        "mm-S1_S0-63x65@65x67-bf16",
        "mm-R_S0-1x64@64x64-bf16",
        "mm-S1_R-1x64@64x64-bf16",
        "mm-S1_S0-1x64@64x64-bf16",
        "mm-R_S0-64x64@64x1-bf16",
        "mm-S1_R-64x64@64x1-bf16",
        "mm-S1_S0-64x64@64x1-bf16",
        "linear-xR-wS1-bnone",
        "linear-xR-wS1-bR",
        "linear-xR-wS1-bS0",
        "linear-xS1-wR-bnone",
        "linear-xS1-wR-bR",
        "linear-xS1-wR-bS0",
        "linear-xS1-wS1-bnone",
        "linear-xS1-wS1-bR",
        "linear-xS1-wS1-bS0",
        "mmb-S1_S0",
    ],
    REASON_KSPLIT,
)

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
MM_SHAPES = [
    ((64, 64), (64, 64)),
    ((63, 65), (65, 67)),
    ((1, 64), (64, 64)),
    ((64, 64), (64, 1)),
]


def _valid(spec, shape):
    return not (spec[0] == "S" and shape[int(spec[1:])] < 2)


def _mark(cid):
    return (
        [pytest.mark.xfail(strict=True, reason=KNOWN_IDS[cid])]
        if cid in KNOWN_IDS
        else []
    )


def _mm_cases():
    for sa, sb in MM_SHAPES:
        for specs in itertools.product(
            [s for s in SPECS_2D if _valid(s, sa)],
            [s for s in SPECS_2D if _valid(s, sb)],
        ):
            for dt in DTYPES:
                if dt == "fp32" and (sa, sb) != ((64, 64), (64, 64)):
                    continue
                a, b = ("x".join(map(str, s)) for s in (sa, sb))
                cid = f"mm-{'_'.join(specs)}-{a}@{b}-{dt}"
                yield pytest.param(sa, sb, specs, dt, id=cid, marks=_mark(cid))


@pytest.mark.parametrize(("sa", "sb", "specs", "dt"), list(_mm_cases()))
def test_matmul(native_module, mesh, sa, sb, specs, dt):
    """a @ b gives the CPU result in a legal placement for every pair of placements."""
    xs = [host(sa, DTYPES[dt], seed=30), host(sb, DTYPES[dt], seed=31)]
    run_case(
        native_module,
        mesh,
        "matmul",
        f"matmul|{specs}|{sa}|{sb}|{dt}",
        torch.matmul,
        xs,
        list(specs),
    )


def _linear_cases():
    for xs_, ws_, bs_ in itertools.product(
        ["R", "S0", "S1", "P"], ["R", "S0", "S1"], ["none", "R", "S0"]
    ):
        cid = f"linear-x{xs_}-w{ws_}-b{bs_}"
        yield pytest.param(xs_, ws_, bs_, id=cid, marks=_mark(cid))


@pytest.mark.parametrize(("xs_", "ws_", "bs_"), list(_linear_cases()))
def test_linear(native_module, mesh, xs_, ws_, bs_):
    """F.linear(x, w, b) for x [64, 64], w [96, 64], b [96] in each placement."""
    x = host((64, 64), torch.bfloat16, seed=40)
    w = host((96, 64), torch.bfloat16, seed=41)
    b = host((96,), torch.bfloat16, seed=42)
    if bs_ == "none":
        run_case(
            native_module,
            mesh,
            "matmul",
            f"linear|{xs_}|{ws_}|none",
            torch.nn.functional.linear,
            [x, w],
            [xs_, ws_],
        )
    else:
        run_case(
            native_module,
            mesh,
            "matmul",
            f"linear|{xs_}|{ws_}|{bs_}",
            torch.nn.functional.linear,
            [x, w, b],
            [xs_, ws_, bs_],
        )


@pytest.mark.parametrize(
    "specs",
    [
        pytest.param(s, id=f"mmb-{'_'.join(s)}", marks=_mark(f"mmb-{'_'.join(s)}"))
        for s in [("R", "R"), ("S0", "R"), ("S1", "S0"), ("R", "S1"), ("S1", "R")]
    ],
)
def test_matmul_batched(native_module, mesh, specs):
    """A [2, 32, 64] left operand against a [64, 48] weight."""
    xs = [
        host((2, 32, 64), torch.bfloat16, seed=50),
        host((64, 48), torch.bfloat16, seed=51),
    ]
    run_case(
        native_module,
        mesh,
        "matmul",
        f"matmul_batched|{specs}",
        torch.matmul,
        xs,
        list(specs),
    )
