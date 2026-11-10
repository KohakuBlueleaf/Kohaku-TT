"""Mesh: binary elementwise ops and where: every pair of placements of the two
operands (replicated, sharded on each dim, partial sum), odd sizes, broadcasting a row
or a column, bf16 and fp32 and int32. Each result is compared with the CPU fp32 result
and with one card's error for the same op, and its placement must be a legal one."""

import itertools

import pytest
import torch
from mesh_support import SPECS_2D, host, run_case

pytestmark = [pytest.mark.hw, pytest.mark.dual]

REASON_MUL_PARTIAL = (
    "a bf16 partial sum is rounded on each card before the sum (multiply)"
)

# Case ids known to fail, with the reason.
KNOWN_IDS = dict.fromkeys(
    [
        "mul-R_P-64x64+64x64-bf16",
        "mul-R_P-64x64+64x64-fp32",
        "mul-S0_P-64x64+64x64-bf16",
        "mul-S0_P-64x64+64x64-fp32",
        "mul-S1_P-64x64+64x64-bf16",
        "mul-S1_P-64x64+64x64-fp32",
        "mul-P_R-64x64+64x64-bf16",
        "mul-P_R-64x64+64x64-fp32",
        "mul-P_S0-64x64+64x64-bf16",
        "mul-P_S0-64x64+64x64-fp32",
        "mul-P_S1-64x64+64x64-bf16",
        "mul-P_S1-64x64+64x64-fp32",
        "mul-R_P-63x65+63x65-bf16",
        "mul-S0_P-63x65+63x65-bf16",
        "mul-S1_P-63x65+63x65-bf16",
        "mul-R_P-64x64+1x64-bf16",
        "mul-S0_P-64x64+1x64-bf16",
        "mul-S1_P-64x64+1x64-bf16",
        "mul-P_R-64x64+1x64-bf16",
        "mul-P_S1-64x64+1x64-bf16",
        "mul-R_P-64x64+64x1-bf16",
        "mul-S0_P-64x64+64x1-bf16",
        "mul-S1_P-64x64+64x1-bf16",
    ],
    REASON_MUL_PARTIAL,
)

OPS = {
    "add": torch.add,
    "sub": torch.sub,
    "mul": torch.mul,
    "div": lambda a, b: a / (b.abs() + 1.0),
    "maximum": torch.maximum,
    "where": lambda c, a, b: torch.where(c > 0, a, b),
}
EXACT = {"maximum"}
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def _valid(spec, shape):
    return not (
        spec[0] == "S" and (len(shape) <= int(spec[1:]) or shape[int(spec[1:])] < 2)
    )


def _cases():
    shapes = [
        ((64, 64), (64, 64)),
        ((63, 65), (63, 65)),
        ((64, 64), (1, 64)),
        ((64, 64), (64, 1)),
    ]
    for op in OPS:
        n_in = 3 if op == "where" else 2
        for sa, sb in shapes:
            if op == "where" and sa != sb:
                continue
            if op != "where" and sa != sb and op not in ("add", "mul"):
                continue
            shapes_in = [sa, sb] if n_in == 2 else [sa, sa, sb]
            pool = [[s for s in SPECS_2D if _valid(s, sh)] for sh in shapes_in]
            for specs in itertools.product(*pool):
                # Three-operand where: only the pairs of the first two operands cross
                # with every placement; the third follows the second.
                if n_in == 3 and specs[2] != specs[1]:
                    continue
                for dt in DTYPES:
                    if dt == "fp32" and (sa, sb) != ((64, 64), (64, 64)):
                        continue
                    key = (op, specs, sa, sb, dt)
                    marks = []
                    a, b = ("x".join(map(str, s)) for s in (sa, sb))
                    cid = f"{op}-{'_'.join(specs)}-{a}+{b}-{dt}"
                    if cid in KNOWN_IDS:
                        marks.append(
                            pytest.mark.xfail(strict=True, reason=KNOWN_IDS[cid])
                        )
                    yield pytest.param(*key, marks=marks, id=cid)


@pytest.mark.parametrize(("op", "specs", "sa", "sb", "dt"), list(_cases()))
def test_binary_float(native_module, mesh, op, specs, sa, sb, dt):
    """A binary op gives the CPU result in a legal placement, for every pair of
    operand placements."""
    dtype = DTYPES[dt]
    shapes = [sa, sb] if len(specs) == 2 else [sa, sa, sb]
    xs = [host(sh, dtype, seed=10 + i) for i, sh in enumerate(shapes)]
    run_case(
        native_module,
        mesh,
        "binary",
        f"{op}|{specs}|{sa}|{sb}|{dt}",
        OPS[op],
        xs,
        list(specs),
        exact=op in EXACT,
    )


@pytest.mark.parametrize(
    "specs", [("R", "R"), ("S0", "R"), ("R", "S1"), ("S0", "S1"), ("S1", "S1")]
)
@pytest.mark.parametrize("op", ["add", "mul"])
def test_binary_int32(native_module, mesh, op, specs):
    """int32 add and mul are exact for each placement pair."""
    xs = [host((64, 64), torch.int32, seed=20 + i) for i in range(2)]
    run_case(
        native_module,
        mesh,
        "binary",
        f"int_{op}|{specs}",
        OPS[op],
        xs,
        list(specs),
        exact=True,
    )
