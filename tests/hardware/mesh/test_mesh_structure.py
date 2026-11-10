"""Mesh: structure ops: cat, slice, index, reshape, transpose, permute, clone,
embedding and where on tensors in every placement (replicated, sharded on each dim,
partial sum), odd sizes and a size-1 dim. Each result is compared with the CPU result
and with one card's, and its placement must be a legal one. Moves of data are exact."""

import pytest
import torch
from mesh_support import SPECS_2D, host, run_case

pytestmark = [pytest.mark.hw, pytest.mark.dual]

# Case ids known to fail, with the reason.
KNOWN_IDS = {}

SHAPES = [(64, 64), (63, 65), (1, 64)]

UNARY_STRUCT = {
    "transpose": lambda x: x.t().contiguous(),
    "reshape_flat": lambda x: x.reshape(-1),
    "reshape_split": lambda x: x.reshape(x.shape[0], 1, x.shape[1]),
    "clone": lambda x: x.clone(),
    "slice_rows": lambda x: x[1:],
    "slice_cols": lambda x: x[:, 1:],
    "slice_step": lambda x: x[:, ::2],
    "index_row": lambda x: x[0],
    "unsqueeze_perm": lambda x: x.unsqueeze(0).permute(2, 0, 1).contiguous(),
    "cat_self_0": lambda x: torch.cat([x, x], 0),
    "cat_self_1": lambda x: torch.cat([x, x], 1),
    "flip": lambda x: x.flip(0),
    "contiguous_view": lambda x: x.t(),
}


def _valid(spec, shape):
    return not (spec[0] == "S" and shape[int(spec[1:])] < 2)


def _mark(cid):
    return (
        [pytest.mark.xfail(strict=True, reason=KNOWN_IDS[cid])]
        if cid in KNOWN_IDS
        else []
    )


def _cases():
    for op in UNARY_STRUCT:
        for shape in SHAPES:
            for spec in (s for s in SPECS_2D if _valid(s, shape)):
                cid = f"{op}-{spec}-{'x'.join(map(str, shape))}"
                yield pytest.param(op, spec, shape, id=cid, marks=_mark(cid))


@pytest.mark.parametrize(("op", "spec", "shape"), list(_cases()))
def test_structure(native_module, mesh, op, spec, shape):
    """A move of data gives the CPU result in a legal placement (bf16)."""
    x = host(shape, torch.bfloat16, seed=70)
    run_case(
        native_module,
        mesh,
        "structure",
        f"{op}|{spec}|{shape}",
        UNARY_STRUCT[op],
        [x],
        [spec],
        exact=True,
    )


def _cat_cases():
    for sa in ("R", "S0", "S1", "P"):
        for sb in ("R", "S0", "S1", "P"):
            for dim in (0, 1):
                cid = f"cat{dim}-{sa}_{sb}"
                yield pytest.param(sa, sb, dim, id=cid, marks=_mark(cid))


@pytest.mark.parametrize(("sa", "sb", "dim"), list(_cat_cases()))
def test_cat_mixed(native_module, mesh, sa, sb, dim):
    """torch.cat of two [64, 64] tensors in different placements is exact."""
    xs = [
        host((64, 64), torch.bfloat16, seed=71),
        host((64, 64), torch.bfloat16, seed=72),
    ]
    run_case(
        native_module,
        mesh,
        "structure",
        f"cat{dim}|{sa}|{sb}",
        lambda a, b: torch.cat([a, b], dim),
        xs,
        [sa, sb],
        exact=True,
    )


@pytest.mark.parametrize("spec", ["R", "S0", "S1"])
@pytest.mark.parametrize("vocab", [64, 63])
def test_embedding_table(native_module, mesh, spec, vocab):
    """An embedding lookup with the table replicated or sharded on either dim; ids
    replicated. The rows are exact."""
    table = host((vocab, 64), torch.bfloat16, seed=73)
    ids = torch.randint(0, vocab, (4, 8), generator=torch.Generator().manual_seed(74))

    def fn(t, i):
        return torch.nn.functional.embedding(i, t)

    run_case(
        native_module,
        mesh,
        "structure",
        f"embedding|{spec}|{vocab}",
        fn,
        [table, ids],
        [spec, "R"],
        exact=True,
    )


@pytest.mark.parametrize("idx_spec", ["R", "S0"])
@pytest.mark.parametrize("spec", ["R", "S0", "S1"])
def test_index_select(native_module, mesh, spec, idx_spec):
    """index_select rows of a [64, 64] tensor with ids in each placement is exact."""
    x = host((64, 64), torch.bfloat16, seed=75)
    ids = torch.randint(0, 64, (16,), generator=torch.Generator().manual_seed(76))

    def fn(a, i):
        return torch.index_select(a, 0, i)

    run_case(
        native_module,
        mesh,
        "structure",
        f"index_select|{spec}|{idx_spec}",
        fn,
        [x, ids],
        [spec, idx_spec],
        exact=True,
    )


@pytest.mark.parametrize("spec", SPECS_2D)
def test_cpu_roundtrip(native_module, mesh, spec):
    """A tensor in any placement comes back to the host with its values."""
    x = host((63, 65), torch.bfloat16, seed=77)
    run_case(
        native_module,
        mesh,
        "structure",
        f"cpu|{spec}",
        lambda a: a + 0,
        [x],
        [spec],
        exact=True,
    )
