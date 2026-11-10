"""Mesh: gather along the last dim of replicated, sharded and pending-sum inputs; values
are small integers, so every result is exact. Inputs sharded on a dim before the last
gather their own rows on each chip, with no conversion and no fallback."""

import pytest
import torch
from mesh_support import counters_around, place, read

pytestmark = [pytest.mark.hw, pytest.mark.dual]

DTYPES = [torch.bfloat16, torch.float32]


def _values(dtype, rows=64, width=96, picks=40):
    gen = torch.Generator().manual_seed(3)
    value = torch.randint(-4, 5, (rows, width), generator=gen).to(dtype)
    index = torch.randint(0, width, (rows, picks), generator=gen)
    return value, index


@pytest.mark.parametrize("spec", ["R", "S0", "S1", "P"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_gather_last_dim(native_module, mesh, spec, dtype):
    value, index = _values(dtype)
    x = place(native_module, mesh, value, spec)
    want = torch.gather(read(x), 1, index)
    got = torch.gather(x, 1, index.to(mesh))
    assert torch.equal(read(got), want)


@pytest.mark.parametrize(("xspec", "ispec"), [("S0", "S0"), ("S0", "R"), ("R", "S0")])
@pytest.mark.parametrize("dtype", DTYPES)
def test_gather_rows_stay_on_their_chip(native_module, mesh, xspec, ispec, dtype):
    """x and index sharded (or one of them replicated) on the rows: the result is
    sharded on the rows and nothing is gathered or reduced."""
    value, index = _values(dtype)
    x = place(native_module, mesh, value, xspec)
    idx = place(native_module, mesh, index, ispec)
    got, delta, fallbacks = counters_around(lambda: torch.gather(x, 1, idx))
    assert torch.equal(read(got), torch.gather(value, 1, index))
    assert native_module.placement(got) == "Sharded(dim=0)"
    assert not fallbacks
    for name in ("gathers", "all_gathers", "all_reduces", "reduce_scatters"):
        assert not delta.get(name), (name, delta)


@pytest.mark.parametrize("dtype", DTYPES)
def test_gather_index_sharded_on_picks(native_module, mesh, dtype):
    """A replicated input with an index sharded on the last dim gives a result sharded
    on it, each chip gathering its own picks."""
    value, index = _values(dtype)
    x = place(native_module, mesh, value, "R")
    idx = place(native_module, mesh, index, "S1")
    got, delta, fallbacks = counters_around(lambda: torch.gather(x, 1, idx))
    assert torch.equal(read(got), torch.gather(value, 1, index))
    assert native_module.placement(got) == "Sharded(dim=1)"
    assert not fallbacks
    assert not delta.get("all_gathers")


@pytest.mark.parametrize("dtype", DTYPES)
def test_gather_input_sharded_on_last_dim_converts(native_module, mesh, dtype):
    """An input sharded on the gathered dim is made whole on every chip first."""
    value, index = _values(dtype)
    x = place(native_module, mesh, value, "S1")
    got, delta, _ = counters_around(lambda: torch.gather(x, 1, index.to(mesh)))
    assert torch.equal(read(got), torch.gather(value, 1, index))
    assert native_module.placement(got) == "Replicated"
    assert delta.get("all_gathers", 0) + delta.get("gathers", 0) >= 1


@pytest.mark.parametrize("dtype", DTYPES)
def test_gather_pending_sum_completes_first(native_module, mesh, dtype):
    """A pending sum is completed by an all-reduce before the gather."""
    value, index = _values(dtype)
    x = place(native_module, mesh, value, "P")
    want = torch.gather(read(x), 1, index)
    got, delta, _ = counters_around(lambda: torch.gather(x, 1, index.to(mesh)))
    assert torch.equal(read(got), want)
    assert delta.get("all_reduces", 0) >= 1


@pytest.mark.parametrize("dtype", DTYPES)
def test_gather_rows_not_dividing_the_mesh(native_module, mesh, dtype):
    """Rows that do not divide the mesh (shards padded) still give the exact rows."""
    value, index = _values(dtype, rows=33)
    x = place(native_module, mesh, value, "S0")
    idx = place(native_module, mesh, index, "S0")
    got = torch.gather(x, 1, idx)
    assert torch.equal(read(got), torch.gather(value, 1, index))


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("spec", ["R", "S0", "S1"])
def test_gather_other_dims_match_cpu(native_module, mesh, dim, spec):
    """A gather along a dim before the last, of a placed 3-d input, gives the CPU
    values."""
    gen = torch.Generator().manual_seed(5)
    value = torch.randint(-4, 5, (4, 32, 64), generator=gen).to(torch.bfloat16)
    index = torch.randint(0, value.shape[dim], (4, 32, 64), generator=gen)
    x = place(native_module, mesh, value, spec)
    got = torch.gather(x, dim, index.to(mesh))
    assert torch.equal(read(got), torch.gather(value, dim, index))


@pytest.mark.parametrize("ispec", ["R", "S0"])
def test_gather_fewer_index_rows_than_input(native_module, mesh, ispec):
    """An index with fewer rows than a row-sharded input converts the input (the
    shards would not line up) and gives the CPU values."""
    gen = torch.Generator().manual_seed(7)
    value = torch.randint(-4, 5, (64, 96), generator=gen).to(torch.bfloat16)
    index = torch.randint(0, 96, (40, 30), generator=gen)
    x = place(native_module, mesh, value, "S0")
    idx = place(native_module, mesh, index, ispec)
    got = torch.gather(x, 1, idx)
    assert torch.equal(read(got), torch.gather(value, 1, index))


@pytest.mark.parametrize("dtype", DTYPES)
def test_gather_replicated_chips_agree(native_module, mesh, dtype):
    """A replicated gather runs on both chips: the rows chip 0 holds (top half) and chip
    1 holds (bottom half), then the columns each holds, are right."""
    value, index = _values(dtype)
    x = place(native_module, mesh, value, "R")
    got = torch.gather(x, 1, index.to(mesh))
    want = torch.gather(value, 1, index)
    assert torch.equal(read(got), want)
    assert torch.equal(read(native_module.shard(got, 0)), want)
    assert torch.equal(read(native_module.shard(got, 1)), want)
