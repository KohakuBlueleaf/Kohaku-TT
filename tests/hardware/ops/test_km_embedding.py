"""Regression: the row-gather program written at the metal level (tests/km_embedding.py)
equals the backend's row gather bit for bit on the card, and no table or index size
compiles a kernel.
"""

import pytest
import torch

from km_embedding import embedding_program, plan, register

# (table shape, dtype, index counts): the tables the backend gathers with its own
# kernel (fp32, bf16 slices, and 2-D bf16 tables up to 128 rows looked up).
CASES = {
    "fp32": ((1000, 256), torch.float32, (1, 31, 32, 33, 128, 500)),
    "fp32, ragged dim": ((5000, 96), torch.float32, (1, 33, 500)),
    "fp32, slices": ((3, 700, 64), torch.float32, (1, 32, 33, 500)),
    "bf16": ((1000, 256), torch.bfloat16, (1, 31, 32, 33, 128)),
    "bf16, slices": ((3, 700, 64), torch.bfloat16, (1, 32, 33, 500)),
}


def _counted(native, fn):
    """(result of fn(), counter increases over it), with the stream drained."""
    native.synchronize()
    before = native.counters()
    out = fn()
    native.synchronize()
    after = native.counters()
    return out, {name: after[name] - before[name] for name in after}


def _inputs(dev, shape, dtype, n, seed=0):
    """A tt table of `shape` and `n` int32 row indices into it."""
    g = torch.Generator().manual_seed(seed + n)
    table = torch.randn(*shape, generator=g).to(dtype)
    index = torch.randint(0, shape[-2], (n,), generator=g, dtype=torch.int32)
    return table.to(dev), index.to(dev)


# The sweep runs first: it counts the program builds of each core count.
@pytest.mark.hw
def test_sizes_compile_nothing(native):
    import kohakutt.dsl.metal as km

    ops, dev = register(), native.device(0)
    formats = {"table": "float32", "index": "int32", "out": "float32"}
    program = embedding_program.trace(km.Machine(*native.compute_grid()), formats)
    shapes = [((1000, 256), 33), ((1000, 256), 500), ((5000, 96), 33), ((64, 160), 5)]
    shapes += [((3, 700, 64), 33), ((1000, 256), 1)]
    seen = set()
    for round_ in range(2):
        for shape, n in shapes:
            cores = program.core_count(plan(shape, n))
            table, index = _inputs(dev, shape, torch.float32, n)
            _, grown = _counted(native, lambda t=table, i=index: ops.embedding_km(t, i))
            if seen:
                assert grown["kernel_compiles"] == 0, (shape, n)
                assert grown["program_entries"] == (cores not in seen), (shape, n)
            if round_:
                assert grown["programs"] == grown["desc_hits"] == 1, (shape, n)
            seen.add(cores)
    assert len(seen) >= 4


@pytest.mark.hw
@pytest.mark.parametrize("name", list(CASES))
def test_equals_the_backend(native, name):
    ops, dev = register(), native.device(0)
    shape, dtype, counts = CASES[name]
    for n in counts:
        table, index = _inputs(dev, shape, dtype, n)
        want, grown = _counted(
            native, lambda t=table, i=index: torch.index_select(t, t.dim() - 2, i)
        )
        assert grown["programs"] == 1, n
        got = ops.embedding_km(table, index)
        assert torch.equal(got.cpu(), want.cpu()), n
        assert torch.equal(got.cpu(), table.cpu().index_select(-2, index.cpu().long()))
        row = ops.embedding_km(table, index.reshape(1, n))
        assert torch.equal(row.cpu(), want.cpu()), n


@pytest.mark.hw
def test_the_index_is_taken_as_stored(native):
    ops, dev = register(), native.device(0)
    table, index = _inputs(dev, (64, 64), torch.float32, 8)
    with pytest.raises(Exception, match="int32 tensor of shape"):
        ops.embedding_km(table, index.reshape(2, 4))
    with pytest.raises(Exception, match="int32 tensor of shape"):
        ops.embedding_km(table, index.float())
