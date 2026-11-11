"""A per-core stage table on the card (tests/km_stages.py): every kernel switches on
the row's stage word; copies and sums land where the table says, and a new table
builds nothing."""

import pytest
import torch

from km_stages import run, stages_program


def _expected(x, rows_per_core):
    tiles = x.float().reshape(32, -1, 32)
    out = {}
    for rows in rows_per_core:
        for stage, src, count, dst in rows:
            got = tiles[:, src] if stage == 0 else tiles[:, src] + tiles[:, src + 1]
            assert count == stage + 1
            out[dst] = got
    return out


@pytest.mark.hw
def test_stage_table(native):
    import kohakutt.dsl.metal as km

    dev = native.device(0)
    cols, rows = native.compute_grid()
    launcher = km.Launcher(stages_program(km.Machine(cols, rows)))
    tables = [
        [[[0, 0, 1, 0], [1, 2, 2, 1], [0, 7, 1, 2], [1, 5, 2, 3]], [[1, 0, 2, 4]]],
        [[[1, 6, 2, 0]], [[0, 1, 1, 1], [0, 2, 1, 2]], [[1, 3, 2, 3], [0, 4, 1, 4]]],
    ]
    g = torch.Generator().manual_seed(0)
    x = torch.randn(32, 8 * 32, generator=g).bfloat16()
    for i, rows_per_core in enumerate(tables):
        native.synchronize()
        before = native.counters()
        out = run(launcher, x.to(dev), rows_per_core).cpu().float()
        if i:
            grown = native.counters()["kernel_compiles"] - before["kernel_compiles"]
            assert grown == 0
        got = out.reshape(32, -1, 32)
        for dst, want in _expected(x, rows_per_core).items():
            torch.testing.assert_close(got[:, dst], want, rtol=1e-2, atol=1e-2)
