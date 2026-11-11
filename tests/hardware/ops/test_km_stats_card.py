"""Typed two-pass statistics on the card (tests/km_stats.py): row means of scale * x
with the statistics kept in the registers, or spilled to two slots while a generated
function scales each chunk, agree with torch in fp32; new sizes compile nothing."""

import pytest
import torch

from km_stats import run, stats_program


@pytest.mark.hw
@pytest.mark.parametrize("spill", [True, False])
def test_row_means(native, spill):
    import kohakutt.dsl.metal as km

    dev = native.device(0)
    cols, rows = native.compute_grid()
    launcher = km.Launcher(stats_program(km.Machine(cols, rows), spill=spill))
    for seed, (row_tiles, col_tiles) in enumerate([(2, 3), (9, 1), (4, 16)]):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(row_tiles * 32, col_tiles * 32, generator=g) + 3.0
        scale = 1.75 if spill else 1.0
        native.synchronize()
        before = native.counters()["kernel_compiles"]
        out = run(launcher, x.to(dev), scale).cpu()
        assert seed == 0 or native.counters()["kernel_compiles"] == before
        want = (x.double() * scale).mean(-1).float()
        torch.testing.assert_close(out[:, 0], want, rtol=2e-6, atol=2e-6)
