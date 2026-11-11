"""The matrix unit's whole-tile ops (add / sub / mul, row / column / scalar broadcasts,
accumulation onto a slot) and reductions (row sum, column max, scalar sum) of
tests/km_tiles.py on the card agree with torch, and new sizes compile nothing."""

import pytest
import torch

from km_tiles import OUT_TILES, expected, run, tiles_program


@pytest.mark.hw
def test_tiles_and_reductions_agree_with_torch(native):
    import kohakutt.dsl.metal as km

    dev = native.device(0)
    cols, rows = native.compute_grid()
    launcher = km.Launcher(tiles_program(km.Machine(cols, rows)))
    for seed, row_tiles in enumerate([2, 37, 5]):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(row_tiles * 32, 64, generator=g).bfloat16()
        y = torch.randn(row_tiles * 32, 64, generator=g).bfloat16()
        v = torch.randn(32, 32, generator=g).bfloat16()
        s = torch.ones(32, 32).bfloat16()
        native.synchronize()
        before = native.counters()
        out = run(launcher, x.to(dev), y.to(dev), v.to(dev), s.to(dev)).cpu().float()
        grown = native.counters()["kernel_compiles"] - before["kernel_compiles"]
        assert seed == 0 or grown == 0, row_tiles
        want = expected(x, y, v)
        for t in range(OUT_TILES):
            got, ref = out[:, 32 * t : 32 * (t + 1)], want[:, 32 * t : 32 * (t + 1)]
            assert torch.allclose(got, ref, rtol=2e-2, atol=6e-2), (row_tiles, t)
