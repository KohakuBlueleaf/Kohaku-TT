"""Packer accumulation across a loop's iterations on the card (tests/km_accumulate.py):
the per-row sum of x's tiles, packed into one held page, equals torch's in fp32."""

import pytest
import torch

from km_accumulate import accumulate_program, run


@pytest.mark.hw
def test_tiles_summed_in_one_page(native):
    import kohakutt.dsl.metal as km

    dev = native.device(0)
    cols, rows = native.compute_grid()
    launcher = km.Launcher(accumulate_program(km.Machine(cols, rows)))
    for seed, (row_tiles, col_tiles) in enumerate([(2, 1), (5, 7), (40, 3)]):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(row_tiles * 32, col_tiles * 32, generator=g)
        out = run(launcher, x.to(dev)).cpu()
        want = x.reshape(row_tiles * 32, col_tiles, 32).sum(1)
        torch.testing.assert_close(out, want, rtol=1e-6, atol=1e-5)
