"""The 2-D matmul on the card (tests/km_grid2d.py): a multicast along each row (a row
lane) and w down each column; every output tile equals torch's a @ w."""

import pytest
import torch

from km_grid2d import grid2d_program, run


@pytest.mark.hw
def test_grid2d_matmul(native):
    import kohakutt.dsl.metal as km

    dev = native.device(0)
    cols, rows = native.compute_grid()
    launcher = km.Launcher(grid2d_program(km.Machine(cols, rows)))
    for seed, kt in enumerate([1, 4, 9]):
        g = torch.Generator().manual_seed(seed)
        a = torch.randn(rows * 32, kt * 32, generator=g).bfloat16()
        w = (torch.randn(kt * 32, cols * 32, generator=g) / 8).bfloat16()
        out = run(launcher, a.to(dev), w.to(dev), cols).cpu().float()
        want = a.float() @ w.float()
        torch.testing.assert_close(out, want, rtol=2e-2, atol=5e-2)
