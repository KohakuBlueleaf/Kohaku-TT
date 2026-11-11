"""Data kept in L1 across the stages of one program, on the card (tests/km_sync.py):
after a barrier of every core each core fetches its right neighbour's held tile; the
output equals the input with the tiles moved accordingly, launch after launch."""

import pytest
import torch

from km_sync import neighbours, sync_program


@pytest.mark.hw
def test_tiles_fetched_from_the_neighbour_after_a_sync(native):
    import kohakutt.dsl.metal as km

    dev = native.device(0)
    cols, rows = native.compute_grid()
    machine = km.Machine(cols, rows)
    launcher = km.Launcher(sync_program(machine))
    cores = launcher.program.core_count({})
    for seed in range(3):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(32, cores * 32, generator=g).bfloat16()
        (out,) = launcher(
            {"x": x.to(dev)}, outputs={"out": ([32, cores * 32], torch.bfloat16)}
        )
        tiles = x.reshape(32, cores, 32)
        want = tiles[:, neighbours(cols, cores), :].reshape(32, cores * 32)
        assert torch.equal(out.cpu(), want), seed
