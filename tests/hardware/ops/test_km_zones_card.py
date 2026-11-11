"""Profiler zones and watcher waypoints of tests/km_zones.py on the card: the program
copies exactly (zones and waypoints compiled out, as in a plain build)."""

import pytest
import torch

from km_zones import zones_program


@pytest.mark.hw
def test_zoned_program_copies_exactly(native):
    import kohakutt.dsl.metal as km

    dev = native.device(0)
    cols, rows = native.compute_grid()
    launcher = km.Launcher(zones_program(km.Machine(cols, rows)))
    x = torch.randn(96, 64).bfloat16()
    (out,) = launcher(
        {"x": x.to(dev)},
        outputs={"out": ([96, 64], torch.bfloat16)},
        runtime={"tiles": 6},
    )
    assert torch.equal(out.cpu(), x)
