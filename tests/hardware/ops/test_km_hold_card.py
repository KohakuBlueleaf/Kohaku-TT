"""A tile held in L1 and updated in place by the packer (tests/km_hold.py) on the
card: written x, copied back and halved, accumulated onto with 1.0: x / 2 + 1."""

import pytest
import torch

from km_hold import hold_program


@pytest.mark.hw
def test_held_tile_updated_in_place(native):
    import kohakutt.dsl.metal as km

    launcher = km.Launcher(hold_program(km.Machine(*native.compute_grid())))
    tiles = 6
    gen = torch.Generator().manual_seed(0)
    x = torch.randint(-64, 64, (32, 32 * tiles), generator=gen).float()
    (out,) = launcher(
        {"x": x.to(native.device(0))},
        outputs={"out": ([32, 32 * tiles], torch.float32)},
        runtime={"tiles": tiles},
    )
    assert torch.equal(out.cpu(), x / 2 + 1)
