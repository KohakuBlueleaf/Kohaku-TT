"""A compute kernel branching on a flag it computed (tests/km_flag.py) on the card:
each tile is doubled exactly where row 0 holds an element above 3."""

import pytest
import torch

from km_flag import flag_program


@pytest.mark.hw
def test_branch_on_a_computed_flag(native):
    import kohakutt.dsl.metal as km

    dev = native.device(0)
    launcher = km.Launcher(flag_program(km.Machine(*native.compute_grid())))
    g = torch.Generator().manual_seed(15)
    tiles = 8
    x = torch.rand(32, 32 * tiles, generator=g) * 2
    hot = [1, 2, 5]
    for t in hot:
        x[0, 32 * t + 7] = 3.5
    x[5, 32 * 3 + 1] = 9.0  # above 3, but not in row 0
    (out,) = launcher(
        {"x": x.to(dev)},
        outputs={"out": (list(x.shape), torch.float32)},
        runtime={"tiles": tiles},
    )
    want = x.clone()
    for t in hot:
        want[:, 32 * t : 32 * (t + 1)] *= 2
    assert torch.equal(out.cpu(), want)
