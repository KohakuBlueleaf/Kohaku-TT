"""Exchange inside groups of cores on the card (tests/km_group.py): after a group
barrier every core fetches its group's held tiles and writes their sum; a core count
that splits a group is refused before anything runs."""

import pytest
import torch

from km_group import GROUP, group_program


@pytest.mark.hw
def test_group_sums(native):
    import kohakutt.dsl.metal as km

    dev = native.device(0)
    cols, rows = native.compute_grid()
    launcher = km.Launcher(group_program(km.Machine(cols, rows)))
    for seed, groups in enumerate([1, 3, (cols * rows) // GROUP]):
        cores = groups * GROUP
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(32, cores * 32, generator=g).bfloat16()
        (out,) = launcher(
            {"x": x.to(dev)},
            outputs={"out": ([32, cores * 32], torch.bfloat16)},
            runtime={"working": cores},
        )
        sums = x.float().reshape(32, groups, GROUP, 32).sum(2, keepdim=True)
        want = sums.expand(32, groups, GROUP, 32).reshape(32, cores * 32)
        torch.testing.assert_close(out.cpu().float(), want, rtol=2e-2, atol=5e-2)
    with pytest.raises(ValueError, match="groups 4 cores"):
        launcher(
            {"x": x.to(dev)},
            outputs={"out": ([32, cores * 32], torch.bfloat16)},
            runtime={"working": 6},
        )
