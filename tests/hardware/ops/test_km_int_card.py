"""int32 tiles through the compute kernel on the card (tests/km_int.py): lanewise
int32 arithmetic, comparisons, bitwise ops and shifts, and the int32 / fp32
conversions, equal to torch."""

import pytest
import torch

from km_int import OPS_RUN, expected, int_program, run


@pytest.mark.hw
def test_int32_ops_equal_torch(native):
    import kohakutt.dsl.metal as km

    dev = native.device(0)
    cols, rows = native.compute_grid()
    launcher = km.Launcher(int_program(km.Machine(cols, rows)))
    g = torch.Generator().manual_seed(0)
    a = torch.randint(-(2**20), 2**20, (32, 96), generator=g, dtype=torch.int32)
    b = torch.randint(-(2**10), 2**10, (32, 96), generator=g, dtype=torch.int32)
    b[:, :8] = a[:, :8]
    s = torch.randint(0, 8, (32, 96), generator=g, dtype=torch.int32)
    x = torch.randn(32, 96, generator=g) * 1000
    out, f, gx = (t.cpu() for t in run(launcher, *(t.to(dev) for t in (a, b, s, x))))
    tiles = out.reshape(32, 3, len(OPS_RUN), 32)
    for i, (name, want) in enumerate(zip(OPS_RUN, expected(a, b, s))):
        got = tiles[:, :, i, :].reshape(32, 96)
        assert torch.equal(got, want), name
    assert torch.equal(f, a.float())
    assert torch.equal(gx, x.trunc().int())
