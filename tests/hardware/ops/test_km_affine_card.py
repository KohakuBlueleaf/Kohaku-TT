"""The norm affine step with broadcast operands loaded once per band
(tests/km_affine.py) on the card: two tiles per call equal one tile per call bit for
bit, and both equal torch's (x - m) r gamma + beta."""

import pytest
import torch

from km_affine import affine_program


@pytest.mark.hw
def test_affine_over_two_tiles(native):
    import kohakutt.dsl.metal as km

    dev = native.device(0)
    machine = km.Machine(*native.compute_grid())
    two = km.Launcher(affine_program(machine, tiles=2))
    one = km.Launcher(affine_program(machine, tiles=1))
    g = torch.Generator().manual_seed(13)
    rows, pairs = 3, 2
    x = torch.randn(32 * rows, 64 * pairs, generator=g)
    m = torch.randn(32 * rows, 1, generator=g).expand(-1, 32).contiguous()
    r = torch.rand(32 * rows, 1, generator=g).add(0.5).expand(-1, 32).contiguous()
    gamma = torch.randn(1, 64 * pairs, generator=g).expand(32, -1).contiguous()
    beta = torch.randn(1, 64 * pairs, generator=g).expand(32, -1).contiguous()
    inputs = {"x": x, "m": m, "r": r, "g": gamma, "b": beta}
    tt = {k: v.to(dev) for k, v in inputs.items()}
    outs = []
    for launcher in (two, one):
        (out,) = launcher(
            tt,
            outputs={"out": (list(x.shape), torch.float32)},
            runtime={"rows": rows, "pairs": pairs},
        )
        outs.append(out.cpu())
    assert torch.equal(outs[0].view(torch.int32), outs[1].view(torch.int32))
    want = (x.double() - m[:, :1].double()) * r[:, :1].double() * gamma[:1].double()
    want = want + beta[:1].double()
    torch.testing.assert_close(outs[0].double(), want, rtol=1e-5, atol=1e-5)
