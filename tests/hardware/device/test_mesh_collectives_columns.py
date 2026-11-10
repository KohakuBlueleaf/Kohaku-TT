"""fp32 partial sums on the 1x2 mesh reduce-scatter to sharded columns in one program
(no all-reduce plus partition), bit for bit equal to the CPU sum of the two chips'
partials and to one card's add of the same two values."""

import pytest
import torch


def _complete(kohakutt, mesh, lead, tokens, dim, spec):
    torch.manual_seed(0)
    up = kohakutt.colwise(torch.nn.Linear(dim, 4 * dim, bias=False)).to(mesh)
    down = kohakutt.rowwise(torch.nn.Linear(4 * dim, dim, bias=False)).to(mesh)
    gen = torch.Generator().manual_seed(1)
    x = kohakutt.replicate(torch.randn(*lead, tokens, dim, generator=gen).to(mesh))
    partial = down(torch.relu(up(x)))
    placement = kohakutt.placement(partial)
    parts = [
        torch.ops.ktt.local_shard(partial, c).cpu()[..., :tokens, :dim]
        for c in range(2)
    ]
    torch.tt.synchronize()
    before = torch.tt.mesh_stats()
    y = torch.ops.ktt.distribute(partial, spec)
    torch.tt.synchronize()
    after = torch.tt.mesh_stats()
    ran = {k: after[k] - before[k] for k in after if after[k] != before[k]}
    one_card = (
        kohakutt.replicate(parts[0].to(mesh)) + kohakutt.replicate(parts[1].to(mesh))
    ).cpu()
    return partial, placement, parts, y, ran, one_card


@pytest.mark.hw
@pytest.mark.dual
@pytest.mark.parametrize(
    ("lead", "tokens", "dim", "own"),
    [
        ((1,), 128, 256, True),
        ((), 128, 256, True),
        ((1,), 100, 192, True),
        ((), 37, 448, True),
        ((1,), 64, 160, False),
    ],
)
def test_fp32_reduce_scatter_columns(native_module, lead, tokens, dim, own):
    kohakutt = native_module
    mesh = kohakutt.mesh([0, 1], shape=(1, 2))
    column = len(lead) + 1
    try:
        with torch.no_grad():
            partial, placement, parts, y, ran, one_card = _complete(
                kohakutt, mesh, lead, tokens, dim, f"R,S({column})"
            )
            got, result = y.cpu(), kohakutt.placement(y)
    finally:
        kohakutt.close()
    want = parts[0] + parts[1]
    assert placement == "Partial(sum)"
    assert result == f"Sharded(dim={column})"
    assert torch.equal(got, want)
    assert torch.equal(one_card, want)
    if own:
        assert ran == {"reduce_scatters": 1}, ran
