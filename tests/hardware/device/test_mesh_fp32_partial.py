"""Regression: fp32 partial sums on the 1x2 mesh complete by Kohaku-TT's own collectives
(reduce-scatter to rows sharded, all-reduce to replicated), bit for bit equal to the
fp32 sum of the two chips' partials; also at tt-metal#38538's [1, 1, 8192, 392], where
TT-NN's reduce-scatter hangs."""

import pytest
import torch


@pytest.mark.hw
@pytest.mark.dual
@pytest.mark.parametrize(
    ("lead", "rows", "dim"), [((), 512, 1024), ((1, 1), 8192, 392)]
)
def test_fp32_reduce_scatter_exact(native_module, lead, rows, dim):
    kohakutt = native_module
    mesh = kohakutt.mesh([0, 1], shape=(1, 2))
    got = {}
    try:
        torch.manual_seed(0)
        up = kohakutt.colwise(torch.nn.Linear(dim, 4 * dim, bias=False)).to(mesh)
        down = kohakutt.rowwise(torch.nn.Linear(4 * dim, dim, bias=False)).to(mesh)
        gen = torch.Generator().manual_seed(1)
        x = kohakutt.replicate(torch.randn(*lead, rows, dim, generator=gen).to(mesh))
        with torch.no_grad():
            partial = down(torch.relu(up(x)))
            placement = kohakutt.placement(partial)
            parts = [torch.ops.ktt.local_shard(partial, c) for c in range(2)]
            want = (parts[0] + parts[1])[..., :rows, :dim]
            row_dim = len(lead)
            for name, spec in (("sharded", f"R,S({row_dim})"), ("replicated", "R,R")):
                torch.tt.synchronize()
                before = torch.tt.mesh_stats()
                y = torch.ops.ktt.distribute(partial, spec)
                torch.tt.synchronize()
                after = torch.tt.mesh_stats()
                got[name] = (
                    kohakutt.placement(y),
                    after["reduce_scatters"] - before["reduce_scatters"],
                    after["all_reduces"] - before["all_reduces"],
                    y.cpu(),
                )
    finally:
        kohakutt.close()
    assert placement == "Partial(sum)"
    assert got["sharded"][0] == f"Sharded(dim={row_dim})"
    assert got["replicated"][0] == "Replicated"
    for name, (_, scatters, reduces, value) in got.items():
        assert (scatters, reduces) == ((1, 0) if name == "sharded" else (0, 1)), name
        assert torch.equal(value, want), name
