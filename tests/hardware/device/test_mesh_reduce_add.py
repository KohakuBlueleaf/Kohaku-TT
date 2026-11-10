"""Regression: on the 1x2 mesh a partial sum becomes replicated by one all-reduce
program, and `partial + replicated` runs inside that program (no reduce-scatter,
all-gather or separate add); both chips hold the same bits, the fp32 sum of the
partials (+ addend).
"""

import pytest
import torch


def _stats():
    torch.tt.synchronize()
    return torch.tt.mesh_stats()


@pytest.mark.hw
@pytest.mark.dual
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(("rows", "dim"), [(256, 512), (96, 160)])
def test_mesh_reduce_add(native_module, dtype, rows, dim):
    kohakutt = native_module
    mesh = kohakutt.mesh([0, 1], shape=(1, 2))
    got = {}
    try:
        torch.manual_seed(0)
        up = kohakutt.colwise(torch.nn.Linear(dim, 4 * dim, bias=False).to(dtype)).to(
            mesh
        )
        down = kohakutt.rowwise(torch.nn.Linear(4 * dim, dim, bias=False).to(dtype)).to(
            mesh
        )
        x = kohakutt.replicate(torch.randn(rows, dim).to(dtype).to(mesh))
        with torch.no_grad():
            partial = down(torch.relu(up(x)))
            placement = kohakutt.placement(partial)
            parts = [torch.ops.ktt.local_shard(partial, c).float() for c in range(2)]
            base = (parts[0] + parts[1])[:rows, :dim]
            for name, run, want in (
                ("reduce", lambda: torch.ops.ktt.distribute(partial, "R,R"), base),
                ("add", lambda: partial + x, base + x.cpu().float()),
                ("radd", lambda: x + partial, base + x.cpu().float()),
            ):
                before = _stats()
                ops = kohakutt.counters()["ops"]
                y = run()
                after = _stats()
                got[name] = (
                    kohakutt.placement(y),
                    {
                        k: after[k] - before[k]
                        for k in ("all_reduces", "reduce_scatters", "all_gathers")
                    },
                    kohakutt.counters()["ops"] - ops,
                    [
                        torch.ops.ktt.local_shard(y, c).float()[:rows, :dim]
                        for c in range(2)
                    ],
                    want.to(dtype).float(),
                )
    finally:
        kohakutt.close()
    assert placement == "Partial(sum)"
    for name, (placed, delta, ops, chips, want) in got.items():
        assert placed == "Replicated", name
        assert delta == {"all_reduces": 1, "reduce_scatters": 0, "all_gathers": 0}, (
            name,
            delta,
        )
        assert ops <= 1, (name, ops)
        assert torch.equal(chips[0], chips[1]), name
        if dtype == torch.float32:
            assert torch.equal(chips[0], want), name
        else:
            # One bf16 step: the packer's rounding of the fp32 sum.
            assert ((chips[0] - want).abs() <= want.abs() * 2.0**-7 + 1e-30).all(), name
