"""With KTT_MESH_FUSE=1 on the 1x2 mesh, `partial + replicated` followed by a linear on
the sum runs as one program (the all-reduce in column order, the matmul behind it) and
gives the values of the two separate programs."""

import pytest
import torch


@pytest.mark.hw
@pytest.mark.dual
def test_mesh_fused_linear(native_module, monkeypatch):
    kohakutt = native_module
    rows, dim, blocks = 256, 512, 3
    torch.manual_seed(0)
    ups = [torch.nn.Linear(dim, 2 * dim, bias=False).bfloat16() for _ in range(blocks)]
    downs = [
        torch.nn.Linear(2 * dim, dim, bias=False).bfloat16() for _ in range(blocks)
    ]
    x = torch.randn(rows, dim).bfloat16()
    ref = x.double()
    for up, down in zip(ups, downs):
        ref = ref + torch.relu(ref @ up.weight.double().T) @ down.weight.double().T

    mesh = kohakutt.mesh([0, 1], shape=(1, 2))
    got = {}
    try:
        ups_d = [kohakutt.colwise(m).to(mesh) for m in ups]
        downs_d = [kohakutt.rowwise(m).to(mesh) for m in downs]
        xd = kohakutt.replicate(x.to(mesh))

        def forward():
            before = torch.tt.mesh_stats()["fused_linears"]
            h = xd
            with torch.no_grad():
                for up, down in zip(ups_d, downs_d):
                    h = h + down(torch.relu(up(h)))
            fused = torch.tt.mesh_stats()["fused_linears"] - before
            chips = [
                torch.ops.ktt.local_shard(h, c).float()[:rows, :dim] for c in range(2)
            ]
            return chips, fused

        # The first forward creates the weights' local views; later ones can fuse.
        forward()
        got["separate"] = forward()
        monkeypatch.setenv("KTT_MESH_FUSE", "1")
        got["fused"] = forward()
    finally:
        kohakutt.close()
    for name, (chips, _) in got.items():
        assert torch.equal(chips[0], chips[1]), name
        rel = float((chips[0].double() - ref).norm() / ref.norm())
        assert rel <= 1e-2, (name, rel)
    # Blocks 2 and 3: the all-reduce of the block before runs inside their up matmul.
    assert got["separate"][1] == 0
    assert got["fused"][1] == blocks - 1
