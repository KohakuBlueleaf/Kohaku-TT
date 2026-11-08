"""Mesh placements on a 1x2 mesh: placed round trips are bit-exact, ops on placed
tensors fall back to a counted gather, KTT_MESH_STRICT=1 raises instead."""

import os

import pytest
import torch


@pytest.fixture(scope="module")
def mesh(native_module):
    kohakutt = native_module
    kohakutt.mesh([0, 1], shape=(1, 2))
    yield kohakutt
    kohakutt.close()


CASES = [
    ((64, 96), "R,S(0)"),
    ((64, 96), "R,S(1)"),
    ((50, 70), "R,S(0)"),  # uneven shards: padded tail
    ((3, 50, 70), "R,S(2)"),
    ((33,), "R,S(0)"),  # rank 1: device shape (1, 33)
    ((4, 64), "R,R"),
]


@pytest.mark.hw
@pytest.mark.dual
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int32])
@pytest.mark.parametrize("shape,spec", CASES)
def test_round_trip(mesh, shape, spec, dtype):
    if dtype == torch.int32:
        x = torch.randint(-1000, 1000, shape, dtype=dtype)
    else:
        x = torch.randn(shape).to(dtype)
    t = torch.ops.ktt.distribute(x.to("tt"), spec)
    back = t.cpu()
    assert torch.equal(back, x)
    if spec != "R,R":
        assert "S(" in torch.ops.ktt.placement(t)


@pytest.mark.hw
@pytest.mark.dual
def test_chips_hold_shards(mesh):
    x = torch.randn(50, 70)
    t = torch.ops.ktt.distribute(x.to("tt"), "R,S(1)")
    a = torch.ops.ktt.local_shard(t, 0)
    b = torch.ops.ktt.local_shard(t, 1)
    assert a.shape == b.shape == (50, 35)
    assert torch.equal(a, x[:, :35]) and torch.equal(b, x[:, 35:])
    t = torch.ops.ktt.distribute(x.to("tt"), "R,S(0)")
    b = torch.ops.ktt.local_shard(t, 1)
    assert b.shape == (25, 70) and torch.equal(b, x[25:])


@pytest.mark.hw
@pytest.mark.dual
def test_partial_sum_round_trip(mesh):
    x = torch.randn(32, 64)
    t = torch.ops.ktt.distribute(x.to("tt"), "R,P(sum)")
    assert torch.ops.ktt.placement(t) == "[R, P(sum)]"
    assert torch.equal(t.cpu(), x)


@pytest.mark.hw
@pytest.mark.dual
def test_redistribute(mesh):
    x = torch.randn(48, 80)
    a = torch.ops.ktt.distribute(x.to("tt"), "R,S(0)")
    b = torch.ops.ktt.distribute(a, "R,S(1)")
    assert torch.ops.ktt.placement(b) == "[R, S(1)]"
    assert torch.equal(b.cpu(), x)


@pytest.mark.hw
@pytest.mark.dual
def test_gather_fallback_counted(mesh):
    x = torch.randn(64, 64)
    t = torch.ops.ktt.distribute(x.to("tt"), "R,S(0)")
    torch.ops.ktt.reset_mesh_counters()
    y = (t * 2 + 1).cpu()
    torch.testing.assert_close(y, x * 2 + 1)
    gathers, nbytes, _, _ = torch.ops.ktt.mesh_counters()
    assert gathers == 1 and nbytes == 64 * 64 * 4
    # The cached replicated copy serves later reads.
    (t - 1).cpu()
    assert torch.ops.ktt.mesh_counters()[0] == 1


@pytest.mark.hw
@pytest.mark.dual
def test_partial_view_write(mesh):
    x = torch.randn(64, 64)
    t = torch.ops.ktt.distribute(x.to("tt"), "R,S(1)")
    t[:8] = 0
    x[:8] = 0
    assert torch.equal(t.cpu(), x)


@pytest.mark.hw
@pytest.mark.dual
def test_auto_placement(mesh, monkeypatch):
    monkeypatch.setenv("KTT_MESH_AUTO", "1")
    big = torch.randn(1024, 512)  # 2 MiB: sharded on its leading dim
    t = big.to("tt")
    assert torch.ops.ktt.placement(t) == "[R, S(0)]"
    assert torch.equal(t.cpu(), big)
    odd = torch.randn(513, 1024)  # leading dim odd: the largest dim that divides
    assert torch.ops.ktt.placement(odd.to("tt")) == "[R, S(1)]"
    small = torch.randn(64, 64)
    assert torch.ops.ktt.placement(small.to("tt")) == "[]"
    e = torch.empty(2048, 1024, device="tt")
    assert torch.ops.ktt.placement(e) == "[R, S(0)]"
    monkeypatch.setenv("KTT_MESH_AUTO", "0")
    assert torch.ops.ktt.placement(big.to("tt")) == "[]"


@pytest.mark.hw
@pytest.mark.dual
def test_mesh_memory_totals(mesh, monkeypatch):
    free, total = torch.tt.mem_get_info()
    assert total > 2 * 30 * 2**30  # two P150A chips of 32 GiB
    monkeypatch.setenv("KTT_MESH_AUTO", "1")
    torch.tt.synchronize()
    before = torch.tt.memory_allocated()
    x = torch.empty(4096, 1024, device="tt")  # 16 MiB, sharded: 8 MiB per chip
    assert torch.tt.memory_allocated() - before == 16 * 2**20
    assert torch.tt.memory_allocated(per_chip=True) * 2 == torch.tt.memory_allocated()
    del x


@pytest.mark.hw
@pytest.mark.dual
@pytest.mark.parametrize("draw", ["randn", "rand", "dropout"])
def test_rng_replicas_identical(mesh, draw):
    torch.tt.manual_seed(7)
    if draw == "randn":
        t = torch.randn(96, 160, device="tt")
    elif draw == "rand":
        t = torch.rand(96, 160, device="tt")
    else:
        t = torch.nn.functional.dropout(torch.ones(96, 160, device="tt"), 0.5)
    assert torch.equal(torch.ops.ktt.local_shard(t, 0), torch.ops.ktt.local_shard(t, 1))


@pytest.mark.hw
@pytest.mark.dual
@pytest.mark.skipif(os.environ.get("KTT_FABRIC") != "1", reason="needs KTT_FABRIC=1")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int32])
@pytest.mark.parametrize(
    "shape,spec",
    [((64, 96), "R,S(0)"), ((64, 96), "R,S(1)"), ((50, 70), "R,S(1)"),
     ((2, 3, 64, 64), "R,S(1)"), ((64, 64), "R,P(sum)")],
)
def test_device_gather(mesh, shape, spec, dtype):
    if dtype == torch.int32:
        x = torch.randint(-1000, 1000, shape, dtype=dtype)
    else:
        x = torch.randn(shape).to(dtype)
    t = torch.ops.ktt.distribute(x.to("tt"), spec)
    torch.ops.ktt.reset_mesh_counters()
    r = torch.ops.ktt.distribute(t, "R,R")
    a, b = torch.ops.ktt.local_shard(r, 0), torch.ops.ktt.local_shard(r, 1)
    _, _, device_gathers, failures = torch.ops.ktt.mesh_counters()
    assert torch.equal(a.reshape(shape), x) and torch.equal(b.reshape(shape), x)
    assert (device_gathers, failures) == (1, 0)


@pytest.mark.hw
@pytest.mark.dual
def test_strict_raises(mesh, monkeypatch):
    monkeypatch.setenv("KTT_MESH_STRICT", "1")
    t = torch.ops.ktt.distribute(torch.randn(64, 64).to("tt"), "R,S(0)")
    with pytest.raises(RuntimeError, match="KTT_MESH_STRICT"):
        (t + 1).cpu()
