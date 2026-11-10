"""Mesh: empty tensors: placing a tensor with a zero-size dim on a mesh."""

import pytest
import torch

pytestmark = [pytest.mark.hw, pytest.mark.dual]


@pytest.mark.parametrize(
    ("shape", "dim"), [((0, 64), 0), ((0, 64), 1), ((64, 0), 0), ((64, 0), 1)]
)
def test_shard_empty(native_module, mesh, shape, dim):
    """Sharding an empty tensor on any dim returns an empty tensor of the same shape in
    the requested placement, which ops and replicate() accept."""
    k = native_module
    x = torch.zeros(shape, dtype=torch.bfloat16).to(mesh)
    s = k.shard(x, dim)
    assert k.placement(s) == f"Sharded(dim={dim})"
    assert tuple(s.cpu().shape) == shape
    assert tuple((s + 1).cpu().shape) == shape
    r = k.replicate(s)
    assert k.placement(r) == "Replicated"
    assert tuple(r.cpu().shape) == shape


@pytest.mark.parametrize("shape", [(0, 64), (64, 0), (0,)])
def test_replicate_and_ops_on_empty(native_module, mesh, shape):
    """An empty replicated tensor survives an add and a cast."""
    x = native_module.replicate(torch.zeros(shape, dtype=torch.bfloat16).to(mesh))
    assert tuple((x + 1).cpu().shape) == shape
    assert tuple(x.float().cpu().shape) == shape
