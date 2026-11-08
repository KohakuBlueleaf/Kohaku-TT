"""Regression: placement on a 1x2 mesh. Module marks land as written, a default
column -> row Linear pair runs with no conversion and leaves a pending sum, and a
captured replay on a new sharded input equals the eager result bit for bit."""

import pytest
import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, kohakutt, d):
        super().__init__()
        self.norm = kohakutt.replicate(nn.LayerNorm(d))
        self.up = kohakutt.colwise(nn.Linear(d, 4 * d))
        self.down = kohakutt.rowwise(nn.Linear(4 * d, d))

    def forward(self, x):
        return x + self.down(nn.functional.gelu(self.up(self.norm(x))))


class Pair(nn.Module):
    """An unmarked up / down Linear pair (placed by the default pairing)."""

    def __init__(self, d):
        super().__init__()
        self.up_proj = nn.Linear(d, 4 * d, bias=False)
        self.down_proj = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.up_proj(x)))


@pytest.mark.hw
@pytest.mark.dual
def test_marks_land(native_module):
    kohakutt = native_module
    mesh = kohakutt.mesh([0, 1], shape=(1, 2))
    try:
        block = Block(kohakutt, 1024).bfloat16().to(mesh)
        got = {n: kohakutt.placement(p) for n, p in block.named_parameters()}
    finally:
        kohakutt.close()
    assert got == {
        "norm.weight": "Replicated",
        "norm.bias": "Replicated",
        "up.weight": "Sharded(dim=0)",
        "up.bias": "Sharded(dim=0)",
        "down.weight": "Sharded(dim=1)",
        "down.bias": "Replicated",
    }


@pytest.mark.hw
@pytest.mark.dual
def test_default_pair_one_reduce_scatter(native_module):
    kohakutt = native_module
    mesh = kohakutt.mesh([0, 1], shape=(1, 2))
    try:
        pair = Pair(1024).bfloat16().to(mesh)
        placements = (
            kohakutt.placement(pair.up_proj.weight),
            kohakutt.placement(pair.down_proj.weight),
        )
        x = kohakutt.replicate(torch.randn(256, 1024, dtype=torch.bfloat16).to(mesh))
        with torch.no_grad():
            pair(x)
            torch.tt.synchronize()
            before = torch.tt.mesh_stats()
            out = pair(x)
            torch.tt.synchronize()
            after = torch.tt.mesh_stats()
            pending = kohakutt.placement(out)
    finally:
        kohakutt.close()
    assert placements == ("Sharded(dim=0)", "Sharded(dim=1)")
    # The pair runs with no conversion; its output is a pending sum that one
    # reduce-scatter or all-reduce completes where it is consumed.
    assert pending == "Partial(sum)"
    assert {k: after[k] - before[k] for k in after if after[k] != before[k]} == {}


@pytest.mark.hw
@pytest.mark.dual
def test_replay_of_sharded_input_equals_eager(native_module):
    kohakutt = native_module
    mesh = kohakutt.mesh([0, 1], shape=(1, 2))
    try:
        torch.manual_seed(0)
        lin = nn.Linear(2048, 2048).bfloat16().to(mesh)
        gen = torch.Generator().manual_seed(1)
        x = torch.randn(1024, 2048, generator=gen, dtype=torch.bfloat16).to(mesh)
        x2 = torch.randn(1024, 2048, generator=gen, dtype=torch.bfloat16).to(mesh)
        sharded = kohakutt.placement(x)
        with torch.no_grad():
            step = kohakutt.capture(lambda t: lin(t) + t, x)
            replay = step(x2).cpu()
            eager = (lin(x2) + x2).cpu()
            step.release()
    finally:
        kohakutt.close()
    assert sharded == "Sharded(dim=0)"
    assert torch.equal(replay, eager)
