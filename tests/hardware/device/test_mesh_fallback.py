"""Regression: on a 1x2 mesh an op without a placed form for its input gathers the
input and runs replicated. It warns once per (op, placement), is counted, and gives
the CPU's result; KTT_MESH_STRICT=1 raises instead, naming the op, the placement and
what to do."""

import warnings

import pytest
import torch

ROWS, DIM = 128, 64


@pytest.fixture(scope="module")
def mesh(native_module):
    """The 1x2 mesh, open for the module."""
    device = native_module.mesh([0, 1], shape=(1, 2))
    try:
        yield device
    finally:
        native_module.close()


@pytest.fixture(autouse=True)
def knobs(monkeypatch):
    monkeypatch.delenv("KTT_MESH_STRICT", raising=False)
    monkeypatch.delenv("KTT_FALLBACK_WARN", raising=False)
    torch.tt.reset_mesh_stats()


def _silent(run):
    """run(), failing on any warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return run()


@pytest.mark.hw
@pytest.mark.dual
def test_fallback_warns_once_per_op_and_placement(native_module, mesh, monkeypatch):
    kohakutt = native_module
    x_host = torch.randn(ROWS, DIM, generator=torch.Generator().manual_seed(0))
    rows = kohakutt.shard(x_host.to(mesh), 0)
    cols = kohakutt.shard(x_host.to(mesh), 1)
    want = torch.cumsum(x_host, 0)
    with torch.no_grad():
        with pytest.warns(UserWarning, match=r"cumsum\S* gathers an input placed") as w:
            got = torch.cumsum(rows, 0).cpu()
        message = str(w[0].message)
        again = _silent(lambda: torch.cumsum(rows, 0).cpu())
        with pytest.warns(UserWarning, match=r"placed \[R, S\(1\)\]"):
            other = torch.cumsum(cols, 0).cpu()
        counts = torch.tt.mesh_fallback_ops()
        torch.tt.reset_mesh_stats()
        with pytest.warns(UserWarning, match=r"placed \[R, S\(0\)\]"):
            torch.cumsum(rows, 0)
        monkeypatch.setenv("KTT_FALLBACK_WARN", "0")
        torch.tt.reset_mesh_stats()
        _silent(lambda: torch.cumsum(rows, 0))
        quiet = torch.tt.mesh_fallback_ops()
    # The warning names the op, the placement, the way out and the strict knob.
    assert "placed [R, S(0)]" in message
    assert "kohakutt.replicate(x)" in message
    assert "KTT_MESH_STRICT=1" in message
    assert sum(counts.values()) == 3 and all("cumsum" in op for op in counts)
    assert sum(quiet.values()) == 1
    for value in (got, again, other):
        torch.testing.assert_close(value, want, rtol=1e-4, atol=1e-4)


@pytest.mark.hw
@pytest.mark.dual
def test_view_and_partial_write_fall_back(native_module, mesh):
    kohakutt = native_module
    x_host = torch.randn(ROWS, DIM, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        x = kohakutt.shard(x_host.to(mesh), 0)
        # A reshape that mixes the sharded dim with another one.
        with pytest.warns(UserWarning, match=r"tanh\S* gathers .* a view its shards"):
            viewed = torch.tanh(x.reshape(DIM, ROWS)).cpu()
        ones = kohakutt.replicate(torch.ones(2, DIM).to(mesh))
        with pytest.warns(UserWarning, match=r"copy_ .* a write into part of a tensor"):
            x[:2] = ones
        written = x.cpu()
        placed = kohakutt.placement(x)
    want = x_host.clone()
    want[:2] = 1.0
    torch.testing.assert_close(
        viewed, torch.tanh(x_host.reshape(DIM, ROWS)), rtol=1e-3, atol=1e-3
    )
    assert torch.equal(written, want)
    assert placed == "Replicated"


@pytest.mark.hw
@pytest.mark.dual
def test_strict_raises_with_the_way_out(native_module, mesh, monkeypatch):
    kohakutt = native_module
    x_host = torch.randn(ROWS, DIM, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        x = kohakutt.shard(x_host.to(mesh), 0)
        # Gathered once without the knob: the raise does not depend on a kept copy.
        with pytest.warns(UserWarning):
            torch.cumsum(x, 0)
        ones = kohakutt.replicate(torch.ones(2, DIM).to(mesh))
        monkeypatch.setenv("KTT_MESH_STRICT", "1")
        with pytest.raises(RuntimeError) as raised:
            torch.cumsum(x, 0)
        with pytest.raises(RuntimeError, match="a write into part of a tensor"):
            x[:2] = ones
        # Ops with a placed form, and the replicated input, run under the knob.
        placed = torch.tanh(x).cpu()
        got = _silent(lambda: torch.cumsum(kohakutt.replicate(x), 0).cpu())
        counts = torch.tt.mesh_fallback_ops()
    message = str(raised.value)
    assert "cumsum" in message
    assert "has no placed form for an input placed [R, S(0)]" in message
    assert "(KTT_MESH_STRICT=1)" in message
    assert "kohakutt.replicate(x)" in message
    # Only the call before the knob was counted.
    assert sum(counts.values()) == 1
    torch.testing.assert_close(placed, torch.tanh(x_host), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(got, torch.cumsum(x_host, 0), rtol=1e-4, atol=1e-4)
