"""Regression: kohakutt.placement_log() on a 1x2 mesh. Nothing is recorded by default;
inside record_placements() each op call records its inputs' and output's placements,
the conversions (and whether a kept copy served them) and the reason."""

import pytest
import torch
from torch import nn

ROWS, DIM = 128, 64
PARTIAL, ROWS_SHARDED, WHOLE = "[R, P(sum)]", "[R, S(0)]", "[R, R]"


@pytest.fixture(scope="module")
def mesh(native_module):
    """The 1x2 mesh, open for the module."""
    device = native_module.mesh([0, 1], shape=(1, 2))
    try:
        yield device
    finally:
        native_module.close()


@pytest.fixture(autouse=True)
def knobs(native_module, monkeypatch):
    monkeypatch.setenv("KTT_MESH_SHARD_MIN", "1")
    monkeypatch.setenv("KTT_FALLBACK_WARN", "0")
    for name in ("KTT_MESH_CONV_CACHE", "KTT_MESH_CONV_FREE", "KTT_MESH_STRICT"):
        monkeypatch.delenv(name, raising=False)
    native_module.reset_placement_log()


def _pair(kohakutt, mesh):
    """(up, down, x): a column -> row Linear pair and a replicated [ROWS, DIM] input."""
    torch.manual_seed(0)
    up = kohakutt.colwise(nn.Linear(DIM, 4 * DIM, bias=False).bfloat16()).to(mesh)
    down = kohakutt.rowwise(nn.Linear(4 * DIM, DIM, bias=False).bfloat16()).to(mesh)
    x = kohakutt.replicate(torch.randn(ROWS, DIM).bfloat16().to(mesh))
    return up, down, x


def _one(log, name):
    """The single record whose op starts with `name`."""
    found = [r for r in log if r["op"].startswith(name)]
    assert len(found) == 1, (name, [r["op"] for r in log])
    return found[0]


@pytest.mark.hw
@pytest.mark.dual
def test_log_is_off_by_default(native_module, mesh):
    kohakutt = native_module
    up, down, x = _pair(kohakutt, mesh)
    with torch.no_grad():
        down(torch.relu(up(x))).pow(2)
        before = kohakutt.placement_log()
        with kohakutt.record_placements():
            torch.tanh(kohakutt.shard(x, 0))
        inside = kohakutt.placement_log()
        torch.tanh(kohakutt.shard(x, 0))
        after = kohakutt.placement_log()
        kohakutt.reset_placement_log()
        emptied = kohakutt.placement_log()
    assert before == []
    assert [r["op"].split(".")[0] for r in inside] == ["ktt::distribute", "tanh"]
    assert after == inside
    assert emptied == []


@pytest.mark.hw
@pytest.mark.dual
def test_log_names_placements_conversions_and_reasons(native_module, mesh):
    kohakutt = native_module
    up, down, x = _pair(kohakutt, mesh)
    scale = kohakutt.shard(torch.rand(ROWS, DIM).add(0.5).bfloat16().to(mesh), 0)
    with torch.no_grad(), kohakutt.record_placements():
        p = down(torch.relu(up(x)))
        p.pow(2)
        torch.tanh(p)
        p * scale
        torch.cumsum(p, 0)
        p + x
        kohakutt.shard(x, 0)
    log = kohakutt.placement_log()
    for record in log:
        assert set(record) == {"op", "inputs", "output", "conversions", "reason"}
    linears = [r for r in log if r["op"].startswith("linear")]
    # The pair: output features split, then input features split into a partial sum.
    assert [(r["inputs"], r["output"], r["reason"]) for r in linears] == [
        ([WHOLE, ROWS_SHARDED], "[R, S(1)]", "shard"),
        (["[R, S(1)]", "[R, S(1)]"], PARTIAL, "reduce"),
    ]
    assert all(r["conversions"] == [] for r in linears)
    completed = {
        "input": 0,
        "from": PARTIAL,
        "to": ROWS_SHARDED,
        "kind": "reduce-scatter",
    }
    power = _one(log, "pow")
    assert (power["inputs"], power["output"]) == ([PARTIAL], ROWS_SHARDED)
    assert power["conversions"] == [completed | {"reused": False}]
    assert power["reason"] == "shard"
    # The next consumers take the kept copy; the product does so because of it.
    assert _one(log, "tanh")["conversions"] == [completed | {"reused": True}]
    assert _one(log, "tanh")["reason"] == "shard"
    product = _one(log, "mul")
    assert product["inputs"] == [PARTIAL, ROWS_SHARDED]
    assert product["conversions"] == [completed | {"reused": True}]
    assert product["reason"] == "shard (memory)"
    gathered = _one(log, "cumsum")
    assert (gathered["inputs"], gathered["output"]) == ([PARTIAL], WHOLE)
    assert gathered["conversions"] == [
        {
            "input": 0,
            "from": PARTIAL,
            "to": WHOLE,
            "kind": "all-reduce",
            "reused": False,
        }
    ]
    assert gathered["reason"].startswith("fallback: ")
    added = _one(log, "add")
    assert (added["inputs"], added["output"]) == ([PARTIAL, WHOLE], WHOLE)
    assert added["conversions"][0]["kind"] == "all-reduce"
    assert added["reason"].startswith("partial + replicated")
    marked = _one(log, "ktt::distribute")
    assert (marked["inputs"], marked["output"]) == ([WHOLE], ROWS_SHARDED)
    assert marked["conversions"][0]["kind"] == "partition"
    assert marked["reason"] == "mark"


@pytest.mark.hw
@pytest.mark.dual
def test_log_names_the_size_floor(native_module, mesh, monkeypatch):
    kohakutt = native_module
    up, down, x = _pair(kohakutt, mesh)
    monkeypatch.delenv("KTT_MESH_SHARD_MIN")
    with torch.no_grad(), kohakutt.record_placements():
        down(torch.relu(up(x))).pow(2)
    power = _one(kohakutt.placement_log(), "pow")
    # Below the floor a partial sum completes to replicated, not to the cheaper shard.
    assert power["output"] == WHOLE
    assert power["conversions"][0]["kind"] == "all-reduce"
    assert power["reason"] == "replicated (size floor)"
