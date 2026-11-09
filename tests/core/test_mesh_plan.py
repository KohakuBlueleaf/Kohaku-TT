"""mesh_plan parsing (no device): Dynamo source names to module paths, and plan entries
to placements."""

import pytest
import torch
from torch import nn

from kohakutt.torch_backend.mesh_compile import _kind, _path, graph_pairs, plan_marks


def test_source_names_to_paths():
    src = (
        "L['self']._modules['layers']._modules['0']._modules['fc1']"
        "._parameters['weight']"
    )
    assert _path(src) == "layers.0.fc1.weight"
    assert _path("L['m']._buffers['running_mean']") == "running_mean"
    assert _path("L['x']") is None


def test_styles_by_use():
    plan = {"layers.*.fc1": "colwise", "layers.*.fc2": "rowwise", "emb": "colwise"}
    assert plan_marks(plan, "layers.0.fc1.weight", "linear") == "S(0)"
    assert plan_marks(plan, "layers.0.fc1.bias", "linear") == "S(0)"
    assert plan_marks(plan, "layers.3.fc2.weight", "linear") == "S(1)"
    assert plan_marks(plan, "layers.3.fc2.bias", "linear") == "R"
    assert plan_marks(plan, "emb.weight", "embedding") == "S(1)"
    assert plan_marks(plan, "head.weight", "linear") is None


def test_explicit_values_and_first_match():
    plan = {
        "head.weight": "R",
        "head": "colwise",
        "norm*": "replicate",
        "w": {"weight": "shard(1)"},
    }
    assert plan_marks(plan, "head.weight", "linear") == "R"
    assert plan_marks(plan, "head.bias", "linear") == "S(0)"
    assert plan_marks(plan, "norm1.weight", None) == "R"
    assert plan_marks(plan, "w.weight", None) == "S(1)"
    assert plan_marks(plan, "w.bias", None) is None


def test_bad_values_raise():
    with pytest.raises(ValueError):
        plan_marks({"x": "colwise"}, "x.weight", None)
    with pytest.raises(ValueError):
        plan_marks({"x": "S(a)"}, "x.weight", "linear")


def _pairs(module, *inputs):
    """{tensor path: role} that graph_pairs gives `module`'s Linear weights."""
    out = {}

    def backend(gm, example_inputs):
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        paths = {n: _path(n.meta["grapharg"].source.name) for n in placeholders}
        params = {
            n: v
            for n, v in zip(placeholders, example_inputs)
            if paths[n] and paths[n].endswith("weight")
        }
        out.update({paths[w]: r for w, r in graph_pairs(gm, params).items()})
        return gm

    torch._dynamo.reset()
    torch.compile(module, backend=backend)(*inputs)
    return out


class _Mlp(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.l1 = nn.Linear(d, 4 * d)
        self.l2 = nn.Linear(4 * d, d)

    def forward(self, x):
        return x + self.l2(nn.functional.gelu(self.l1(self.norm(x))))


class _GatedMlp(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = nn.Linear(d, 2 * d, bias=False)
        self.b = nn.Linear(d, 2 * d, bias=False)
        self.c = nn.Linear(2 * d, d, bias=False)

    def forward(self, x):
        return self.c(nn.functional.silu(self.a(x)) * self.b(x))


class _Attention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(d, d)
        self.to_k = nn.Linear(d, d)
        self.to_v = nn.Linear(d, d)
        self.to_out = nn.Linear(d, d)

    def forward(self, x):
        b, s, d = x.shape

        def split(t):
            return t.view(b, s, self.heads, d // self.heads).transpose(1, 2)

        q, k, v = split(self.to_q(x)), split(self.to_k(x)), split(self.to_v(x))
        o = nn.functional.scaled_dot_product_attention(q, k, v)
        return self.to_out(o.transpose(1, 2).reshape(b, s, d))


class _Chain(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(d, d) for _ in range(3))

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


def test_graph_pairs():
    assert _pairs(nn.Sequential(_Mlp(8), _Mlp(8)), torch.randn(2, 8)) == {
        "0.l1.weight": "colwise",
        "0.l2.weight": "rowwise",
        "1.l1.weight": "colwise",
        "1.l2.weight": "rowwise",
    }
    assert _pairs(_GatedMlp(8), torch.randn(2, 8)) == {
        "a.weight": "colwise",
        "b.weight": "colwise",
        "c.weight": "rowwise",
    }
    assert _pairs(_Attention(16, 4), torch.randn(2, 5, 16)) == {
        "to_q.weight": "colwise",
        "to_k.weight": "colwise",
        "to_v.weight": "colwise",
        "to_out.weight": "rowwise",
    }
    # The last Linear reaches the graph output, so only the first pair is taken.
    assert _pairs(_Chain(8), torch.randn(2, 8)) == {
        "layers.0.weight": "colwise",
        "layers.1.weight": "rowwise",
    }


def test_apply_plan_sets_marks():
    from kohakutt.device.marks import apply_plan

    m = nn.Sequential(_Mlp(8), nn.Embedding(4, 8))
    assert (
        apply_plan(
            m, {"0.l1": "colwise", "0.l2": "rowwise", "1": "colwise", "0.norm": "R"}
        )
        is m
    )
    assert m[0].l1._ktt_marks == {"weight": "S(0)", "bias": "S(0)"}
    assert m[0].l2._ktt_marks == {"weight": "S(1)", "bias": "R"}
    assert m[1]._ktt_marks == {"weight": "S(1)"}
    assert m[0].norm._ktt_marks == {"weight": "R", "bias": "R"}


def test_conflict_warns_or_raises_under_strict(monkeypatch):
    from kohakutt.device.marks import conflict

    monkeypatch.delenv("KTT_MESH_STRICT", raising=False)
    with pytest.warns(UserWarning, match="placed S\\(1\\)"):
        conflict("kohakutt: w is marked S(0) (mark) and placed S(1) in forward")
    monkeypatch.setenv("KTT_MESH_STRICT", "1")
    with pytest.raises(RuntimeError, match="KTT_MESH_STRICT=1"):
        conflict("kohakutt: w is marked S(0) (mark) and placed S(1) in forward")


def test_kind_from_graph():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(10, 8)
            self.fc = nn.Linear(8, 8)

        def forward(self, i):
            return self.fc(self.emb(i))

    kinds = {}

    def backend(gm, example_inputs):
        for n in gm.graph.nodes:
            if n.op == "placeholder":
                path = _path(n.meta["grapharg"].source.name)
                if path:
                    kinds[path] = _kind(n)
        return gm

    torch.compile(M(), backend=backend)(torch.tensor([1, 2]))
    assert kinds == {
        "emb.weight": "embedding",
        "fc.weight": "linear",
        "fc.bias": "linear",
    }
