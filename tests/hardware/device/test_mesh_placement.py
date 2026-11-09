"""Regression: placement on a 1x2 mesh. Module marks land as written, a default
column -> row Linear pair is split or replicated by its first call's rows and then runs
with no conversion (a split pair leaves a pending sum), and a captured replay on a new
sharded input equals the eager result bit for bit."""

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
def test_default_pair_one_reduce_scatter(native_module, monkeypatch):
    kohakutt = native_module
    monkeypatch.setenv("KTT_MESH_AUTO_PAIRS", "always")
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
@pytest.mark.parametrize(
    ("d", "rows", "want"),
    [
        (1024, 256, ("Replicated", "Replicated", "Replicated")),
        (2048, 1024, ("Sharded(dim=0)", "Sharded(dim=1)", "Partial(sum)")),
    ],
)
def test_default_pair_by_rows(native_module, monkeypatch, d, rows, want):
    kohakutt = native_module
    monkeypatch.delenv("KTT_MESH_AUTO_PAIRS", raising=False)
    mesh = kohakutt.mesh([0, 1], shape=(1, 2))
    try:
        torch.manual_seed(0)
        pair = Pair(d).bfloat16()
        gen = torch.Generator().manual_seed(1)
        x_host = torch.randn(rows, d, generator=gen, dtype=torch.bfloat16)
        want_out = pair(x_host).float()
        pair = pair.to(mesh)
        x = kohakutt.replicate(x_host.to(mesh))
        with torch.no_grad():
            out = pair(x)
            got = (
                kohakutt.placement(pair.up_proj.weight),
                kohakutt.placement(pair.down_proj.weight),
                kohakutt.placement(out),
            )
            torch.tt.synchronize()
            before = torch.tt.mesh_stats()
            pair(x)
            torch.tt.synchronize()
            after = torch.tt.mesh_stats()
            out = out.cpu().float()
    finally:
        kohakutt.close()
    # The first call decides the pair by its rows: split where the matmul time saved
    # exceeds the all-reduce, else replicated; later calls run with no conversion.
    assert got == want
    assert {k: after[k] - before[k] for k in after if after[k] != before[k]} == {}
    torch.testing.assert_close(out, want_out, rtol=0.05, atol=0.05)


class SeqBlock(nn.Module):
    """A Block whose forward marks its input sharded on the sequence dim."""

    def __init__(self, kohakutt, d):
        super().__init__()
        self.kohakutt = kohakutt
        self.block = Block(kohakutt, d)

    def forward(self, x):
        return self.block(self.kohakutt.shard(x, 0))


class Unnamed(nn.Module):
    """Three same-width Linears with names the default pairing does not know."""

    def __init__(self, d):
        super().__init__()
        self.a = nn.Linear(d, d)
        self.b = nn.Linear(d, d)
        self.c = nn.Linear(d, d)

    def forward(self, x):
        return self.c(self.b(torch.relu(self.a(x))))


class Reshard(nn.Module):
    """A colwise Linear whose forward asks its weight for another placement."""

    def __init__(self, kohakutt, d):
        super().__init__()
        self.kohakutt = kohakutt
        self.lin = kohakutt.colwise(nn.Linear(d, d, bias=False))

    def forward(self, x):
        return nn.functional.linear(x, self.kohakutt.shard(self.lin.weight, 1))


def _compile(module, **options):
    from kohakutt.torch_backend import compiler

    compiler.register()
    torch._dynamo.reset()
    return torch.compile(module, backend="ktt_tt_aot", options=options or None)


def _placements(kohakutt, module):
    return {n: kohakutt.placement(p) for n, p in module.named_parameters()}


@pytest.mark.hw
@pytest.mark.dual
def test_compiled_activation_mark_equals_eager(native_module):
    kohakutt = native_module
    mesh = kohakutt.mesh([0, 1], shape=(1, 2))
    try:
        torch.manual_seed(0)
        block = SeqBlock(kohakutt, 1024).bfloat16().to(mesh)
        compiled = _compile(block)
        gen = torch.Generator().manual_seed(1)
        xs = [
            kohakutt.replicate(
                torch.randn(512, 1024, generator=gen, dtype=torch.bfloat16).to(mesh)
            )
            for _ in range(6)
        ]
        with torch.no_grad():
            # Two eager calls, the capture, then replays.
            outs = [compiled(x) for x in xs]
            got = [o.cpu() for o in outs]
            got_placement = kohakutt.placement(outs[-1])
            want_out = block(xs[-1])
            want_placement = kohakutt.placement(want_out)
            want = [block(x).cpu() for x in xs]
    finally:
        torch._dynamo.reset()
        kohakutt.close()
    assert got_placement == want_placement
    for g, w in zip(got, want):
        assert torch.equal(g, w)


@pytest.mark.hw
@pytest.mark.dual
def test_compiled_activation_mark_backward(native_module):
    kohakutt = native_module
    mesh = kohakutt.mesh([0, 1], shape=(1, 2))
    try:
        torch.manual_seed(0)
        block = SeqBlock(kohakutt, 1024).to(mesh)
        x = kohakutt.replicate(torch.randn(256, 1024).to(mesh))
        block(x).square().mean().backward()
        want = {n: p.grad.cpu() for n, p in block.named_parameters()}
        block.zero_grad(set_to_none=True)
        _compile(block)(x).square().mean().backward()
        got = {n: p.grad.cpu() for n, p in block.named_parameters()}
    finally:
        torch._dynamo.reset()
        kohakutt.close()
    assert got.keys() == want.keys()
    for n in want:
        torch.testing.assert_close(got[n], want[n], rtol=1e-3, atol=1e-4, msg=n)


@pytest.mark.hw
@pytest.mark.dual
def test_mesh_plan_equals_marks_and_overrides_them(native_module):
    kohakutt = native_module
    mesh = kohakutt.mesh([0, 1], shape=(1, 2))
    plan = {"a": "colwise", "b": "rowwise", "c": "replicate"}
    want_placements = {
        "a.weight": "Sharded(dim=0)",
        "a.bias": "Sharded(dim=0)",
        "b.weight": "Sharded(dim=1)",
        "b.bias": "Replicated",
        "c.weight": "Replicated",
        "c.bias": "Replicated",
    }
    try:
        torch.manual_seed(0)
        state = Unnamed(1024).bfloat16().state_dict()
        gen = torch.Generator().manual_seed(1)
        xs = [
            kohakutt.replicate(
                torch.randn(256, 1024, generator=gen, dtype=torch.bfloat16).to(mesh)
            )
            for _ in range(4)
        ]

        def run(model, **options):
            compiled = _compile(model, **options)
            with torch.no_grad():
                out = [compiled(x).cpu() for x in xs]
            return out, _placements(kohakutt, model)

        # The plan on an unmarked model.
        planned = Unnamed(1024).bfloat16()
        planned.load_state_dict(state)
        got_plan, placed_plan = run(planned.to(mesh), mesh_plan=plan)
        # The same placements as marks in code.
        marked = Unnamed(1024).bfloat16()
        marked.load_state_dict(state)
        kohakutt.colwise(marked.a)
        kohakutt.rowwise(marked.b)
        kohakutt.replicate(marked.c)
        got_marks, placed_marks = run(marked.to(mesh))
        # Code marks that the plan overrides.
        overridden = Unnamed(1024).bfloat16()
        overridden.load_state_dict(state)
        for lin in (overridden.a, overridden.b, overridden.c):
            kohakutt.rowwise(lin)
        got_over, placed_over = run(overridden.to(mesh), mesh_plan=plan)
    finally:
        torch._dynamo.reset()
        kohakutt.close()
    assert placed_plan == want_placements
    assert placed_marks == want_placements
    assert placed_over == want_placements
    for p, m, o in zip(got_plan, got_marks, got_over):
        assert torch.equal(p, m)
        assert torch.equal(o, m)


@pytest.mark.hw
@pytest.mark.dual
def test_activation_mark_conflict_warns(native_module, monkeypatch):
    kohakutt = native_module
    mesh = kohakutt.mesh([0, 1], shape=(1, 2))
    monkeypatch.delenv("KTT_MESH_STRICT", raising=False)
    try:
        model = Reshard(kohakutt, 1024).bfloat16().to(mesh)
        x = kohakutt.replicate(torch.randn(256, 1024, dtype=torch.bfloat16).to(mesh))
        with torch.no_grad():
            with pytest.warns(UserWarning, match="marked S\\(0\\).*placed S\\(1\\)"):
                want = model(x).cpu()
            with pytest.warns(UserWarning, match="lin.weight is marked S\\(0\\)"):
                got = _compile(model)(x).cpu()
            weight = kohakutt.placement(model.lin.weight)
            monkeypatch.setenv("KTT_MESH_STRICT", "1")
            with pytest.raises(RuntimeError, match="KTT_MESH_STRICT=1"):
                model(x)
            with pytest.raises(Exception, match="KTT_MESH_STRICT=1"):
                _compile(model)(x)
    finally:
        torch._dynamo.reset()
        kohakutt.close()
    assert weight == "Sharded(dim=0)"
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


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
