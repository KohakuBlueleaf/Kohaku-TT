"""Regression: torch.compile on tt: compiled inference and training match eager."""

import pytest
import torch._dynamo

from kohakutt.bench.capture import compile_tt_check
from kohakutt.torch_backend import compiler as tt_compile


@pytest.mark.hw
def test_compiled_inference_matches_eager(native):
    tt_compile.register()
    torch._dynamo.reset()
    assert compile_tt_check.inference()["eq_eager"]
    tt_compile.release()


@pytest.mark.hw
@pytest.mark.parametrize("kind", ["mlp", "cnn"])
def test_compiled_training_matches_eager(native, kind):
    tt_compile.register()
    torch._dynamo.reset()
    eager, _ = compile_tt_check.train(None, 4, kind)
    compiled, _ = compile_tt_check.train("ktt_tt_aot", 4, kind)
    tt_compile.release()
    assert compiled == eager


class _Counter(torch.nn.Module):
    """Adds its call count to x and advances it in the graph (an in-graph counter)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("count", torch.zeros((), dtype=torch.int64))

    def forward(self, x):
        y = x + self.count.float()
        self.count.add_(1)
        return y


@pytest.mark.hw
def test_compiled_counter_starts_from_caller_state(native):
    tt_compile.register()
    torch._dynamo.reset()
    model = _Counter().to("tt")
    compiled = torch.compile(model, backend="ktt_tt")
    x = torch.randn(32, 32)
    got = [compiled(x.to("tt")).cpu() for _ in range(6)]
    count = int(model.count.cpu())
    tt_compile.release()
    for step, y in enumerate(got):
        assert torch.equal(y, x + step), step
    assert count == 6


@pytest.mark.hw
def test_compiled_python_int_input_is_not_baked(native):
    tt_compile.register()
    torch._dynamo.reset()
    compiled = torch.compile(lambda x, n: x * n + n, backend="ktt_tt", dynamic=True)
    x = torch.randn(32, 32)
    xt = x.to("tt")
    got = [compiled(xt, n).cpu() for n in (1, 2, 3, 3, 4, 5, 3)]
    tt_compile.release()
    for n, y in zip((1, 2, 3, 3, 4, 5, 3), got, strict=True):
        assert torch.equal(y, x * n + n), n


@pytest.mark.hw
def test_captured_int_input_change_runs_eagerly(native):
    x = torch.randn(32, 32)
    step = native.capture(lambda t, n: t * n, x.to("tt"), 2)
    assert torch.equal(step(x.to("tt"), 2).cpu(), x * 2)
    assert torch.equal(step(x.to("tt"), 5).cpu(), x * 5)
    assert (step.replays, step.eager_calls) == (1, 1)
    step.release()


class _State(torch.nn.Module):
    """A [1] call counter, a [4, 32] running sum and a weight nudged in place: all
    updated in the graph."""

    def __init__(self):
        super().__init__()
        self.register_buffer("count", torch.zeros(1))
        self.register_buffer("acc", torch.zeros(4, 32))
        self.weight = torch.nn.Parameter(torch.ones(32), requires_grad=False)

    def forward(self, x):
        y = (x + self.count) * self.weight
        self.count.add_(1)
        self.acc.mul_(0.5).add_(x)
        self.weight.add_(0.125)
        return y


@pytest.mark.hw
@pytest.mark.parametrize("backend", ["ktt_tt", "ktt_tt_aot"])
def test_compiled_state_updates_once_per_call(native, backend):
    """Each call, the capturing one included, applies the graph's in-place updates once
    (the capture records without running); the capturing call's output stays valid
    after later replays."""
    tt_compile.register()
    torch._dynamo.reset()
    eager, model = _State().to("tt"), _State().to("tt")
    compiled = torch.compile(model, backend=backend)
    xs = [torch.randn(4, 32) for _ in range(7)]
    outs, wants = [], []
    with torch.no_grad():
        for x in xs:
            outs.append(compiled(x.to("tt")))
            wants.append(eager(x.to("tt")).cpu())
            for name in ("count", "acc", "weight"):
                got, want = getattr(model, name), getattr(eager, name)
                assert torch.equal(got.cpu(), want.cpu()), name
    tt_compile.release()
    for got, want in zip(outs, wants, strict=True):
        assert torch.equal(got.cpu(), want)


@pytest.mark.hw
def test_compiled_with_0d_trainable_parameter(native):
    tt_compile.register()
    torch._dynamo.reset()
    net = torch.nn.Linear(32, 32).to("tt")
    net.temperature = torch.nn.Parameter(torch.tensor(2.0).to("tt"))
    compiled = torch.compile(lambda x: net(x) * net.temperature, backend="ktt_tt")
    x = torch.randn(4, 32).to("tt")
    for _ in range(4):
        got = compiled(x)
    want = net(x) * net.temperature
    tt_compile.release()
    assert torch.equal(got.detach().cpu(), want.detach().cpu())


@pytest.mark.hw
@pytest.mark.parametrize("kind", ["sgd", "adamw"])
def test_compiled_optimizer_step_matches_eager(native, kind):
    tt_compile.register()
    torch._dynamo.reset()

    def make():
        torch.manual_seed(0)
        params = [torch.nn.Parameter(torch.randn(64, 32).to("tt")) for _ in range(2)]
        if kind == "sgd":
            return params, torch.optim.SGD(params, lr=0.1, momentum=0.9)
        return params, torch.optim.AdamW(params, lr=0.1, capturable=True)

    (pe, oe), (pc, oc) = make(), make()
    step = torch.compile(oc.step, backend="ktt_tt")
    gen = torch.Generator().manual_seed(1)
    for _ in range(6):
        grads = [torch.randn(64, 32, generator=gen) for _ in range(2)]
        for p, q, g in zip(pe, pc, grads, strict=True):
            p.grad, q.grad = g.to("tt"), g.to("tt")
        oe.step()
        step()
        for p, q in zip(pe, pc, strict=True):
            assert torch.equal(q.detach().cpu(), p.detach().cpu())
    tt_compile.release()


@pytest.mark.hw
@pytest.mark.parametrize("backend", ["ktt_tt", "ktt_tt_aot"])
def test_capturing_call_output_viewing_an_input_survives_replays(native, backend):
    """The capturing call's outputs that view an input are not the trace's stable
    input buffers (later replays copy their inputs into those)."""
    tt_compile.register()
    torch._dynamo.reset()
    compiled = torch.compile(lambda x: (x.reshape(-1), x * 2), backend=backend)
    xs = [torch.randn(4, 32) for _ in range(5)]
    try:
        outs = [compiled(x.to("tt")) for x in xs]
    finally:
        tt_compile.release()
    for call, (x, (view, double)) in enumerate(zip(xs, outs, strict=True)):
        assert torch.equal(view.cpu(), x.reshape(-1)), call
        assert torch.equal(double.cpu(), x * 2), call


@pytest.mark.hw
def test_capturing_forward_saved_input_survives_next_forward(native):
    """Two forwards, then one backward: the capturing forward's input saved for
    backward is not overwritten by the next forward's replay."""
    tt_compile.register()
    torch._dynamo.reset()
    wt = torch.nn.Parameter(torch.randn(32, 32).to("tt"))

    def f(x):
        return (x @ wt).sum()

    compiled = torch.compile(f, backend="ktt_tt_aot")
    xs = [torch.randn(4, 32).to("tt") for _ in range(6)]
    grads = []
    try:
        for fn in (f, compiled):
            for a, b in zip(xs[::2], xs[1::2], strict=True):
                wt.grad = None
                (fn(a) + fn(b)).backward()
                grads.append(wt.grad.cpu())
    finally:
        tt_compile.release()
    for step, (want, got) in enumerate(zip(grads[:3], grads[3:], strict=True)):
        assert torch.equal(got, want), step


@pytest.mark.hw
def test_capture_whose_function_raises_releases_its_trace(native):
    calls = []

    def f(t):
        calls.append(t)
        if len(calls) == 2:  # the capture pass (warmup=1: the first is the prepare)
            raise ValueError("raised in the capture pass")
        return t * 2

    with pytest.raises(ValueError, match="capture pass"):
        native.capture(f, torch.randn(32, 32).to("tt"), warmup=1)
    y = torch.randn(1, 32, 8, 8)
    yt = y.to("tt")
    torch.ops.ktt._debug_to_nhwc(yt)  # refused while any trace is alive
    assert torch.equal(yt.cpu(), y)
