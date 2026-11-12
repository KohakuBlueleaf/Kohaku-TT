"""Regression: eager training on tt and captured training replay, in any order and
next to other live traces; a capture of a non-capturable AdamW step raises."""

import pytest
import torch
import torch.nn.functional as F

import kohakutt
from kohakutt.bench.capture import graph_check
from kohakutt.bench.models.train_check import Block, train
from kohakutt.torch_backend import compiler as tt_compile


@pytest.mark.hw
@pytest.mark.parametrize("kind", ["mlp", "block", "cnn"])
@pytest.mark.parametrize("foreach", [False, True])
def test_training_on_tt(native, kind, foreach):
    native.reset_fallback_counts()
    losses, _ = train("tt", 5, foreach, clip=foreach, kind=kind)
    assert losses[-1] < losses[0]
    assert native.fallback_counts() == {}


@pytest.mark.hw
def test_captured_function_matches_eager(native):
    assert graph_check.small()["replay_eq_eager"]


@pytest.mark.hw
@pytest.mark.parametrize("kind", ["mlp", "cnn"])
def test_captured_training_matches_eager(native, kind):
    record = graph_check.training(6, kind)
    assert record["max_rel_dev"] == 0.0
    assert record["loss_decreases"]
    assert record["replay_ms_per_step"] < record["eager_ms_per_step"]
    torch.tt.synchronize()


def _block_step(optimizer, **options):
    """A seeded fp32 Block on tt, its optimizer and a training step(x, y) -> loss."""
    torch.manual_seed(0)
    model = Block().to("tt")
    opt = optimizer(model.parameters(), lr=1e-2, **{"foreach": False, **options})

    def step(x, y):
        opt.zero_grad(set_to_none=False)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        loss.backward()
        opt.step()
        return loss.detach()

    return model, step


def _batches(steps):
    gen = torch.Generator().manual_seed(1)
    return [
        (torch.randn(4, 32, 64, generator=gen), torch.randint(0, 16, (4, 32)))
        for _ in range(steps)
    ]


@pytest.mark.hw
def test_capture_rejects_non_capturable_adamw(native):
    _, step = _block_step(torch.optim.AdamW)
    x, y = (t.to("tt") for t in _batches(1)[0])
    with pytest.raises(RuntimeError, match=r"AdamW\(\.\.\., capturable=True\)"):
        native.capture(step, x, y)


@pytest.mark.hw
@pytest.mark.parametrize(
    "optimizer, options",
    [
        (torch.optim.AdamW, {"capturable": True}),
        (torch.optim.AdamW, {"capturable": True, "foreach": True}),
        (torch.optim.SGD, {"momentum": 0.9}),
    ],
    ids=["adamw-capturable", "adamw-capturable-foreach", "sgd-momentum"],
)
def test_captured_training_bit_exact(native, optimizer, options):
    batches = [tuple(t.to("tt") for t in b) for b in _batches(21)]
    eager_model, eager = _block_step(optimizer, **options)
    model, step = _block_step(optimizer, **options)
    for _ in range(3):  # the capture's warm-up steps
        eager(*batches[0])
    captured = native.capture(step, *batches[0], warmup=3)
    try:
        for x, y in batches[1:]:
            assert torch.equal(captured(x, y).cpu(), eager(x, y).cpu())
    finally:
        captured.release()
    for want, got in zip(eager_model.parameters(), model.parameters(), strict=True):
        assert torch.equal(got.cpu(), want.cpu())


def _live_trace():
    """A captured step kept alive (later captures run while a trace is alive)."""
    x = torch.randn(32, 64).bfloat16().to("tt")
    step = kohakutt.capture(lambda t: t * 2 + 1, x)
    step(x)
    return step


@pytest.mark.hw
@pytest.mark.parametrize("kind", ["mlp", "cnn"])
def test_captured_training_next_to_a_live_trace(native, kind):
    live = _live_trace()
    try:
        record = graph_check.training(6, kind)
    finally:
        live.release()
    assert record["max_rel_dev"] == 0.0


@pytest.mark.hw
def test_captured_attention_block_alone(native):
    torch.manual_seed(0)
    model = Block().bfloat16().to("tt")
    x = torch.randn(4, 32, 64).bfloat16().to("tt")

    def forward(t):
        with torch.no_grad():
            return model(t)

    step = kohakutt.capture(forward, x)
    try:
        got, want = step(x).cpu(), forward(x).cpu()
    finally:
        step.release()
    assert torch.equal(got, want)


@pytest.mark.hw
def test_captured_training_after_compile(native):
    tt_compile.register()
    torch._dynamo.reset()
    net = torch.nn.Sequential(torch.nn.Linear(64, 48), torch.nn.GELU()).to("tt")
    compiled = torch.compile(net, backend="ktt_tt")
    x = torch.randn(8, 64).to("tt")
    try:
        with torch.no_grad():
            for _ in range(4):  # the compiled graph's trace is captured on call 3
                compiled(x)
        record = graph_check.training(6, "cnn")
    finally:
        tt_compile.release()
    assert record["max_rel_dev"] == 0.0
