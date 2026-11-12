"""A host read or write inside a trace capture raises an error naming the aten op and
the Python line; Hugging Face's host-side mask check is skipped in a capture."""

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM

import kohakutt
from kohakutt.bench.models.census import hf_config
from kohakutt.torch_backend.capture_errors import CaptureHostTransfer


def _x():
    g = torch.Generator().manual_seed(0)
    return torch.randn(2, 16, 32, generator=g).bfloat16().to("tt")


def _named(fn, x, op, line):
    with pytest.raises(CaptureHostTransfer) as caught:
        kohakutt.capture(fn, x, warmup=2).release()
    text = str(caught.value)
    assert f"aten op {op}" in text and line in text, text
    # the device is usable after the failed capture
    assert torch.equal((x * 2).cpu(), x.cpu() * 2)


@pytest.mark.hw
def test_item_names_op_and_line(native):
    def read(x):
        y = x * 2
        if y.sum().item() > 1e9:  # host read
            return y
        return y + 1

    with torch.no_grad():
        _named(read, _x(), "aten._local_scalar_dense", "y.sum().item() > 1e9")


@pytest.mark.hw
def test_cpu_names_op_and_line(native):
    def to_host(x):
        return (x * 2).cpu()  # host read

    with torch.no_grad():
        _named(to_host, _x(), "aten._to_copy", "(x * 2).cpu()")


@pytest.mark.hw
def test_constant_made_in_fn_captures(native):
    def scale(x):
        return x * torch.arange(32).bfloat16().to("tt")  # a constant made in fn

    x = _x()
    with torch.no_grad():
        want = scale(x).cpu()
        trace = kohakutt.capture(scale, x, warmup=2)
        got = [trace(x).cpu() for _ in range(2)]
        trace.release()
    assert all(torch.equal(g, want) for g in got)


@pytest.mark.hw
@pytest.mark.parametrize("stride", [1, 2])
def test_conv_weight_views_capture(native, stride):
    """F.conv1d and a conv2d of a weight view reuse the weight prepared before the
    capture (cached for the weight's storage)."""
    x = _x()
    w = (torch.randn(64, 16, 3) * 0.2).bfloat16().to("tt")

    def convs(x):
        return (
            F.conv1d(x, w, padding=1, stride=stride),
            F.conv2d(x[:, :, None], w[:, :, None], padding=(0, 1), stride=stride),
        )

    with torch.no_grad():
        want = [t.cpu() for t in convs(x)]
        trace = kohakutt.capture(convs, x, warmup=2)
        got = [[t.cpu() for t in trace(x)] for _ in range(2)]
        trace.release()
    assert all(torch.equal(a, b) for g in got for a, b in zip(g, want))


def _conv1d_step():
    torch.manual_seed(0)
    model = torch.nn.Conv1d(16, 64, 3, padding=1).bfloat16().to("tt")
    opt = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)

    def step(x, y):
        opt.zero_grad(set_to_none=False)
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
        return loss.detach()

    return model, step


@pytest.mark.hw
def test_conv1d_training_captured_matches_eager(native):
    """A captured conv1d training step sees every optimizer update of its weight: 3
    replays equal 3 eager steps, loss and weights."""
    g = torch.Generator().manual_seed(2)
    batches = [
        (
            torch.randn(2, 16, 32, generator=g).bfloat16().to("tt"),
            torch.randn(2, 64, 32, generator=g).bfloat16().to("tt"),
        )
        for _ in range(4)
    ]
    eager_model, eager = _conv1d_step()
    model, step = _conv1d_step()
    for _ in range(2):  # the capture's warm-up steps
        eager(*batches[0])
    captured = kohakutt.capture(step, *batches[0], warmup=2)
    try:
        for x, y in batches[1:]:
            assert torch.equal(captured(x, y).cpu(), eager(x, y).cpu())
    finally:
        captured.release()
    for want, got in zip(eager_model.parameters(), model.parameters(), strict=True):
        assert torch.equal(got.cpu(), want.cpu())


@pytest.mark.hw
def test_hf_forward_without_cache_captures(native):
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(hf_config("qwen")).bfloat16().eval().to("tt")
    ids = torch.randint(0, 512, (1, 8)).to("tt")

    def forward(ids):
        return model(ids, use_cache=False).logits

    with torch.no_grad():
        want = forward(ids).cpu()
        trace = kohakutt.capture(forward, ids, warmup=2)
        got = trace(ids).cpu()
        trace.release()
    assert torch.equal(got, want)
