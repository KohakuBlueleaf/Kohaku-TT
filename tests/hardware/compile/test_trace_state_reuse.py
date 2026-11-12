"""A replayed trace's in-place state handed a new tensor of the same layout (HF's
static cache, new each generate) takes over the trace's buffer; every other tensor
keeps its own values: the previous holder, views, alternating instances."""

import gc

import pytest
import torch
import torch._dynamo
from transformers import Qwen2ForCausalLM

import kohakutt
from kohakutt.bench.models.census import hf_config
from kohakutt.torch_backend import compiler as tt_compile
from kohakutt.torch_backend import trace as trace_module
from kohakutt.torch_backend.trace import Captured


def _step(state, x):
    state.add_(x)
    return state * 2


def _full(value):
    return torch.full((32, 32), float(value)).bfloat16()


def _tt(value):
    return _full(value).to("tt")


def _same(t, value):
    return torch.equal(t.cpu(), _full(value))


def _ptr(t):
    return t.untyped_storage().data_ptr()


def _ops(trace, *args):
    before = kohakutt.counters()["ops"]
    trace(*args)
    return kohakutt.counters()["ops"] - before


@pytest.mark.hw
def test_takeover_keeps_every_holders_values(native):
    a, x = _tt(0), _tt(1)
    trace = Captured(_step, [a, x], 1, static=[True, False])
    assert _same(a, 1)  # the prepare pass ran once
    b = _tt(10)
    assert torch.equal(trace(b, x).cpu(), _full(22))
    assert _same(b, 11) and _same(a, 1)  # a (the capture's tensor) moved off, kept
    ptr = _ptr(b)
    assert _ops(trace, b, x) <= 3  # x in, the replay, the output copy
    assert _same(b, 12) and _ptr(b) == ptr and _same(a, 1)
    # two tensors alternating through the trace keep their own counts
    for _ in range(2):
        trace(a, x)
        trace(b, x)
    assert _same(a, 3) and _same(b, 14) and _ptr(a) != _ptr(b)
    trace.release()


@pytest.mark.hw
def test_views_are_never_moved(native):
    a, x = _tt(0), _tt(1)
    trace = Captured(_step, [a, x], 1, static=[True, False])
    # a live view of the incoming tensor: copied in and out, the view follows
    c = _tt(5)
    view, ptr = c[0], _ptr(c)
    trace(c, x)
    assert _same(c, 6) and torch.equal(view.cpu(), _full(6)[0]) and _ptr(c) == ptr
    del view
    trace(c, x)  # now c alone: it takes the buffer over
    assert _same(c, 7) and _same(a, 1)
    # a live view of the holder: it cannot be moved, the call runs eagerly
    held_view = c[1]
    d = _tt(20)
    eager = trace.eager_calls
    assert torch.equal(trace(d, x).cpu(), _full(42))
    assert trace.eager_calls == eager + 1
    assert _same(d, 21) and _same(c, 7) and torch.equal(held_view.cpu(), _full(7)[1])
    trace.release()


@pytest.mark.hw
def test_module_buffers_of_two_instances(native):
    class Counter(torch.nn.Module):
        def __init__(self, start):
            super().__init__()
            self.register_buffer("count", _tt(start))

        def forward(self, x):
            return _step(self.count, x)

    first, second, x = Counter(0), Counter(100), _tt(1)
    trace = Captured(_step, [first.count, x], 1, static=[True, False])
    for _ in range(3):
        trace(second.count, x)
        trace(first.count, x)
    assert _same(first.count, 4) and _same(second.count, 103)
    trace.release()


@pytest.mark.hw
def test_dropped_trace_leaves_no_bookkeeping(native):
    x = _tt(1)
    dropped = Captured(_step, [_tt(0), x], 1, static=[True, False])
    del dropped  # never released
    gc.collect()
    a = _tt(0)
    trace = Captured(_step, [a, x], 1, static=[True, False])
    c = _tt(3)
    view = c[0]  # a view: refused whatever address c's storage got
    trace(c, x)
    assert _same(c, 4) and torch.equal(view.cpu(), _full(4)[0])
    del view
    lone = _tt(7)
    trace(lone, x)
    assert _ops(trace, lone, x) <= 3 and _same(lone, 9)
    trace.release()


@pytest.mark.hw
def test_hf_generate_replays_and_retained_caches_keep_values(native):
    tt_compile.register()
    torch._dynamo.reset()
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(hf_config("qwen")).bfloat16().eval().to("tt")
    model.generation_config.cache_implementation = "static"
    ids = torch.randint(0, 512, (1, 8), generator=torch.Generator().manual_seed(1))
    ids = ids.to("tt")

    def generate():
        return model.generate(
            ids, max_new_tokens=12, do_sample=False, return_dict_in_generate=True
        )

    with torch.no_grad():
        eager = generate().sequences.cpu()
        model.forward = torch.compile(model.forward, backend="ktt_tt", dynamic=False)
        first = generate()  # its cache is the decode trace's capture example
        kept = [(t.keys.cpu(), t.values.cpu()) for t in first.past_key_values.layers]
        for _ in range(2):
            assert torch.equal(generate().sequences.cpu(), eager)
        replays = sum(t.replays for t in tt_compile.CAPTURED)
        last = generate()
    calls = sum(t.replays for t in tt_compile.CAPTURED) - replays
    tt_compile.release()
    assert torch.equal(first.sequences.cpu(), eager)
    assert torch.equal(last.sequences.cpu(), eager)
    assert calls == 12  # prefill + 11 decode steps, each one replay
    for layer, (k, v) in zip(first.past_key_values.layers, kept, strict=True):
        assert torch.equal(layer.keys.cpu(), k) and torch.equal(layer.values.cpu(), v)
    for old, new in zip(first.past_key_values.layers, last.past_key_values.layers):
        assert _ptr(old.keys) != _ptr(new.keys)
    assert native.fallback_counts() == {}


@pytest.mark.hw
def test_steady_calls_skip_the_plan_until_an_input_changes(native, monkeypatch):
    """Calls with the same tensors as the last one replay without checking the inputs'
    storages again; a static input rebound since (set_) is planned again."""
    a, x = _tt(0), _tt(1)
    trace = Captured(_step, [a, x], 1, static=[True, False])
    b = _tt(10)
    trace(b, x)  # b takes the buffer over
    trace(b, x)  # checked once more, then kept as the steady call
    checks = []
    same = trace_module._same_storage
    monkeypatch.setattr(
        trace_module, "_same_storage", lambda *t: checks.append(1) or same(*t)
    )
    for _ in range(2):
        trace(b, x)
    assert checks == [] and _same(b, 14)
    b.set_(_tt(100))  # rebound behind the trace's back
    assert torch.equal(trace(b, x).cpu(), _full(202))
    assert checks and _same(b, 101)
    trace.release()
