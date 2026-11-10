"""A trace keeps the tensors alive at the end of its capture and fences the device
memory freed during it: the footprint stays near the eager peak, later allocations
cannot land where a replay writes, and several traces replay side by side."""

import os

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from kohakutt.torch_backend import compiler as tt_compile
from kohakutt.torch_backend.trace import footprint

CHIP = int(os.environ.get("KTT_TEST_CHIP", "0"))
HOLD = os.environ.get("KTT_TRACE_HOLD") == "1"

MB = 1 << 20
STEPS = 24


def _in_use():
    torch.tt.synchronize()
    free, total = torch.tt.mem_get_info()
    return total - free


class _Peak(TorchDispatchMode):
    """The highest device memory in use after any op."""

    def __init__(self):
        super().__init__()
        self.peak = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self.peak = max(self.peak, _in_use())
        return out


def _chain(t):
    """2 * STEPS intermediates of t's size; one is alive at a time."""
    for i in range(STEPS):
        t = t * 0.5 + float(i % 3)
    return t


def _host(seed, shape=(1024, 1024)):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(-8, 8, shape, generator=g).bfloat16()


def _poison(count=48):
    """Tensors of known values in the lowest free device memory."""
    base = torch.ones(1024, 1024, dtype=torch.bfloat16).to("tt")
    return [base] + [base + float(i) for i in range(1, count)]


def _check_poison(tensors):
    for i, t in enumerate(tensors):
        assert bool((t.cpu() == float(i + 1)).all()), f"tensor {i} was overwritten"


def _check_footprint(kohakutt):
    host = _host(0)
    x = host.to("tt")
    size = x.numel() * 2
    with torch.no_grad():
        _chain(x)
        base = _in_use()
        with _Peak() as mode:
            want = _chain(x).cpu()
        peak = mode.peak - base
        before = _in_use()
        step = kohakutt.capture(_chain, x, warmup=1)
        try:
            kept = _in_use() - before
            f = footprint(step)
            assert f["created"] >= 2 * STEPS
            if HOLD:
                assert f["held"] == f["created"], f
            else:
                assert f["fences"] >= 1 and f["fenced_bytes"] > 0
                assert f["held"] * 4 <= f["created"], f
                # The stable input buffer is allocated before the capture.
                assert kept - size <= 1.25 * (peak + size), (kept, peak, f)
            assert torch.equal(step(x).cpu(), want)
        finally:
            step.release()
        assert _in_use() <= before + size


def _check_replay_keeps_later_tensors(kohakutt):
    host = _host(1)
    x, x2 = host.to("tt"), (host * 0.5).to("tt")
    with torch.no_grad():
        want, want2 = _chain(x).cpu(), _chain(x2).cpu()
        step = kohakutt.capture(_chain, x, warmup=1)
        try:
            first = step(x).cpu()
            later = _poison()
            again = step(x).cpu()
            other = step(x2).cpu()
            _check_poison(later)
            assert torch.equal(first, want) and torch.equal(again, want)
            assert torch.equal(other, want2)
        finally:
            step.release()


def _check_two_traces(kohakutt):
    hosts = [_host(seed) for seed in (2, 3, 4)]

    def other(t):
        for i in range(STEPS // 2):
            t = t * 0.25 - float(i % 2)
        return t

    with torch.no_grad():
        xs = [h.to("tt") for h in hosts]
        wants = [(_chain(x).cpu(), other(x).cpu()) for x in xs]
        a = kohakutt.capture(_chain, xs[0], warmup=1)
        b = kohakutt.capture(other, xs[0], warmup=1)
        try:
            later = _poison(24)
            for _ in range(2):
                for x, (want_a, want_b) in zip(xs, wants, strict=True):
                    assert torch.equal(a(x).cpu(), want_a)
                    assert torch.equal(b(x).cpu(), want_b)
            _check_poison(later)
        finally:
            a.release()
            b.release()


def _check_compiled(kohakutt, backend):
    torch.manual_seed(0)
    layers = []
    for _ in range(6):
        layers += [torch.nn.Linear(1024, 1024), torch.nn.GELU()]
    model = torch.nn.Sequential(*layers).bfloat16().to("tt").eval()
    x = (_host(5) / 8).to("tt")
    tt_compile.register()
    torch._dynamo.reset()
    try:
        compiled = torch.compile(model, backend=backend)
        with torch.no_grad():
            want = model(x).cpu()
            for _ in range(tt_compile.EAGER_CALLS + 1):
                first = compiled(x).cpu()
            later = _poison()
            again = compiled(x).cpu()
            _check_poison(later)
        assert torch.equal(first, again)
        assert torch.allclose(first.float(), want.float(), atol=0.05, rtol=0.05)
    finally:
        tt_compile.release()
        torch._dynamo.reset()


def _check_fence_at_address():
    """A free range is reserved at its own address past a lower free block; an
    allocated one raises."""
    low, a, b, c = (
        torch.full((1024, 1024), float(i)).bfloat16().to("tt") for i in range(4)
    )
    torch.tt.synchronize()
    r_low, r_a, r_b = (torch.ops.ktt._debug_buffer_range(t) for t in (low, a, b))
    assert r_low[0] < r_b[0]
    del low, b
    torch.tt.synchronize()
    assert torch.ops.ktt._debug_fence(*r_b) >= 1
    with pytest.raises(RuntimeError, match="not free"):
        torch.ops.ktt._debug_fence(*r_a)
    # Nothing stays reserved: the next tensor takes the lowest free block.
    again = torch.zeros(1024, 1024, dtype=torch.bfloat16).to("tt")
    torch.tt.synchronize()
    assert torch.ops.ktt._debug_buffer_range(again) == r_low
    assert bool((a.cpu() == 1.0).all()) and bool((c.cpu() == 3.0).all())


CHECKS = {
    "footprint": _check_footprint,
    "later_tensors": _check_replay_keeps_later_tensors,
    "two_traces": _check_two_traces,
    "ktt_tt": lambda k: _check_compiled(k, "ktt_tt"),
    "ktt_tt_aot": lambda k: _check_compiled(k, "ktt_tt_aot"),
}


@pytest.mark.hw
@pytest.mark.parametrize("check", list(CHECKS))
def test_trace_one_chip(native_module, check):
    native_module.open(CHIP)
    try:
        CHECKS[check](native_module)
    finally:
        native_module.close()


@pytest.mark.hw
@pytest.mark.dual
@pytest.mark.parametrize("check", list(CHECKS))
def test_trace_mesh(native_module, check):
    native_module.mesh([0, 1], shape=(1, 2))
    try:
        CHECKS[check](native_module)
    finally:
        native_module.close()


@pytest.mark.hw
def test_fence_at_address(native_module):
    native_module.open(CHIP)
    try:
        _check_fence_at_address()
    finally:
        native_module.close()
