"""A trace's programs keep their circular-buffer region: activations allocated after the
capture stay out of it, so a replay cannot overwrite them (TT-Metal checks that region
only when a program is enqueued, and a replay enqueues nothing)."""

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.hw
def test_outputs_without_copy(native):
    """capture(copy_outputs=False) returns the trace's own output tensor on every call,
    with the replay's values, for inputs stored as Nhwc too."""
    g = torch.Generator().manual_seed(1)
    # Small integers: every product and sum is exact in bf16.
    a, b = (
        torch.randint(-8, 8, (2, 32, 8, 8), generator=g).bfloat16() for _ in range(2)
    )
    at, bt = a.to("tt"), b.to("tt")
    torch.ops.ktt._debug_to_nhwc(at)
    torch.ops.ktt._debug_to_nhwc(bt)
    step = native.capture(lambda x: x * 2.0 + 1.0, at, warmup=2, copy_outputs=False)
    try:
        first = step(at)
        assert torch.equal(first.cpu(), a * 2.0 + 1.0)
        second = step(bt)
        assert second is first
        assert torch.equal(second.cpu(), b * 2.0 + 1.0)
    finally:
        step.release()


@pytest.mark.hw
def test_replay_keeps_later_l1_activations(native):
    """A captured attention with large circular buffers, then activations that the
    placement policy would put in L1: every activation keeps its values over a replay,
    no L1 buffer lies inside the trace's circular-buffer region, and the replay's output
    is unchanged."""
    g = torch.Generator().manual_seed(0)
    q, k, v = (
        torch.randn(1, 2, 1024, 256, generator=g).bfloat16().to("tt") for _ in range(3)
    )
    step = native.Captured(lambda a: F.scaled_dot_product_attention(a, k, v), (q,), 2)
    try:
        first = step(q).cpu()
        base = torch.ones(2048, 2048, dtype=torch.bfloat16).to("tt")
        acts = [base + float(i + 1) for i in range(7)]
        torch.tt.synchronize()
        cb_end, lowest, _ = torch.ops.ktt._debug_trace_l1(step.trace_id)
        assert cb_end > 0, "the trace's circular-buffer region is unknown"
        assert lowest < 0 or lowest >= cb_end, (lowest, cb_end)
        again = step(q).cpu()
        torch.tt.synchronize()
        assert torch.equal(first, again)
        for i, a in enumerate(acts):
            assert bool((a.cpu() == float(i + 2)).all()), f"activation {i} changed"
    finally:
        step.release()
