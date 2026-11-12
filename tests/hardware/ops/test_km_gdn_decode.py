"""The GDN decode step at the metal level (examples/kernels/dsl_metal/gdn_decode_km.py)
on the card: the state within fp32 rounding of float64 and o within bf16 rounding over
steps, no worse than transformers' step; the state written in place as an in-place op
(version, private base, refused views), carried through a capture and torch.compile."""

import functools
import importlib.util
from pathlib import Path

import pytest

EXAMPLE = (
    Path(__file__).resolve().parents[3] / "examples/kernels/dsl_metal/gdn_decode_km.py"
)


@functools.cache
def _example():
    spec = importlib.util.spec_from_file_location("gdn_decode_km_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.hw
@pytest.mark.parametrize(
    "seqs, heads", [(1, (16, 32)), (1, (16, 48)), (2, (16, 32))], ids=str
)
def test_steps_match_float64(native, seqs, heads):
    errors = _example().compare((seqs, *heads, 128, 128, 4))
    for step in range(4):
        # fp32 state: a few roundings per step; o: one bf16 rounding (2^-8).
        assert errors["state"][step] < 1e-6, errors
        assert errors["o"][step] <= 2**-8, errors
        assert errors["state"][step] <= errors["hf state"][step], errors
        assert errors["o"][step] <= errors["hf o"][step] + 2**-9, errors


def _steps(seed, steps, shape=(1, 16, 32, 128, 128)):
    m = _example()
    state, per_step, a_log, dt_bias = m.inputs((*shape, steps), seed)
    want, refs = state.double(), []
    for qkv, a, b in per_step:
        want, o = m.reference(want, qkv, a, b, a_log, dt_bias, shape[1])
        refs.append((want, o))
    return state, per_step, (a_log, dt_bias), refs


def _close(got, want, bound):
    return _example().relative(got, want) < bound


@pytest.mark.hw
def test_state_write_is_in_place_and_seen(native):
    import torch

    state, per_step, params, refs = _steps(1, 1)
    dev = [p.to("tt") for p in params]
    # A state whose base came from another tensor (copy_): the write stays private.
    source = state.to("tt")
    tt_state = torch.empty_like(source)
    tt_state.copy_(source)
    before = tt_state._version
    qkv, a, b = (x.to("tt") for x in per_step[0])
    torch.ops.kohakutt_example.gdn_decode_km(tt_state, qkv, a, b, *dev)
    assert tt_state._version > before
    assert _close(tt_state.cpu(), refs[0][0], 1e-6)
    assert torch.equal(source.cpu(), state)
    # A view that is not its whole storage is refused, not written through a copy.
    with pytest.raises(RuntimeError, match="written in place"):
        wide = torch.zeros(2, *state.shape[1:]).to("tt")
        torch.ops.kohakutt_example.gdn_decode_km(wide[1:], qkv, a, b, *dev)


@pytest.mark.hw
def test_capture_and_compile_carry_the_state(native):
    import torch

    import kohakutt

    state, per_step, params, refs = _steps(2, 6)
    dev = [p.to("tt") for p in params]
    op = torch.ops.kohakutt_example.gdn_decode_km
    steps = [[x.to("tt") for x in s] for s in per_step]
    # Capture: the state is used in place (a closure tensor), each replay advances it.
    held = state.to("tt")
    captured = kohakutt.capture(lambda q, a, b: op(held, q, a, b, *dev), *steps[0])
    held.copy_(state.to("tt"))
    for (q, a, b), (want, o) in zip(steps, refs, strict=True):
        got = captured(q, a, b)
        assert _close(held.cpu(), want, 1e-6) and _close(got.cpu(), o, 2**-8)
    captured.release()
    # torch.compile (ktt_tt): the schema's write makes the state an in-place input
    # (no static mark), through the eager calls, the capture and the replays.
    fn = torch.compile(lambda s, q, a, b: op(s, q, a, b, *dev), backend="ktt_tt")
    s = state.to("tt")
    for (q, a, b), (want, o) in zip(steps, refs, strict=True):
        got = fn(s, q, a, b)
        assert _close(s.cpu(), want, 1e-6) and _close(got.cpu(), o, 2**-8)
