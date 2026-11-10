"""Trace capture / replay of tt work (the torch.cuda.graphs counterpart):
step = capture(fn, *inputs) warms up and captures; step(*inputs) replays.
The contract (state, host transfers, output lifetime) is in docs/concepts/trace.md.
"""

import os

import torch
from torch.utils._pytree import tree_flatten, tree_map

from kohakutt.core.numerics import current_numerics

# TT-NN's trace allocation tracker (read by TT-Metal at startup): when on, each replay
# records how many live buffers were allocated while the trace was active.
TRACKING = os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"


class ReplayMismatch(RuntimeError):
    """A call whose host (CPU) tensor input differs from the captured value."""


class Captured:
    """A captured tt trace with stable input buffers and outputs copied out per call."""

    def __init__(self, fn, example_inputs, warmup, static=None, copy_outputs=True):
        """Runs fn `warmup` times eagerly (steady state: every program compiled, lazy
        state such as grads and optimizer buffers created), then captures it.

        static[i] marks input i as used in place (e.g. a parameter: never copied);
        other inputs get stable clones that each call copies into. The trace keeps the
        Numerics active at capture (self.numerics). copy_outputs=False makes each call
        return the trace's own output tensors, which the next replay overwrites.
        """
        self.fn = fn
        self.copy_outputs = copy_outputs
        self.numerics = current_numerics()
        static = static or [False] * len(example_inputs)
        self.static = list(static)
        # Stable buffers are contiguous: copy_into writes whole contiguous tensors, and
        # any input layout copies into them by value.
        self.inputs = [
            (
                x
                if keep or not isinstance(x, torch.Tensor)
                else x.clone(memory_format=torch.contiguous_format)
            )
            for x, keep in zip(example_inputs, static, strict=True)
        ]
        for _ in range(max(warmup, 1) - 1):
            fn(*self.inputs)

        # The last warm-up call also compiles the trace's state write-back copies.
        torch.ops.ktt.prepare_begin()
        try:
            fn(*self.inputs)
        finally:
            torch.ops.ktt.prepare_end()
        torch.tt.synchronize()
        self.trace_id = torch.ops.ktt.capture_begin()
        try:
            self.outputs = fn(*self.inputs)
        finally:
            self.held = torch.ops.ktt.capture_end()
        self.replays = 0
        self.unsafe = []

    def __call__(self, *inputs):
        """fn(*inputs) by replay; returns copies of the outputs."""
        if self.trace_id is None:
            raise RuntimeError("released trace")
        # Host tensors are part of the captured program by value (the trace never reads
        # them on device): a different value cannot be replayed.
        for dst, src in zip(self.inputs, inputs, strict=True):
            host = isinstance(src, torch.Tensor) and src.device.type == "cpu"
            if host and not (src.shape == dst.shape and torch.equal(src, dst)):
                raise ReplayMismatch("a host tensor input differs from the capture")
        # A static input replaced by another tensor (e.g. a new KV cache) is copied in,
        # and copied back out after the replay (the trace may update it in place).
        write_back = []
        for dst, src, keep in zip(self.inputs, inputs, self.static, strict=True):
            if keep:
                if src is not dst and not _same_storage(src, dst):
                    dst.copy_(src)
                    write_back.append((src, dst))
            elif (
                isinstance(src, torch.Tensor)
                and src is not dst
                and src.device.type != "cpu"
            ):
                torch.ops.ktt.copy_into(dst, src)
        if TRACKING:
            self.unsafe.append(torch.ops.ktt.unsafe_buffers(self.trace_id))
        torch.ops.ktt.replay(self.trace_id)
        self.replays += 1
        for src, dst in write_back:
            src.copy_(dst)
        if not self.copy_outputs:
            return self.outputs
        return tree_map(
            lambda t: t.clone() if isinstance(t, torch.Tensor) else t, self.outputs
        )

    def release(self):
        """Releases the trace and the device tensors it holds."""
        if self.trace_id is not None:
            torch.ops.ktt.release(self.trace_id)
        self.trace_id, self.outputs, self.inputs = None, None, []

    def __repr__(self):
        leaves = [
            t for t in tree_flatten(self.outputs)[0] if isinstance(t, torch.Tensor)
        ]
        return (
            f"Captured(trace={self.trace_id}, held={self.held}, outputs={len(leaves)})"
        )


def footprint(step):
    """What Captured `step` keeps on the device: tensors held and created in the
    capture, fences, and device bytes per chip fenced and left allocated."""
    held, created, fences, fenced, kept = torch.ops.ktt.trace_footprint(step.trace_id)
    return {
        "held": held,
        "created": created,
        "fences": fences,
        "fenced_bytes": fenced,
        "kept_bytes": kept,
    }


def _same_storage(a, b):
    """True when tensors a and b view the same storage at the same offset."""
    return (
        isinstance(a, torch.Tensor)
        and isinstance(b, torch.Tensor)
        and a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
        and a.storage_offset() == b.storage_offset()
    )


def capture(fn, *example_inputs, warmup=3, copy_outputs=True):
    """A Captured replay of fn(*inputs) built from tt tensors example_inputs, after
    `warmup` eager calls. copy_outputs=False returns the trace's own output tensors
    (no device copy per call); the next replay overwrites them."""
    return Captured(fn, example_inputs, warmup, copy_outputs=copy_outputs)
