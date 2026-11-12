"""Trace capture / replay of tt work (the torch.cuda.graphs counterpart):
step = capture(fn, *inputs) warms up and captures; step(*inputs) replays.
The contract (state, host transfers, output lifetime) is in docs/concepts/trace.md.
"""

import operator
import os
import threading
import weakref

import torch
from torch.utils._pytree import tree_flatten, tree_map

from kohakutt.core.numerics import current_numerics
from kohakutt.torch_backend.capture_errors import (
    CaptureHostTransfer,
    OpLocator,
    hf_tracing,
    named,
)

# TT-NN's trace allocation tracker (read by TT-Metal at startup): when on, each replay
# records how many live buffers were allocated while the trace was active.
TRACKING = os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
_VERSION = operator.attrgetter("_version")


# Live traces, for the aliases they hold of their buffers (_trace_aliases).
_LIVE = weakref.WeakSet()

# Captures in progress on this thread (prepare and capture passes), for the optimizer
# check.
_local = threading.local()


def capturing():
    """True while a Captured runs its prepare or capture pass on this thread."""
    return getattr(_local, "depth", 0) > 0


class ReplayMismatch(RuntimeError):
    """A call whose host (CPU) tensor input differs from the captured value."""


class Captured:
    """A captured tt trace with stable input buffers and outputs copied out per call."""

    def __init__(
        self,
        fn,
        example_inputs,
        warmup,
        static=None,
        copy_outputs=True,
        keep_prepared=False,
    ):
        """Runs fn `warmup` times eagerly, then captures it. static[i]: input i is used
        in place, others are copied into stable clones per call. copy_outputs=False
        returns the trace's own outputs (the next replay overwrites them);
        keep_prepared=True keeps the last warm-up call's outputs in self.prepared."""
        self.fn = fn
        self.copy_outputs = copy_outputs
        self.numerics = current_numerics()  # the Numerics the trace was captured in
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
        # Used-in-place state inputs (not Parameters) whose buffer a replacing tensor
        # may take over (_plan); holders: the tensor on each buffer now.
        self.adoptable = {
            i
            for i, (x, keep) in enumerate(zip(example_inputs, static, strict=True))
            if keep and _adoptable(x)
        }
        self.holders = {i: weakref.ref(example_inputs[i]) for i in self.adoptable}
        for _ in range(max(warmup, 1) - 1):
            fn(*self.inputs)

        _local.depth = getattr(_local, "depth", 0) + 1
        # HF code sees the passes as a CUDA graph capture (hf_tracing).
        with hf_tracing():
            try:
                self._capture(fn)
            finally:
                _local.depth -= 1
        self.prepared = self._prepared if keep_prepared else None
        del self._prepared
        # The trace's buffers of adoptable inputs, held by private aliases from here on.
        for i in self.adoptable:
            self.inputs[i] = self.inputs[i].detach()
        _LIVE.add(self)
        # The last call that rebound nothing: its static inputs (_steady_copies).
        self.steady = None
        self.dynamic = [i for i, keep in enumerate(self.static) if not keep]
        self.replays = 0
        self.eager_calls = 0
        self.unsafe = []

    def _capture(self, fn):
        """The prepare pass (one eager call) and the capture pass of fn."""
        # The last warm-up call also compiles the trace's state write-back copies.
        torch.ops.ktt.prepare_begin()
        try:
            self._prepared = fn(*self.inputs)
        finally:
            torch.ops.ktt.prepare_end()
        torch.tt.synchronize()
        self.trace_id = torch.ops.ktt.capture_begin()
        # Each op's Python line is recorded: a failure names the op and the line.
        locator = OpLocator()
        try:
            with locator:
                self.outputs = fn(*self.inputs)
        except BaseException as error:
            _drop_capture(self.trace_id)
            _raise_named(error, locator)
            raise
        try:
            self.held = torch.ops.ktt.capture_end()
        except RuntimeError as error:
            _raise_named(error, locator)
            raise

    def __call__(self, *inputs):
        """fn(*inputs) by replay; returns copies of the outputs."""
        if self.trace_id is None:
            raise RuntimeError("released trace")
        copies = self._steady_copies(inputs)
        if copies is not None:
            for i in copies:
                torch.ops.ktt.copy_into(self.inputs[i], inputs[i])
            return self._replay(())
        # Non-tensor inputs (a compiled graph's Python ints) are baked into the trace:
        # a call with other values runs fn eagerly instead of replaying.
        for dst, src in zip(self.inputs, inputs, strict=True):
            if _scalar(dst) and (type(src) is not type(dst) or src != dst):
                self.eager_calls += 1
                return self.fn(*inputs)
        # Host tensors are part of the captured program by value (the trace never reads
        # them on device): a different value cannot be replayed.
        for dst, src in zip(self.inputs, inputs, strict=True):
            host = isinstance(src, torch.Tensor) and src.device.type == "cpu"
            if host and not (src.shape == dst.shape and torch.equal(src, dst)):
                raise ReplayMismatch("a host tensor input differs from the capture")
        # A static input replaced by another tensor (e.g. a new KV cache) takes over the
        # trace's buffer, or is copied in and back out after the replay (_plan).
        plans = {
            i: self._plan(i, src, dst)
            for i, (dst, src, keep) in enumerate(
                zip(self.inputs, inputs, self.static, strict=True)
            )
            if keep and src is not dst and not _same_storage(src, dst)
        }
        if "eager" in plans.values():
            self.eager_calls += 1
            return self.fn(*inputs)
        write_back = []
        for i, (dst, src, keep) in enumerate(
            zip(self.inputs, inputs, self.static, strict=True)
        ):
            if i in plans:
                self._release_holder(i, dst)
                dst.copy_(src)
                if plans[i] == "adopt":
                    src.set_(dst)
                    self.holders[i] = weakref.ref(src)
                else:
                    write_back.append((src, dst))
            elif (
                not keep
                and isinstance(src, torch.Tensor)
                and src is not dst
                and src.device.type != "cpu"
            ):
                torch.ops.ktt.copy_into(dst, src)
        self.steady = None if plans else self._steady_record(inputs)
        return self._replay(write_back)

    def _replay(self, write_back):
        """Replays the trace, copies `write_back` (src, buffer) pairs out, returns the
        outputs."""
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

    def _steady_record(self, inputs):
        """The static inputs of a call that rebound nothing (weak references and tensor
        versions), for _steady_copies; None when a host tensor is among the inputs."""
        for x, keep in zip(inputs, self.static, strict=True):
            tensor = isinstance(x, torch.Tensor)
            if (tensor and x.device.type == "cpu") or (keep and not tensor):
                return None
            if keep and x.is_inference():  # no version counter
                return None
        index = [i for i, keep in enumerate(self.static) if keep]
        # itemgetter of 2+ indices returns a tuple (one index: listed twice).
        pick = (
            operator.itemgetter(*(index * 2 if len(index) == 1 else index))
            if index
            else _no_inputs
        )
        held = pick(inputs)
        return pick, [weakref.ref(x) for x in held], tuple(map(_VERSION, held))

    def _steady_copies(self, inputs):
        """The inputs to copy in when the static inputs are the steady record's tensors
        at the versions it saw (not changed in place or rebound since) and the others
        are device tensors or the captured values; None otherwise (full checks)."""
        if self.steady is None or len(inputs) != len(self.inputs):
            return None
        pick, refs, versions = self.steady
        held = pick(inputs)
        if not all(map(operator.is_, held, map(operator.call, refs))):
            return None
        if tuple(map(_VERSION, held)) != versions:
            return None
        copies = []
        for i in self.dynamic:
            src, dst = inputs[i], self.inputs[i]
            if src is dst:
                continue
            if _scalar(dst):
                if type(src) is not type(dst) or src != dst:
                    return None
            elif not isinstance(src, torch.Tensor) or src.device.type == "cpu":
                return None
            else:
                copies.append(i)
        return copies

    def _plan(self, i, src, buffer):
        """How src replaces buffer i: "adopt" (src takes the buffer over), "copy" (in
        before, out after the replay) or "eager" (the buffer has users that cannot be
        moved off it: a holder with views, or another trace's holder)."""
        if i not in self.adoptable:
            return "copy"  # Parameters and the like: the buffer is their own storage
        held = self.holders[i]()
        holder_on = held is not None and _same_storage(held, buffer)
        if _users(buffer) != (1 if holder_on else 0):
            return "eager"
        if (
            _same_layout(src, buffer)
            and _adoptable(src)
            and _users(src) == 1
            and torch.ops.ktt.placement(src) == torch.ops.ktt.placement(buffer)
        ):
            return "adopt"
        return "copy"

    def _release_holder(self, i, buffer):
        """Moves the tensor holding buffer i (if alive) to storage of its own with its
        values, so writes to the buffer no longer reach it."""
        if i not in self.adoptable:
            return
        held = self.holders[i]()
        if held is not None and _same_storage(held, buffer):
            held.set_(held.clone())
        self.holders[i] = _no_tensor

    def release(self):
        """Releases the trace and the device tensors it holds."""
        if self.trace_id is not None:
            torch.ops.ktt.release(self.trace_id)
        _LIVE.discard(self)
        self.trace_id, self.outputs, self.inputs = None, None, []
        self.steady = None

    def __repr__(self):
        leaves = [
            t for t in tree_flatten(self.outputs)[0] if isinstance(t, torch.Tensor)
        ]
        return (
            f"Captured(trace={self.trace_id}, held={self.held}, outputs={len(leaves)})"
        )


def _raise_named(error, locator):
    """Raises `error` named by its aten op and Python line when the locator knows the
    op (returns otherwise)."""
    if isinstance(error, CaptureHostTransfer) or not isinstance(error, Exception):
        return
    renamed = named(error, locator)
    if renamed is not error:
        raise renamed from error


def _drop_capture(trace_id):
    """Ends a capture whose function raised and releases its trace (an op error that
    ended it already released the trace)."""
    try:
        torch.ops.ktt.capture_end()
    except RuntimeError:  # the function's error is the one raised
        return
    torch.ops.ktt.release(trace_id)


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


def _scalar(x):
    """True for the non-tensor input values a call is checked against (numbers,
    strings, None)."""
    return x is None or isinstance(x, (int, float, str))


def _adoptable(t):
    """True for a tensor that can be rebound to a trace's buffer: a whole contiguous
    device storage, not a Parameter, no autograd."""
    return (
        isinstance(t, torch.Tensor)
        and t.device.type != "cpu"
        and not isinstance(t, torch.nn.Parameter)
        and not t.requires_grad
        and t._base is None
        and t.is_contiguous()
        and t.storage_offset() == 0
        and t.untyped_storage().nbytes() == t.numel() * t.element_size()
    )


def _no_inputs(inputs):
    """No static inputs of `inputs`."""
    return ()


def _no_tensor():
    """A holder reference to no tensor."""
    return None


def _trace_aliases(t):
    """The private aliases live traces hold of t's storage."""
    ptr = t.untyped_storage().data_ptr()
    return sum(
        1
        for trace in list(_LIVE)
        for j in trace.adoptable
        if trace.inputs and trace.inputs[j].untyped_storage().data_ptr() == ptr
    )


def _users(t):
    """Tensors (views, aliases) on t's storage, t included, besides traces' aliases."""
    storage = t.untyped_storage()
    refs = torch._C._storage_Use_Count(storage._cdata) - 1  # minus `storage` here
    return refs - _trace_aliases(t)


def _same_layout(a, b):
    """True when tensors a and b have the same shape, dtype and device."""
    return a.shape == b.shape and a.dtype == b.dtype and a.device == b.device


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
