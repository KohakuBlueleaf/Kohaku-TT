"""Tenstorrent chips as torch devices: device(chip) is a one-chip mesh, mesh(chips)
a multi-chip one (tensors replicated across its chips); both return a torch.device.
One mesh is live at a time for now."""

import contextlib
import os
import signal
import sys
import threading

import torch

from kohakutt.torch_backend import registration, varlen

# The live mesh: {"chips": [...], "shape": (rows, cols)}, empty when none is open.
_LIVE = {}

# Signals the process closes the mesh for, with the handler each replaced.
_SIGNALS = (signal.SIGTERM,)
_REPLACED = {}
# True inside open and close; a signal arriving then waits in _PENDING.
_BUSY = []
_PENDING = []


def _on_signal(signum, frame):
    """Closes the mesh, restores the handler it replaced and re-raises `signum`. Runs
    only between Python instructions of the main thread, never inside a native call."""
    if _BUSY:
        _PENDING.append(signum)
        return
    _restore_handlers()
    with contextlib.suppress(Exception):
        torch.ops.ktt.mark_exiting(True)
    os.kill(os.getpid(), signum)


def _install_handlers():
    """Takes over the signals still at their default; a handler the user set stays."""
    if threading.current_thread() is not threading.main_thread():
        return
    for signum in _SIGNALS:
        if signum not in _REPLACED and signal.getsignal(signum) is signal.SIG_DFL:
            _REPLACED[signum] = signal.signal(signum, _on_signal)


def _restore_handlers():
    """Puts back each replaced handler that is still ours."""
    for signum, previous in list(_REPLACED.items()):
        if signal.getsignal(signum) is _on_signal:
            signal.signal(signum, previous)
        del _REPLACED[signum]


@contextlib.contextmanager
def _guarded():
    """Holds signals that arrive in the block until it ends, then handles them."""
    _BUSY.append(1)
    try:
        yield
    finally:
        _BUSY.pop()
        if _PENDING and not _BUSY:
            _on_signal(_PENDING.pop(), None)


def _require_native():
    """Raises RuntimeError when the native layer is not built."""
    if not registration.install():
        raise RuntimeError(
            "kohakutt: the native layer is not built "
            "(cmake -S src/kohakutt_native -B build/native; ninja -C build/native)"
        )


def mesh(chips, shape=None):
    """The torch device of chips `chips` (a sequence of ids) as one mesh of `shape`
    (rows, cols; default 1 x len(chips)); opens it on first use, same chips again
    return the same device."""
    _require_native()
    chips = list(chips)
    shape = tuple(shape or (1, len(chips)))
    if _LIVE:
        if _LIVE["chips"] == chips and _LIVE["shape"] == shape:
            return torch.device("tt", 0)
        raise RuntimeError(
            f"kohakutt: chips {_LIVE['chips']} are already the live mesh; one mesh at "
            "a time for now (close() it first)"
        )
    with _guarded():
        _install_handlers()
        try:
            torch.ops.ktt.open(chips, *shape)
        except RuntimeError:
            _restore_handlers()
            raise
        _LIVE.update(chips=chips, shape=shape)
    return torch.device("tt", 0)


def device(chip=0):
    """The torch device of one chip (a 1x1 mesh); see mesh()."""
    return mesh([chip])


def live_index(device=None):
    """The tt index of `device` (default: the live mesh); raises ValueError unless it is
    a live mesh."""
    if not _LIVE:
        raise ValueError("kohakutt: no mesh is live (kohakutt.device() / mesh() first)")
    if device is None:
        return 0
    device = torch.device(device)
    if device.type != "tt" or device.index not in (0, None):
        raise ValueError(f"kohakutt: {device} is not a live tt mesh")
    return 0


def open(device_ids=0, mesh_shape=None):
    """Older form of mesh() / device(): opens `device_ids` (an int or a sequence)."""
    ids = [device_ids] if isinstance(device_ids, int) else list(device_ids)
    return mesh(ids, mesh_shape)


def close(device=None):
    """Closes mesh `device` (default the live one), releasing every trace
    (torch.compile's included); its tt tensors must be gone first. Process exit does the
    same."""
    _require_native()
    live_index(device)
    with _guarded():
        compiler = sys.modules.get("kohakutt.torch_backend.compiler")
        if compiler is not None:
            compiler.release()
        varlen.release()
        torch.ops.ktt.close()
        _LIVE.clear()
        _restore_handlers()


def synchronize(device=None):
    """Waits until all queued work on mesh `device` (default the live one) is done."""
    live_index(device)
    torch.ops.ktt.synchronize()


def clear_program_cache(device=None):
    """Drops TT-NN's compiled programs on mesh `device` (default the live one) with the
    device buffers they hold (conv metadata in L1_SMALL grows per input shape); they
    rebuild from the on-disk kernel cache on next use."""
    live_index(device)
    torch.ops.ktt.clear_program_cache()
