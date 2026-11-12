"""Errors of a trace capture named by the aten op and the Python line that caused them:
a host read or write inside a capture raises CaptureHostTransfer (patterns and their
workarounds: docs/concepts/trace.md, "Host transfers in a capture")."""

import contextlib
import linecache
import os
import re
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode

# Messages of a host transfer reached inside a capture (TT-Metal's and the bridge's).
HOST_TRANSFER = re.compile(
    r"(Reads|Writes|Event Synchronization) (are|is) not supported during trace capture"
    r"|host (download|upload) inside a trace capture"
)
# The op number a queued bridge op's error carries ("[bridge op 'conv2d' #57]").
_SEQ = re.compile(r"\[bridge op '[^']*' #(\d+)\]")

_SKIP = tuple(
    os.path.dirname(m.__file__) + os.sep
    for m in (torch, sys.modules[__name__.rsplit(".", 1)[0]])
)


class CaptureHostTransfer(RuntimeError):
    """A host read or write inside a trace capture, named by aten op and Python line."""


def _where():
    """The innermost Python frame outside torch and the trace layer, as text; a
    compiled graph's line is mapped to the source line it was traced from."""
    frame = sys._getframe(1)
    while frame is not None and frame.f_code.co_filename.startswith(_SKIP):
        frame = frame.f_back
    if frame is None:
        return "unknown"
    name, line = frame.f_code.co_filename, frame.f_lineno
    code = linecache.getline(name, line).strip()
    if name.startswith("<eval_with_key>"):
        lines = linecache.getlines(name)
        for text in reversed(lines[:line]):
            if "# File: " in text:
                return text.split("# File: ", 1)[1].strip()
        source = _node_source(frame.f_locals.get("self"), code.split(" = ")[0])
        if source:
            return source
    return f"{name}:{line}" + (f" `{code}`" if code else "")


def _node_source(module, node_name):
    """The user line a graph node of `module` (an fx GraphModule) was traced from, per
    its recorded stack trace, or None."""
    graph = getattr(module, "graph", None)
    for node in getattr(graph, "nodes", ()):
        if node.name == node_name:
            lines = (node.meta.get("stack_trace") or "").strip().splitlines()
            files = [
                i for i, text in enumerate(lines) if text.lstrip().startswith("File")
            ]
            if files:
                at = lines[files[-1]].strip()
                code = (
                    lines[files[-1] + 1].strip() if files[-1] + 1 < len(lines) else ""
                )
                return at + (f" `{code}`" if code else "")
            return None
    return None


class OpLocator(TorchDispatchMode):
    """Records, per aten op run in a capture pass, the bridge op numbers it queued and
    where it was called, so a failing op's error can name both."""

    def __init__(self):
        super().__init__()
        self.ops = []  # (first op number, aten op, location)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        first = torch.ops.ktt.op_seq()
        self.ops.append((first, func, _where()))
        index = len(self.ops) - 1
        try:
            return func(*args, **(kwargs or {}))
        except Exception as error:
            text = str(error)
            if HOST_TRANSFER.search(text):
                # A queued op's error surfaces at a later op: its number names it.
                at = None if _SEQ.search(text) else index
                found = named(error, self, at)
                if found is not error:
                    raise found from error
            raise

    def find(self, seq):
        """Index of the op that queued bridge op number `seq`, or None."""
        found = None
        for i, (first, _, _) in enumerate(self.ops):
            if first < 0 or first > seq:
                break
            found = i
        return found


def named(error, locator, index=None):
    """`error` with the aten op and Python line that caused it (CaptureHostTransfer for
    a host transfer); `error` itself when nothing names it."""
    text = str(error)
    if index is None:
        match = _SEQ.search(text)
        index = locator.find(int(match.group(1))) if match else None
    if index is None:
        return error
    _, func, where = locator.ops[index]
    kind = CaptureHostTransfer if HOST_TRANSFER.search(text) else type(error)
    if kind is not CaptureHostTransfer and not isinstance(error, RuntimeError):
        return error
    head = (
        "a host read or write inside a trace capture"
        if kind is CaptureHostTransfer
        else "an op failed inside a trace capture"
    )
    if kind is CaptureHostTransfer:
        head += " (docs/concepts/trace.md, 'Host transfers in a capture')"
    try:
        return kind(f"{head}: aten op {func} at {where}\n{text}")
    except TypeError:  # an error type that does not take one message
        return error


@contextlib.contextmanager
def hf_tracing():
    """Hugging Face's `is_tracing` reads True, as under a CUDA graph capture: the host
    checks it guards (packed-sequence detection in the masks) are skipped."""
    tools = sys.modules.get("transformers.utils.import_utils")
    original = getattr(tools, "is_tracing", None)
    patched = []
    if original is not None:

        def tracing(tensor=None):
            return True

        for name, module in list(sys.modules.items()):
            if name.startswith("transformers") and (
                getattr(module, "is_tracing", None) is original
            ):
                module.is_tracing = tracing
                patched.append(module)
    try:
        yield
    finally:
        for module in patched:
            module.is_tracing = original
