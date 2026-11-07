"""torch.compile backends for tt: "ktt_tt" replays each Dynamo graph, "ktt_tt_aot"
each AOTAutograd forward / backward graph, as tt traces; persistent and in-place-updated
inputs are used in place, others copied per call.
"""

import operator

import torch
import torch._dynamo
import torch.fx
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd

from kohakutt.core.numerics import current_numerics
from kohakutt.fusion.fx import fuse_pointwise
from kohakutt.torch_backend.trace import Captured, ReplayMismatch

# Warm-up calls before capture: compiled graphs are functional, one call compiles all.
WARMUP = 1

# Calls each graph runs eagerly before its trace is captured, so state allocated in the
# first steps (e.g. optimizer moments) exists before capture (docs/design/trace.md).
EAGER_CALLS = 2

# Graphs whose trace has been captured (for inspection and release).
CAPTURED = []


def _is_static(x):
    """True for persistent tensors used in place by a trace (never copied): parameters
    and tensors marked with torch._dynamo.mark_static_address (e.g. static KV caches,
    which graphs update in place)."""
    if not isinstance(x, torch.Tensor):
        return False
    return isinstance(x, torch.nn.Parameter) or hasattr(x, "_dynamo_static_input_type")


def _mutated_inputs(gm):
    """Indices of gm's placeholders that the graph writes in place (an in-place
    method / function or a setitem on the placeholder or a view of it)."""
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    index = {n: i for i, n in enumerate(placeholders)}

    def root(node):
        # Follow the first tensor argument back to a placeholder (views keep storage).
        while (
            isinstance(node, torch.fx.Node) and node.op != "placeholder" and node.args
        ):
            node = node.args[0]
        return node if isinstance(node, torch.fx.Node) else None

    mutated = set()
    for node in gm.graph.nodes:
        name = getattr(node.target, "__name__", str(node.target))
        in_place = node.op in ("call_method", "call_function") and (
            node.target is operator.setitem
            or (name.endswith("_") and not name.endswith("__"))
        )
        if in_place and node.args:
            base = root(node.args[0])
            if base in index:
                mutated.add(index[base])
    return mutated


def _replayed(gm):
    """gm as a callable that captures a trace per active Numerics (after EAGER_CALLS
    eager calls) and replays it after."""
    states = {}
    mutated = _mutated_inputs(gm)

    def run(*args):
        state = states.setdefault(current_numerics(), {"calls": 0})
        if "trace" not in state and state["calls"] < EAGER_CALLS:
            state["calls"] += 1
            return gm(*args)
        if "trace" not in state:
            # Persistent and in-place-updated inputs are used in place; the rest are
            # copied into stable buffers per call.
            static = [_is_static(a) or i in mutated for i, a in enumerate(args)]
            state["trace"] = Captured(gm, args, WARMUP, static=static)
            CAPTURED.append(state["trace"])
        try:
            return state["trace"](*args)
        except ReplayMismatch:
            return gm(*args)

    return run


def ktt_tt(gm, example_inputs):
    """Dynamo backend: the graph runs by tt trace replay."""
    return _replayed(gm)


def _views_as_reshapes(gm):
    """gm with aten view / _unsafe_view calls replaced by reshape (in place); returns
    gm. Same values in these functional graphs, for any tt output layout."""
    aten = torch.ops.aten
    views = {aten.view.default, aten._unsafe_view.default}
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in views:
            node.target = aten.reshape.default
    gm.recompile()
    return gm


def _compiler(gm, example_inputs):
    """AOTAutograd forward/backward compiler: boxed trace replay of the aten graph."""
    return make_boxed_func(_replayed(_views_as_reshapes(gm)))


ktt_tt_aot = aot_autograd(fw_compiler=_compiler, bw_compiler=_compiler)


def _fuse_compiler(gm, example_inputs):
    """AOTAutograd compiler: pointwise chains fused, the graph run eagerly."""
    fuse_pointwise(_views_as_reshapes(gm))
    return make_boxed_func(gm)


ktt_fuse = aot_autograd(fw_compiler=_fuse_compiler, bw_compiler=_fuse_compiler)


def register():
    """Registers "ktt_tt", "ktt_tt_aot" and "ktt_fuse" with torch._dynamo (idempotent;
    installed packages also expose them as torch_dynamo_backends entry points)."""
    for name, fn in (
        ("ktt_tt", ktt_tt),
        ("ktt_tt_aot", ktt_tt_aot),
        ("ktt_fuse", ktt_fuse),
    ):
        if name not in torch._dynamo.list_backends(exclude_tags=()):
            torch._dynamo.register_backend(name=name)(fn)


def release():
    """Releases every captured trace."""
    for trace in CAPTURED:
        trace.release()
    CAPTURED.clear()
