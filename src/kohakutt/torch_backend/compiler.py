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
from torch.utils._pytree import tree_map

from kohakutt.core.numerics import current_numerics
from kohakutt.fusion.fx import fuse_pointwise
from kohakutt.torch_backend.chains import fuse_chains
from kohakutt.torch_backend.mesh_compile import with_options
from kohakutt.torch_backend.trace import Captured, ReplayMismatch

# Warm-up calls before capture: compiled graphs are functional, one call compiles all.
WARMUP = 1

# Calls each graph runs eagerly before its trace is captured, so state allocated in the
# first steps (e.g. optimizer moments) exists before capture (docs/concepts/trace.md).
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


def _schema_writes(node):
    """The arguments of call `node` its op's schema writes in place (`Tensor(a!)`)."""
    target = node.target
    schema = getattr(target, "_schema", None)
    if schema is None and hasattr(target, "overloads"):  # an OpOverloadPacket
        schema = getattr(getattr(target, "default", None), "_schema", None)
    if schema is None:
        return []
    out = []
    for i, arg in enumerate(schema.arguments):
        if arg.alias_info is not None and arg.alias_info.is_write:
            if i < len(node.args):
                out.append(node.args[i])
            elif arg.name in node.kwargs:
                out.append(node.kwargs[arg.name])
    return out


def _mutated_inputs(gm):
    """Indices of gm's placeholders that the graph writes in place (an in-place
    method / function, a setitem, or an op whose schema writes an argument, on the
    placeholder or a view of it)."""
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
        written = [node.args[0]] if in_place and node.args else []
        if node.op == "call_function":
            written += _schema_writes(node)
        for arg in written:
            base = root(arg)
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
            # The capture's prepare call is this call, run on the caller's
            # tensors (each in-place update once); its outputs answer it, replays follow
            trace = Captured(gm, args, WARMUP, static=static, keep_prepared=True)
            state["trace"] = trace
            CAPTURED.append(trace)
            out, trace.prepared = trace.prepared, None
            return _off_stable_inputs(out, trace, args)
        try:
            return state["trace"](*args)
        except ReplayMismatch:
            return gm(*args)

    return run


def _off_stable_inputs(out, trace, args):
    """out with each tensor that is a stable input buffer of trace replaced by the
    caller's tensor, and each one sharing such a buffer's memory cloned."""
    stable = {
        _storage(b): (b, a)
        for b, a, keep in zip(trace.inputs, args, trace.static, strict=True)
        if isinstance(b, torch.Tensor) and not keep
    }

    def own(t):
        if not isinstance(t, torch.Tensor) or _storage(t) not in stable:
            return t
        buffer, arg = stable[_storage(t)]
        return arg if t is buffer else t.clone()

    return tree_map(own, out)


def _storage(t):
    return t.untyped_storage().data_ptr()


def _tt(gm, example_inputs):
    """Dynamo backend: the graph, its op chains fused, runs by tt trace replay."""
    return _replayed(fuse_chains(gm))


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
    """AOTAutograd forward/backward compiler: boxed trace replay of the aten graph, its
    op chains fused."""
    return make_boxed_func(_replayed(_views_as_reshapes(fuse_chains(gm))))


_tt_aot = aot_autograd(fw_compiler=_compiler, bw_compiler=_compiler)


def _fuse_compiler(gm, example_inputs):
    """AOTAutograd compiler: pointwise chains fused, the graph run eagerly."""
    fuse_pointwise(_views_as_reshapes(gm))
    return make_boxed_func(gm)


_fuse = aot_autograd(fw_compiler=_fuse_compiler, bw_compiler=_fuse_compiler)

# The registered backends take torch.compile's options (mesh_plan).
ktt_tt = with_options(_tt, "ktt_tt")
ktt_tt_aot = with_options(_tt_aot, "ktt_tt_aot")
ktt_fuse = with_options(_fuse, "ktt_fuse")


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
