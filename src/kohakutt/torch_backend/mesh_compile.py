"""Mesh placement under torch.compile: the "mesh_plan" compile option (placements of a
compiled module's parameters and buffers by module path), the column -> row pairing of
unmarked Linears from the graph, and the check of activation marks against module
marks."""

import fnmatch
import operator
import re
import warnings

import torch

_ATEN_LINEAR = {torch._C._nn.linear, torch.nn.functional.linear}
_CONV = {
    torch.conv1d,
    torch.conv2d,
    torch.conv3d,
    torch.nn.functional.conv1d,
    torch.nn.functional.conv2d,
    torch.nn.functional.conv3d,
}
_EMBEDDING = {torch.nn.functional.embedding}

# Tokens of a Dynamo source name, e.g.
# L['self']._modules['layers']._modules['0']._parameters['weight'].
_TOKEN = re.compile(r"\._(?:modules|parameters|buffers)\['([^']*)'\]|\.(\w+)|\[(\d+)\]")


def _path(source_name):
    """ "layers.0.weight" from a Dynamo source name of a module's tensor, or None when
    the source is not a module parameter or buffer."""
    if "_parameters[" not in source_name and "_buffers[" not in source_name:
        return None
    rest = source_name.split("]", 1)[1] if source_name.startswith("L[") else source_name
    parts = [a or b or c for a, b, c in _TOKEN.findall(rest)]
    return ".".join(parts) or None


def _kind(node):
    """ "linear", "conv", "embedding" or None: how the graph uses placeholder `node`."""
    for user in node.users:
        if user.op != "call_function":
            continue
        if user.target in _ATEN_LINEAR:
            return "linear"
        if user.target in _CONV:
            return "conv"
        if user.target in _EMBEDDING:
            return "embedding"
    return None


_STYLES = {
    # style: {(kind, tensor leaf): mark}
    "colwise": {
        ("linear", "weight"): "S(0)",
        ("linear", "bias"): "S(0)",
        ("conv", "weight"): "S(0)",
        ("conv", "bias"): "S(0)",
        ("embedding", "weight"): "S(1)",
    },
    "rowwise": {
        ("linear", "weight"): "S(1)",
        ("linear", "bias"): "R",
        ("conv", "weight"): "S(1)",
        ("conv", "bias"): "R",
        ("embedding", "weight"): "S(0)",
    },
}


def _value_mark(value, kind, leaf, path):
    """The mark ("R" / "S(d)") a plan value gives tensor `leaf` used as `kind`."""
    from kohakutt.device.marks import _check

    if isinstance(value, dict):
        value = value.get(leaf)
        if value is None:
            return None
    if value in ("replicate", "R"):
        return "R"
    if value in _STYLES:
        mark = _STYLES[value].get((kind, leaf))
        if mark is None:
            raise ValueError(
                f"kohakutt mesh_plan: {value!r} has no rule for {path!r}"
                f" (used as {kind or 'unknown'}); give its placement, e.g."
                f' {{"{leaf}": "S(0)"}}'
            )
        return mark
    m = re.fullmatch(r"shard\((-?\d+)\)", str(value))
    if m:
        return f"S({m.group(1)})"
    return _check(str(value))


def plan_marks(plan, path, kind):
    """The mark `plan` ({pattern: value}) gives the tensor at `path`, or None. The first
    entry whose pattern matches the tensor path or its module path wins; patterns use
    fnmatch ("layers.*.mlp.fc1"). Values: "colwise", "rowwise", "replicate" / "R",
    "S(d)" / "shard(d)", or {tensor name: placement}."""
    module, _, leaf = path.rpartition(".")
    for pattern, value in plan.items():
        if fnmatch.fnmatchcase(path, pattern) or (
            module and fnmatch.fnmatchcase(module, pattern)
        ):
            return _value_mark(value, kind, leaf, path)
    return None


def _module_tensors(gm, example_inputs):
    """[(placeholder, tt tensor, module path)] for the graph inputs of `gm` that are
    parameters or buffers of the compiled module."""
    out = []
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    for node, value in zip(placeholders, example_inputs):
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "tt"
            or value.is_complex()
        ):
            continue
        source = getattr(node.meta.get("grapharg"), "source", None)
        path = _path(source.name) if source is not None else None
        if path is not None:
            out.append((node, value, path))
    return out


def _place(value, mark, source):
    from kohakutt.device.marks import _spec

    value.data = torch.ops.ktt.distribute(value.detach(), _spec(mark))
    value._ktt_mark = (mark, source)


def apply_plan(gm, example_inputs, plan):
    """Places the module tensors of Dynamo graph `gm` by `plan` before the graph is
    compiled; a plan entry overrides a mark set in code and the default pairing.
    Placement is a property of the tensor, so it stays after the compile. No-op off a
    mesh of 2+ chips."""
    from kohakutt.device.marks import _mesh_size

    if not plan or _mesh_size() < 2:
        return
    for node, value, path in _module_tensors(gm, example_inputs):
        mark = plan_marks(plan, path, _kind(node))
        if mark is None:
            continue
        own = getattr(value, "_ktt_mark", None)
        if own is not None and own[0] == mark:
            value._ktt_mark = (mark, "plan")
            continue
        _place(value, mark, "plan")


def _is_distribute(node):
    return node.op == "call_function" and node.target in (
        torch.ops.ktt.distribute,
        torch.ops.ktt.distribute.default,
    )


def check_activation_marks(gm, example_inputs):
    """Reports (marks.conflict) each kohakutt.shard / replicate in graph `gm` that asks
    a module tensor marked in code or by the plan for another placement."""
    from kohakutt.device.marks import _spec, conflict

    tensors = {n: (v, p) for n, v, p in _module_tensors(gm, example_inputs)}
    for node in gm.graph.nodes:
        if not _is_distribute(node) or node.args[0] not in tensors:
            continue
        value, path = tensors[node.args[0]]
        own = getattr(value, "_ktt_mark", None)
        if own is None or own[1] not in ("mark", "plan"):
            continue
        spec = node.args[1] if len(node.args) > 1 else node.kwargs.get("placement")
        if spec != _spec(own[0]):
            conflict(
                f"kohakutt: {path} is marked {own[0]} ({own[1]}) and placed {spec} in"
                f" forward; the module's tensor keeps {own[0]}, the copy is {spec}"
            )


# Graph pairing: ops a column-split activation passes through unchanged in placement
# kind (elementwise, head reshapes, attention) on its way to the row-split Linear.
_PASS_FUNCTIONS = {
    operator.add,
    operator.mul,
    operator.sub,
    operator.truediv,
    operator.neg,
    operator.getitem,
    torch.add,
    torch.mul,
    torch.sub,
    torch.div,
    torch.neg,
    torch.relu,
    torch.sigmoid,
    torch.tanh,
    torch.exp,
    torch.cat,
    torch.chunk,
    torch.split,
    torch.transpose,
    torch.permute,
    torch.reshape,
    torch.matmul,
    torch.softmax,
    torch.nn.functional.gelu,
    torch.nn.functional.silu,
    torch.nn.functional.relu,
    torch.nn.functional.softmax,
    torch.nn.functional.dropout,
    torch.nn.functional.scaled_dot_product_attention,
}
_PASS_METHODS = {
    "view",
    "reshape",
    "transpose",
    "permute",
    "contiguous",
    "chunk",
    "split",
    "unflatten",
    "flatten",
    "float",
    "to",
    "type_as",
    "mul",
    "add",
    "sub",
    "div",
    "sigmoid",
    "relu",
    "softmax",
    "__getitem__",
}


def _passes(node):
    if node.op == "call_function":
        return node.target in _PASS_FUNCTIONS
    if node.op == "call_method":
        return node.target in _PASS_METHODS
    return False


def _weight_of(node, params):
    """The parameter placeholder that is `node`'s weight when `node` is a linear call
    on a parameter, else None."""
    if node.op != "call_function" or node.target not in _ATEN_LINEAR:
        return None
    weight = node.args[1] if len(node.args) > 1 else node.kwargs.get("weight")
    return weight if weight in params else None


def graph_pairs(gm, params):
    """{weight placeholder: "colwise" | "rowwise"} for Dynamo graph `gm`, over candidate
    weights `params` ({placeholder: parameter}). A group is Linears whose outputs reach
    only other Linears through _PASS_* ops: the sources column-split, the reached
    row-split. Groups are taken in graph order; one that touches a placed Linear in
    another role, or reaches any other op, is skipped."""
    roles = {}
    linears = {n: _weight_of(n, params) for n in gm.graph.nodes}
    linears = {n: w for n, w in linears.items() if w is not None}
    for start in linears:
        seen, stack, ends, sources, ok = set(), [start], set(), {start}, True
        while stack and ok:
            node = stack.pop()
            for user in node.users:
                if user in seen:
                    continue
                seen.add(user)
                if user in linears:
                    ends.add(user)
                elif _passes(user):
                    stack.append(user)
                    # Another Linear feeding the same elementwise op (gate / up) is a
                    # column-split source of the group too.
                    for arg in user.all_input_nodes:
                        if arg in linears and arg not in sources:
                            sources.add(arg)
                            stack.append(arg)
                else:
                    ok = False
                    break
        if not ok or not ends or ends & sources:
            continue
        group = [(linears[s], "colwise") for s in sources]
        group += [(linears[e], "rowwise") for e in ends]
        if any(roles.get(w, r) != r for w, r in group):
            continue
        roles.update(group)
    return roles


def apply_graph_pairs(gm, example_inputs):
    """Places module Linears of `gm` by graph_pairs (column -> row) where neither the
    code nor a mesh_plan marks them, replacing the name pairing (KTT_MESH_AUTO_PAIRS=0:
    off). Weights under KTT_MESH_SHARD_MIN bytes are left as they are."""
    from kohakutt.device.marks import _auto_pairs, _mesh_size, _shard_min

    if not _auto_pairs() or _mesh_size() < 2:
        return
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    params = {}
    for node, value, _ in _module_tensors(gm, example_inputs):
        own = getattr(value, "_ktt_mark", None)
        if own is not None and own[1] in ("mark", "plan"):
            continue
        if value.numel() * value.element_size() >= _shard_min():
            params[node] = value
    for weight, role in graph_pairs(gm, params).items():
        tensors = [(params[weight], _STYLES[role][("linear", "weight")])]
        for user in weight.users:
            if user.op != "call_function" or user.target not in _ATEN_LINEAR:
                continue
            bias = user.args[2] if len(user.args) > 2 else user.kwargs.get("bias")
            if isinstance(bias, torch.fx.Node) and bias.op == "placeholder":
                value = example_inputs[placeholders.index(bias)]
                own = getattr(value, "_ktt_mark", None)
                if own is not None and own[1] in ("mark", "plan"):
                    continue
                if isinstance(value, torch.Tensor) and value.device.type == "tt":
                    tensors.append((value, _STYLES[role][("linear", "bias")]))
        for value, mark in tensors:
            own = getattr(value, "_ktt_mark", None)
            if own is not None and own[0] == mark:
                continue
            _place(value, mark, "auto")


_OPTIONS = {"mesh_plan"}


def with_options(backend, name):
    """`backend` taking torch.compile's `options` (and ignoring `mode`): applies
    options["mesh_plan"], then the graph pairing, then checks activation marks; warns
    on unknown options."""

    def compile_graph(gm, example_inputs, options=None, mode=None, **_):
        options = options or {}
        unknown = set(options) - _OPTIONS
        if unknown:
            warnings.warn(
                f"kohakutt: unknown torch.compile options {sorted(unknown)}"
                f" (known: {sorted(_OPTIONS)})",
                stacklevel=2,
            )
        apply_plan(gm, example_inputs, options.get("mesh_plan"))
        apply_graph_pairs(gm, example_inputs)
        check_activation_marks(gm, example_inputs)
        return backend(gm, example_inputs)

    compile_graph.__name__ = name
    return compile_graph
