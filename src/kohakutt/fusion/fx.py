"""Groups chains of pointwise aten ops in an FX graph into torch.ops.ktt_fusion.chain
calls. A group is a tree of pointwise nodes whose inner values have no other users, with
at most 4 tensor inputs (each full, matrix, row, column or scalar against the group's
output shape) and at most 4 live values; inner dtype casts are absorbed (values stay
fp32 in DST), the output takes the last node's dtype."""

import json
import math

import torch
import torch.fx

from kohakutt.fusion import SLOTS

aten = torch.ops.aten
prims = torch.ops.prims

BINARY = {
    aten.add.Tensor: "add",
    aten.sub.Tensor: "sub",
    aten.mul.Tensor: "mul",
    aten.div.Tensor: "div",
}
SCALAR = {
    aten.add.Scalar: "adds",
    aten.sub.Scalar: "subs",
    aten.mul.Scalar: "muls",
    aten.div.Scalar: "divs",
}
UNARY = {
    aten.exp.default: "exp",
    aten.tanh.default: "tanh",
    aten.sigmoid.default: "sigmoid",
    aten.rsqrt.default: "rsqrt",
    aten.sqrt.default: "sqrt",
    aten.reciprocal.default: "recip",
    aten.neg.default: "neg",
    aten.abs.default: "abs",
    aten.silu.default: "silu",
}
CASTS = {aten._to_copy.default, prims.convert_element_type.default}
FLOATS = (torch.bfloat16, torch.float32)


def _val(node):
    return node.meta.get("val") if isinstance(node, torch.fx.Node) else None


def _pointwise(node):
    """True for a node this pass can fuse (by target, dtypes and arguments)."""
    if node.op != "call_function":
        return False
    val = _val(node)
    if not isinstance(val, torch.Tensor) or val.dtype not in FLOATS or val.dim() < 1:
        return False
    target = node.target
    if target in CASTS:
        kwargs = {k: v for k, v in node.kwargs.items() if k not in ("dtype",)}
        return not kwargs and _val(node.args[0]) is not None
    if target is aten.pow.Tensor_Scalar:
        return node.args[1] == 2
    if target in UNARY or target in SCALAR:
        return True
    if target in BINARY:
        alpha = node.kwargs.get("alpha", 1)
        return alpha == 1 and len(node.args) == 2
    return False


def _layout(shape, out):
    """The reader layout of an input of `shape` against output `out`, or None."""
    shape = [1] * (len(out) - len(shape)) + list(shape)
    if len(shape) != len(out):
        return None
    if len(out) < 2:
        shape, out = [1, *shape], [1, *out]
    lead_in = math.prod(shape[:-2])
    same_lead = list(shape[:-2]) == list(out[:-2])
    if shape == list(out):
        return "full"
    if shape[-2:] == list(out[-2:]) and lead_in == 1:
        return "matrix"
    if not (same_lead or lead_in == 1):
        return None
    if shape[-2] == 1 and shape[-1] == out[-1]:
        return "row" if same_lead or lead_in == 1 else None
    if shape[-1] == 1 and shape[-2] == out[-2]:
        return "column"
    if shape[-1] == 1 and shape[-2] == 1 and lead_in == 1:
        return "scalar"
    return None


class _Group:
    """A fusable tree: its nodes in topological order and its output node."""

    def __init__(self, node):
        self.nodes = [node]
        self.out = node

    def plan(self):
        """(input nodes, layouts, ops, out slot) or None when the group does not fit."""
        members = set(self.nodes)
        out_shape = list(_val(self.out).shape)
        inputs, layouts = [], []
        for node in self.nodes:
            for arg in node.args:
                if isinstance(arg, torch.fx.Node) and arg not in members:
                    if arg in inputs:
                        continue
                    val = _val(arg)
                    if not isinstance(val, torch.Tensor) or val.dtype not in FLOATS:
                        return None
                    layout = _layout(list(val.shape), out_shape)
                    if layout is None:
                        return None
                    inputs.append(arg)
                    layouts.append(layout)
        if not 1 <= len(inputs) <= SLOTS:
            return None
        slot = {node: i for i, node in enumerate(inputs)}
        last_use = {}
        for i, node in enumerate(self.nodes):
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    last_use[arg] = i
        free = [s for s in range(SLOTS) if s >= len(inputs)]
        ops = []
        for i, node in enumerate(self.nodes):
            args = [a for a in node.args if isinstance(a, torch.fx.Node)]
            # Slots of operands whose value dies here are reusable as the destination.
            dying = list(
                dict.fromkeys(
                    slot[a] for a in args if last_use.get(a) == i and a in slot
                )
            )
            dst = dying[0] if dying else (free.pop(0) if free else None)
            if dst is None:
                return None
            target = node.target
            a = slot[node.args[0]] if isinstance(node.args[0], torch.fx.Node) else None
            if target in CASTS:
                ops.append(["copy", dst, a, None])
            elif target is aten.pow.Tensor_Scalar:
                ops.append(["square", dst, a, None])
            elif target in UNARY:
                ops.append([UNARY[target], dst, a, None])
            elif target in SCALAR:
                ops.append([SCALAR[target], dst, a, float(node.args[1])])
            else:
                x, y = node.args
                name = BINARY[target]
                if isinstance(x, torch.fx.Node) and isinstance(y, torch.fx.Node):
                    ops.append([name, dst, slot[x], slot[y]])
                elif isinstance(x, torch.fx.Node):
                    ops.append([name + "s", dst, slot[x], float(y)])
                else:
                    reverse = {"add": "adds", "mul": "muls", "sub": "rsubs"}
                    if name not in reverse:
                        return None
                    ops.append([reverse[name], dst, slot[y], float(x)])
            free.extend(dying[1:])
            slot[node] = dst
        return inputs, layouts, ops, slot[self.out]


def fuse_pointwise(gm):
    """Replaces fusable pointwise chains of `gm` (in place) by ktt_fusion.chain calls;
    returns the number of chains."""
    graph = gm.graph
    order = {node: i for i, node in enumerate(graph.nodes)}
    group_of = {}
    for node in graph.nodes:
        if not _pointwise(node):
            continue
        group = _Group(node)
        # Absorb pointwise producers used only here (their whole groups).
        for arg in node.args:
            if (
                isinstance(arg, torch.fx.Node)
                and arg in group_of
                and len(arg.users) == 1
                and group_of[arg].out is arg
            ):
                merged = group_of[arg].nodes + group.nodes
                trial = _Group(node)
                trial.nodes = sorted(set(merged), key=order.__getitem__)
                if trial.plan() is not None:
                    group = trial
        for member in group.nodes:
            group_of[member] = group
    chains = 0
    done = set()
    for group in {id(g): g for g in group_of.values()}.values():
        if id(group) in done or len(group.nodes) < 2:
            continue
        done.add(id(group))
        plan = group.plan()
        if plan is None:
            continue
        inputs, layouts, ops, out_slot = plan
        val = _val(group.out)
        code = {torch.bfloat16: 15, torch.float32: 6}[val.dtype]
        spec = json.dumps({"inputs": layouts, "ops": ops, "out": out_slot})
        with graph.inserting_before(group.out):
            fused = graph.call_function(
                torch.ops.ktt_fusion.chain.default,
                (list(inputs), spec, list(val.shape), code),
            )
        fused.meta["val"] = val
        group.out.replace_all_uses_with(fused)
        for node in reversed(group.nodes):
            graph.erase_node(node)
        chains += 1
    graph.lint()
    gm.recompile()
    return chains
