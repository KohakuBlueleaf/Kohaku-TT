"""Placement marks on modules: each mark names the mesh placement of a module's
parameters and buffers. Marks are stored on the module (by tensor name), take effect
when the module's tensors are on a mesh of more than one chip (at Module.to() or at
once if they already are), and are no-ops on one chip or off the device."""

import warnings

import torch
from torch import nn

_LINEAR_CONV = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)


def _mesh_size():
    try:
        return int(torch.ops.ktt.mesh_size())
    except (RuntimeError, AttributeError):
        return 0


def _spec(mark):
    """Per-mesh-axis placement string for mark "R" or "S(d)" (sharded on the largest
    mesh axis, other axes replicated)."""
    mesh = list(torch.ops.ktt.mesh_shape())
    spec = ["R"] * len(mesh)
    if mark != "R":
        spec[max(range(len(mesh)), key=lambda a: mesh[a])] = mark
    return ",".join(spec)


def _check(mark):
    if mark == "R":
        return mark
    if (
        mark.startswith("S(")
        and mark.endswith(")")
        and mark[2:-1].lstrip("-").isdigit()
    ):
        return mark
    raise ValueError(f'kohakutt: placement mark {mark!r} is not "R" or "S(dim)"')


def _set(module, marks, default=None):
    """Records `marks` ({tensor name: mark}) and an inherited `default` on `module`."""
    own = module.__dict__.setdefault("_ktt_marks", {})
    for name, mark in marks.items():
        if name not in module._parameters and name not in module._buffers:
            raise ValueError(
                f"kohakutt: {type(module).__name__} has no tensor {name!r}"
            )
        own[name] = _check(mark)
    if default is not None:
        module.__dict__["_ktt_default"] = _check(default)
    if _on_mesh(module):
        apply_marks(module)
    return module


def _on_mesh(module):
    if _mesh_size() < 2:
        return False
    for t in list(module.parameters()) + list(module.buffers()):
        return t.device.type == "tt"
    return False


def colwise(module):
    """Marks `module` column-parallel (output features split) and returns it: Linear and
    Conv weight and bias S(0), Embedding weight S(1)."""
    if isinstance(module, _LINEAR_CONV):
        marks = {"weight": "S(0)"}
        if module.bias is not None:
            marks["bias"] = "S(0)"
    elif isinstance(module, nn.Embedding):
        marks = {"weight": "S(1)"}
    else:
        raise TypeError(
            f"kohakutt.colwise: no column rule for {type(module).__name__};"
            " use kohakutt.place(module, name=...)"
        )
    return _set(module, marks)


def rowwise(module):
    """Marks `module` row-parallel (input features split) and returns it: Linear and
    Conv weight S(1) and bias replicated, Embedding weight S(0) (vocabulary split)."""
    if isinstance(module, _LINEAR_CONV):
        marks = {"weight": "S(1)"}
        if module.bias is not None:
            marks["bias"] = "R"
    elif isinstance(module, nn.Embedding):
        marks = {"weight": "S(0)"}
    else:
        raise TypeError(
            f"kohakutt.rowwise: no row rule for {type(module).__name__};"
            " use kohakutt.place(module, name=...)"
        )
    return _set(module, marks)


def place(module, **marks):
    """Marks named tensors of `module` ("R" or "S(dim)", e.g. weight="S(1)") and
    returns it."""
    return _set(module, marks)


def apply_plan(module, plan):
    """Marks the tensors under `module` by `plan` (the torch.compile mesh_plan format:
    {fnmatch pattern over module or tensor paths: "colwise" | "rowwise" | "replicate" |
    "R" | "S(d)" | "shard(d)" | {tensor name: placement}}, first match wins) and returns
    `module`. Same as writing the marks in code."""
    from kohakutt.torch_backend.mesh_compile import plan_marks

    for path, m in module.named_modules():
        if isinstance(m, nn.Embedding):
            kind = "embedding"
        elif isinstance(m, nn.Linear):
            kind = "linear"
        elif isinstance(m, _LINEAR_CONV):
            kind = "conv"
        else:
            kind = None
        marks = {}
        for name in list(m._parameters) + list(m._buffers):
            if m._parameters.get(name, m._buffers.get(name)) is None:
                continue
            mark = plan_marks(plan, f"{path}.{name}" if path else name, kind)
            if mark is not None:
                marks[name] = mark
        if marks:
            _set(m, marks)
    return module


def conflict(message):
    """Reports a placement conflict: a warning, or a RuntimeError under
    KTT_MESH_STRICT=1."""
    import os

    if os.environ.get("KTT_MESH_STRICT", "0") == "1":
        raise RuntimeError(message + " (KTT_MESH_STRICT=1)")
    warnings.warn(message, stacklevel=3)


def mark_of(tensor):
    """(mark, source) last applied to `tensor` ("mark": set in code, "auto": default
    pairing, "plan": a compile mesh_plan), or None."""
    return getattr(tensor, "_ktt_mark", None)


def replicate_module(module):
    """Marks every tensor of `module` and its descendants replicated, except tensors
    with their own mark or under a descendant with its own default; returns it."""
    return _set(module, {}, default="R")


def shard_module(module, dim):
    """Marks every tensor of `module` and its descendants sharded on `dim`, with the
    same exceptions as replicate_module; returns it."""
    return _set(module, {}, default=f"S({dim})")


_COLWISE_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "qkv_proj",
    "qkv",
    "query",
    "key",
    "value",
    "up_proj",
    "gate_proj",
    "gate_up_proj",
    "fc1",
    "wi",
    "wi_0",
    "wi_1",
    "w1",
    "w3",
    "c_fc",
    "in_proj",
    "lin1",
}
_ROWWISE_NAMES = {
    "o_proj",
    "out_proj",
    "down_proj",
    "fc2",
    "wo",
    "w2",
    "c_proj",
    "dense_4h_to_h",
    "lin2",
}


def _pair_mode():
    """ "0" (no default pairing), "1" (pairs split by rows per call, the default) or
    "always" (KTT_MESH_AUTO_PAIRS; KTT_MESH_AUTO=0 is "0")."""
    import os

    if os.environ.get("KTT_MESH_AUTO", "1") == "0":
        return "0"
    return os.environ.get("KTT_MESH_AUTO_PAIRS", "1")


def _auto_pairs():
    return _pair_mode() != "0"


def _shard_min():
    import os

    return int(os.environ.get("KTT_MESH_SHARD_MIN", str(1 << 20)))


# One chip's bf16 matmul rate (measured MLP stack, P150A) and the eager all-reduce of a
# pair's output (µs: latency + bytes / bytes per µs, as core/torch/mesh_rules.cpp).
_CHIP_FLOPS = 160e12
_ALL_REDUCE_US = (120.0, 77e3)


def pair_pays(rows, flops_per_row, width, elem=2, chips=2):
    """True when splitting a column -> row pair over `chips` chips saves more matmul
    time at `rows` rows per call (`flops_per_row`: the pair's matmul flops per row) than
    the eager all-reduce of its rows x `width` output (`elem` bytes per element)
    costs."""
    saved_us = rows * flops_per_row * (1 - 1 / chips) / _CHIP_FLOPS * 1e6
    latency, rate = _ALL_REDUCE_US
    return saved_us > latency + rows * width * elem / rate


def _residual_width(module):
    """The most common in_features over the Linears under `module` (None if none)."""
    counts = {}
    for m in module.modules():
        if isinstance(m, nn.Linear):
            counts[m.in_features] = counts.get(m.in_features, 0) + 1
    return max(counts, key=counts.get) if counts else None


def _auto_style(name, linear, width):
    """ "colwise", "rowwise" or None for an unmarked Linear named `name`: by its name,
    else by its shape against the residual width (widening -> colwise, narrowing back
    to the width -> rowwise)."""
    leaf = name.rsplit(".", 1)[-1].lower()
    if leaf in _COLWISE_NAMES:
        return "colwise"
    if leaf in _ROWWISE_NAMES:
        return "rowwise"
    if width is not None:
        if linear.in_features == width and linear.out_features > width:
            return "colwise"
        if linear.out_features == width and linear.in_features > width:
            return "rowwise"
    return None


_STYLE_MARKS = {
    "colwise": {"weight": "S(0)", "bias": "S(0)"},
    "rowwise": {"weight": "S(1)", "bias": "R"},
}


class _Pair:
    """The default column -> row pairing of the unmarked Linears of one parent module:
    `members` [(linear, style)], their matmul flops per input row, the width of the
    row-split output, and the decision `split` (None until decided, then True: split,
    False: replicated)."""

    def __init__(self, members, width):
        self.members = members
        self.flops_per_row = sum(
            2.0 * m.in_features * m.out_features for m, _ in members
        )
        rows_out = [m.out_features for m, s in members if s == "rowwise"]
        self.width = max(rows_out) if rows_out else (width or 0)
        self.split = None
        self.hooks = []

    def marks(self, linear, style):
        """{tensor name: mark} of member `linear` under the decision."""
        names = ["weight"] + (["bias"] if linear.bias is not None else [])
        if not self.split:
            return dict.fromkeys(names, "R")
        return {n: _STYLE_MARKS[style][n] for n in names}


def _auto_groups(module):
    """[_Pair] of the default pairing under `module`: unmarked Linears whose weight is
    at least KTT_MESH_SHARD_MIN bytes and that a name or shape rule styles, one pair per
    parent module; [] under KTT_MESH_AUTO_PAIRS=0. A Linear already in a pair keeps
    it."""
    if not _auto_pairs():
        return []
    width = _residual_width(module)
    floor = _shard_min()
    by_parent, pairs = {}, {}
    for name, m in module.named_modules():
        if not isinstance(m, nn.Linear) or "_ktt_marks" in m.__dict__:
            continue
        if m.weight.numel() * m.weight.element_size() < floor:
            continue
        pair = m.__dict__.get("_ktt_pair")
        if pair is not None:
            pairs[id(pair)] = pair
            continue
        style = _auto_style(name, m, width)
        if style is not None:
            by_parent.setdefault(name.rpartition(".")[0], []).append((m, style))
    for members in by_parent.values():
        pair = _Pair(members, width)
        for m, _ in members:
            m.__dict__["_ktt_pair"] = pair
        pairs[id(pair)] = pair
    return list(pairs.values())


def _auto_marks(module):
    """{(id(owner), tensor name): mark} of the decided default pairs under `module`."""
    out = {}
    for pair in _auto_groups(module):
        if pair.split is None:
            continue
        for m, style in pair.members:
            for name, mark in pair.marks(m, style).items():
                out[(id(m), name)] = mark
    return out


def _place_pair(pair):
    """Places the tensors of decided `pair` at their marks, except tensors that already
    carry a placement decision (a mark, a plan, the graph pairing)."""
    for m, style in pair.members:
        for name, mark in pair.marks(m, style).items():
            t = m._parameters[name]
            if t is None or t.device.type != "tt" or mark_of(t) is not None:
                continue
            t.data = torch.ops.ktt.distribute(t.detach(), _spec(mark))
            t._ktt_mark = (mark, "auto")


def _arm(pair):
    """Forward pre-hooks on `pair`'s members: the first eager call of a member decides
    the pair by its input's rows (pair_pays), places it and removes the hooks. No-op
    while torch.compile traces (the graph pairing places compiled weights)."""
    if pair.hooks:
        return

    def decide(linear, args):
        if torch.compiler.is_compiling() or pair.split is not None:
            return
        x = args[0] if args and isinstance(args[0], torch.Tensor) else None
        if x is None or x.device.type != "tt" or x.dim() == 0:
            return
        rows = x.numel() // max(1, x.shape[-1])
        chips = max(torch.ops.ktt.mesh_shape())
        pair.split = pair_pays(
            rows, pair.flops_per_row, pair.width, x.element_size(), chips
        )
        for hook in pair.hooks:
            hook.remove()
        pair.hooks = []
        _place_pair(pair)

    pair.hooks = [m.register_forward_pre_hook(decide) for m, _ in pair.members]


def resolve(module, with_source=False):
    """[(owner module, tensor name, mark)] for every marked tensor under `module`:
    own marks, then inherited defaults, then the default pairing of Linears. With
    `with_source`, each item also carries "mark" (set in code) or "auto" (default
    pairing)."""
    out = []
    auto = _auto_marks(module)

    def walk(m, inherited):
        default = m.__dict__.get("_ktt_default", inherited)
        own = m.__dict__.get("_ktt_marks", {})
        for name in list(m._parameters) + list(m._buffers):
            t = m._parameters.get(name, m._buffers.get(name))
            if t is None:
                continue
            mark, source = own.get(name, default), "mark"
            if mark is None:
                mark, source = auto.get((id(m), name)), "auto"
            if mark is not None:
                out.append((m, name, mark, source) if with_source else (m, name, mark))
        for child in m.children():
            walk(child, default)

    walk(module, None)
    return out


def apply_marks(module):
    """Places every marked tensor under `module` at its mark (a conversion only where
    the placement differs) and arms the default pairs not decided yet (decided at their
    first call; split at once under KTT_MESH_AUTO_PAIRS=always); no-op unless the
    tensors are on a mesh of 2+ chips."""
    if _mesh_size() < 2:
        return module
    for pair in _auto_groups(module):
        if pair.split is None and _pair_mode() == "always":
            pair.split = True
        elif pair.split is None:
            _arm(pair)
    done = {}
    for owner, name, mark, source in resolve(module, with_source=True):
        is_param = name in owner._parameters
        t = owner._parameters[name] if is_param else owner._buffers[name]
        if t.device.type != "tt" or t.is_complex():
            continue
        key = id(t)
        if key in done:
            if done[key] != mark:
                warnings.warn(
                    f"kohakutt: tensor {name!r} of {type(owner).__name__} is shared and"
                    f" marked both {done[key]} and {mark}; keeping {done[key]}",
                    stacklevel=2,
                )
            continue
        done[key] = mark
        placed = torch.ops.ktt.distribute(t.detach(), _spec(mark))
        if is_param:
            t.data = placed
        else:
            owner._buffers[name] = t = placed
        t._ktt_mark = (mark, source)
    return module
