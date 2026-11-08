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
    except RuntimeError:
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


def replicate_module(module):
    """Marks every tensor of `module` and its descendants replicated, except tensors
    with their own mark or under a descendant with its own default; returns it."""
    return _set(module, {}, default="R")


def shard_module(module, dim):
    """Marks every tensor of `module` and its descendants sharded on `dim`, with the
    same exceptions as replicate_module; returns it."""
    return _set(module, {}, default=f"S({dim})")


def resolve(module):
    """[(owner module, tensor name, mark)] for every marked tensor under `module`."""
    out = []

    def walk(m, inherited):
        default = m.__dict__.get("_ktt_default", inherited)
        own = m.__dict__.get("_ktt_marks", {})
        for name in list(m._parameters) + list(m._buffers):
            t = m._parameters.get(name, m._buffers.get(name))
            if t is None:
                continue
            mark = own.get(name, default)
            if mark is not None:
                out.append((m, name, mark))
        for child in m.children():
            walk(child, default)

    walk(module, None)
    return out


def apply_marks(module):
    """Places every marked tensor under `module` at its mark (a conversion only where
    the placement differs); no-op unless the tensors are on a mesh of 2+ chips."""
    if _mesh_size() < 2:
        return module
    done = {}
    for owner, name, mark in resolve(module):
        is_param = name in owner._parameters
        t = owner._parameters[name] if is_param else owner._buffers[name]
        if t.device.type != "tt":
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
            owner._buffers[name] = placed
    return module
