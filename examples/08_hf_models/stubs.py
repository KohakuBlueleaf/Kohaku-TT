"""Minimal stand-ins for the `timm` layers and `torchvision.transforms` that the PixAI
tagger's remote code imports; installed only when the real packages are absent.
"""

import importlib.machinery
import sys
import types

from torch import nn


def _module(name):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    return module


class DropPath(nn.Identity):
    def __init__(self, *args, **kwargs):
        super().__init__()


class Mlp(nn.Module):
    """timm.layers.Mlp: fc1, activation, optional norm, fc2."""

    def __init__(self, in_features, hidden_features=None, out_features=None, **kw):
        super().__init__()
        hidden = hidden_features or in_features
        bias, norm_layer = kw.get("bias", True), kw.get("norm_layer")
        self.fc1 = nn.Linear(in_features, hidden, bias=bias)
        self.act = kw.get("act_layer", nn.GELU)()
        self.norm = norm_layer(hidden) if norm_layer else nn.Identity()
        self.fc2 = nn.Linear(hidden, out_features or in_features, bias=bias)

    def forward(self, x):
        return self.fc2(self.norm(self.act(self.fc1(x))))


def install_stubs():
    """Installs the stand-ins; call before importing the tagger's remote code."""
    # transformers must import with the real torchvision (or none) first
    import transformers.image_processing_base
    import transformers.image_utils
    import transformers.pipelines  # noqa: F401

    try:
        import torchvision.transforms  # noqa: F401
    except ImportError:
        functional = _module("torchvision.transforms.functional")
        transforms = _module("torchvision.transforms")
        transforms.functional = functional
        vision = _module("torchvision")
        vision.transforms = transforms
        sys.modules["torchvision"] = vision
        sys.modules["torchvision.transforms"] = transforms
        sys.modules["torchvision.transforms.functional"] = functional
    try:
        import timm.layers
    except ImportError:
        layers = _module("timm.layers")
        layers.DropPath, layers.Mlp = DropPath, Mlp
        layers.trunc_normal_ = nn.init.trunc_normal_
        timm, models = _module("timm"), _module("timm.models")
        timm.layers = models.layers = layers
        sys.modules.update({"timm": timm, "timm.layers": layers, "timm.models": models})
        sys.modules["timm.models.layers"] = layers
