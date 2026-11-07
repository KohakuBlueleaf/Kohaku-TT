"""Unsupported ops: the error they raise, and counters of the opt-in CPU fallback
(KTT_CPU_FALLBACK=1, loud unless KTT_FALLBACK_WARN=0)."""

import torch

# Unsupported ops raise NotImplementedError whose message starts with UNSUPPORTED.
UNSUPPORTED = "KohakuTTUnsupportedOperator"
KohakuTTUnsupportedOperator = NotImplementedError


def fallback_counts():
    """{overload: calls} that ran on CPU through KTT_CPU_FALLBACK=1."""
    pairs = (item.rsplit("=", 1) for item in torch.ops.ktt.fallback_counts())
    return {name: int(count) for name, count in pairs}


def reset_fallback_counts():
    """Clears fallback_counts()."""
    torch.ops.ktt.reset_fallback_counts()
