"""Numerics: the math settings tt ops are queued with, as a per-thread scope.

    with kohakutt.numerics(fidelity="HiFi3"):
        y = x @ w  # queued with HiFi3

fidelity: FPU multiply passes (LoFi < HiFi2 < HiFi3 < HiFi4); approx: SFPU
approximations (exp, reciprocal, ...); strict: kernel requirements ignored; attention:
"exact", "centered" (K centered over the sequence) or "centered_bfp8" (and Q / K in
bfp8); attention_kernels: the attention kernels allowed (sdpa_kernel). An op reads the
scope when it is queued. Defaults: HiFi2, exact, every kernel (KTT_FIDELITY,
KTT_APPROX).
"""

import contextlib
import functools
import os
import threading
from dataclasses import dataclass, replace

import torch

FIDELITIES = ("LoFi", "HiFi2", "HiFi3", "HiFi4")
ATTENTION = ("exact", "centered", "centered_bfp8")
# Attention kernels: Kohaku-TT's flash kernel, its earlier attention program, TT-NN's
# (and tt-train's) kernels, PyTorch's math composite on tt ops.
KERNELS = ("flash", "own", "ttnn", "math")


@dataclass(frozen=True)
class Numerics:
    """One scope's settings."""

    fidelity: str = "HiFi2"
    approx: bool = False
    strict: bool = False
    attention: str = "exact"
    attention_kernels: tuple = KERNELS

    def __post_init__(self):
        """Rejects unknown fidelities, attention modes and kernels."""
        if self.fidelity not in FIDELITIES:
            raise ValueError(f"fidelity {self.fidelity!r} not in {FIDELITIES}")
        if self.attention not in ATTENTION:
            raise ValueError(f"attention {self.attention!r} not in {ATTENTION}")
        if not self.attention_kernels or any(
            k not in KERNELS for k in self.attention_kernels
        ):
            raise ValueError(
                f"attention kernels {self.attention_kernels!r}: "
                f"one or more of {KERNELS}"
            )
        ordered = tuple(k for k in KERNELS if k in self.attention_kernels)
        object.__setattr__(self, "attention_kernels", ordered)


def _kernel_bits(kernels):
    return sum(1 << KERNELS.index(k) for k in kernels)


def _kernel_names(bits):
    return tuple(k for i, k in enumerate(KERNELS) if bits >> i & 1)


@dataclass(frozen=True)
class Requirements:
    """A kernel's bounds on the scope: fidelity floor / ceiling, exact (no approx)."""

    min_fidelity: str | None = None
    max_fidelity: str | None = None
    exact: bool = False


def resolve(n, requirements):
    """Numerics `n` with `requirements` applied (unless strict)."""
    if n.strict:
        return n
    rank = FIDELITIES.index(n.fidelity)
    if requirements.min_fidelity:
        rank = max(rank, FIDELITIES.index(requirements.min_fidelity))
    if requirements.max_fidelity:
        rank = min(rank, FIDELITIES.index(requirements.max_fidelity))
    return replace(
        n, fidelity=FIDELITIES[rank], approx=n.approx and not requirements.exact
    )


def _env_default():
    """The process default from KTT_FIDELITY (or KTT_MATMUL_FIDELITY) and KTT_APPROX."""
    fidelity = os.environ.get("KTT_FIDELITY") or os.environ.get("KTT_MATMUL_FIDELITY")
    return Numerics(
        fidelity if fidelity in FIDELITIES else "HiFi2",
        os.environ.get("KTT_APPROX") == "1",
    )


_LOCAL = threading.local()


def _stack():
    if not hasattr(_LOCAL, "stack"):
        _LOCAL.stack = []
    return _LOCAL.stack


def _native():
    """True when the native backend keeps the scope (tt ops read it there)."""
    return hasattr(torch.ops.ktt, "numerics_push")


def current_numerics():
    """The calling thread's Numerics."""
    if _native():
        fidelity, approx, strict, attention, kernels = torch.ops.ktt.numerics_current()
        return Numerics(
            FIDELITIES[fidelity],
            bool(approx),
            bool(strict),
            ATTENTION[attention],
            _kernel_names(kernels),
        )
    stack = _stack()
    return stack[-1] if stack else _env_default()


class numerics(contextlib.ContextDecorator):
    """A scope (context manager or decorator) with the given settings over the
    current ones; settings left None are inherited."""

    def __init__(self, fidelity=None, approx=None, strict=None, attention=None):
        settings = dict(
            fidelity=fidelity, approx=approx, strict=strict, attention=attention
        )
        self.overrides = {k: v for k, v in settings.items() if v is not None}
        replace(Numerics(), **self.overrides)  # validates

    def __enter__(self):
        n = replace(current_numerics(), **self.overrides)
        _stack().append(n)
        if _native():
            torch.ops.ktt.numerics_push(
                FIDELITIES.index(n.fidelity),
                n.approx,
                n.strict,
                ATTENTION.index(n.attention),
                _kernel_bits(n.attention_kernels),
            )
        return n

    def __exit__(self, *exc):
        _stack().pop()
        if _native():
            torch.ops.ktt.numerics_pop()
        return False


class sdpa_kernel(numerics):
    """A scope (context manager or decorator) that runs the attention calls queued in
    it on `kernels` only: names from KERNELS, given as arguments or one list. A call
    none of them takes raises, naming each kernel's refusal."""

    def __init__(self, *kernels):
        names = []
        for k in kernels:
            names.extend([k] if isinstance(k, str) else k)
        self.overrides = {"attention_kernels": tuple(names)}
        replace(Numerics(), **self.overrides)  # validates


def numerics_for(module, fidelity=None, approx=None, strict=None, attention=None):
    """Runs `module`'s forward (and its children's) under numerics(...); returns a
    function that restores the original forward."""
    original = module.forward
    scope = dict(fidelity=fidelity, approx=approx, strict=strict, attention=attention)
    numerics(**scope)  # validates

    @functools.wraps(original)
    def forward(*args, **kwargs):
        with numerics(**scope):
            return original(*args, **kwargs)

    module.forward = forward

    def remove():
        module.forward = original

    return remove
