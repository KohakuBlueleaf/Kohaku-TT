"""tinygrad backend: the device `KTT`, which takes kernel ASTs at the to_program seam.
`install()` registers the device and the seam once; users `import kohakutt.tinygrad`.
"""

from __future__ import annotations

import importlib

try:
    importlib.import_module("tinygrad.engine")
except ImportError as error:
    raise ImportError(
        "kohakutt.tinygrad needs tinygrad v0.14.x, which is not importable "
        f"({error}); install it or put it on PYTHONPATH"
    ) from error

_installed = False


def install(executor: str | None = None, overrides: bool = True):
    """Register the KTT device; `executor` is "bridge" (card) or "numpy" (no card).
    `overrides` turns the library-op overrides on or off. Later calls only switch
    the executor and the overrides.
    """
    global _installed
    from kohakutt.tinygrad_backend import compat, device
    from kohakutt.tinygrad_backend import overrides as ov

    if not _installed:
        compat.check_version()
        compat.set_knobs()
        compat.register_device(device.KTTDevice)
        compat.set_seam(device.seam)
        _installed = True
    if executor is not None:
        use_executor(executor)
    ov.enable(overrides)


def numerics(
    fidelity: int = 1, approx: bool = False, strict: bool = False, attention: int = 0
):
    """Numerics scope for kernels queued inside it: fidelity 0..3 (LoFi..HiFi4),
    SFPU approximations, strict mode, attention mode 0..2. No-op on numpy.
    """
    import contextlib

    from kohakutt.tinygrad_backend import device

    backend = device.get_backend()
    core = getattr(backend, "raw_core", None)
    if core is None:
        return contextlib.nullcontext()
    backend.ensure_open()
    return core.numerics(fidelity, approx, strict, attention)


def close():
    """Drop the device tensors of live buffers (host copies kept), release the card."""
    from kohakutt.tinygrad_backend import device

    backend = device.get_backend()
    if hasattr(backend, "close"):
        backend.close()


def use_executor(name: str):
    from kohakutt.tinygrad_backend import device

    if name == "numpy":
        device.set_backend(device.NumpyBackend())
    elif name == "bridge":
        from kohakutt.tinygrad_backend import executor

        device.set_backend(executor.BridgeBackend())
    elif name == "sim":
        from kohakutt.tinygrad_backend import core_sim, executor

        device.set_backend(executor.BridgeBackend(core=core_sim))
    else:
        raise ValueError(f"unknown executor {name!r}; use 'bridge', 'sim' or 'numpy'")
