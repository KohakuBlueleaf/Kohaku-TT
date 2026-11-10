"""Errors and warnings of the KTT device, named by `kohakutt.tinygrad`."""

from __future__ import annotations

import os

DEVICE_DTYPES = ("float32", "bfloat16", "int32")


class DtypeError(TypeError):
    """A kernel uses a dtype KTT does not run: not a device dtype, and the host
    evaluator does not model it or KTT_TINYGRAD_DTYPES=error is set."""


class DtypeWarning(UserWarning):
    """Kernels on a dtype that is not a device dtype run on the host (once a dtype)."""


class HostKernelError(RuntimeError):
    """The host evaluator failed on a kernel the card did not take."""


def dtype_policy() -> str:
    """KTT_TINYGRAD_DTYPES: "host" (default) runs kernels on other dtypes on the host,
    counted and warned once per dtype; "error" raises DtypeError instead."""
    policy = os.environ.get("KTT_TINYGRAD_DTYPES", "host")
    if policy not in ("host", "error"):
        raise ValueError(f"KTT_TINYGRAD_DTYPES={policy!r}: use 'host' or 'error'")
    return policy


def not_modelled(name: str) -> DtypeError:
    return DtypeError(
        f"kohakutt.tinygrad: {name} is not supported on KTT: not a device dtype "
        f"({', '.join(DEVICE_DTYPES)}), and the host evaluator does not model it"
    )


def not_on_device(name: str) -> DtypeError:
    return DtypeError(
        f"kohakutt.tinygrad: {name} is not a device dtype of KTT "
        f"({', '.join(DEVICE_DTYPES)}); KTT_TINYGRAD_DTYPES=error stops its host "
        f"kernels (unset it to run them on the host, counted)"
    )


def host_warning(name: str) -> str:
    return (
        f"kohakutt.tinygrad: {name} is not a device dtype of KTT "
        f"({', '.join(DEVICE_DTYPES)}): kernels on it run on the host, counted in "
        f"executor.FALLBACKS['dtype {name}'] (KTT_TINYGRAD_DTYPES=error raises instead)"
    )
