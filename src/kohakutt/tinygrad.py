"""`import kohakutt.tinygrad` registers the tinygrad device "KTT" (runs on the card).
Knobs: KTT_TINYGRAD_EXECUTOR=numpy (host arrays), KTT_TINYGRAD_OVERRIDES=0 (no
library-op overrides), KTT_TINYGRAD_RAND=host (tinygrad's CPU random stream),
KTT_TINYGRAD_DTYPES=error (raise on non-device dtypes). `kohakutt` imports torch."""

import os

from kohakutt.tinygrad_backend import close, install, numerics, overrides, use_executor
from kohakutt.tinygrad_backend.errors import DtypeError, DtypeWarning, HostKernelError

install(
    executor=os.environ.get("KTT_TINYGRAD_EXECUTOR", "bridge"),
    overrides=os.environ.get("KTT_TINYGRAD_OVERRIDES", "1") != "0",
)

__all__ = [
    "DtypeError",
    "DtypeWarning",
    "HostKernelError",
    "close",
    "install",
    "numerics",
    "overrides",
    "use_executor",
]
