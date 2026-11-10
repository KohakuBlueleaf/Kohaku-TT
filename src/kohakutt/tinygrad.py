"""`import kohakutt.tinygrad` registers the tinygrad device "KTT" (runs on the card).
KTT_TINYGRAD_EXECUTOR=numpy runs on host arrays, KTT_TINYGRAD_OVERRIDES=0 turns the
library-op overrides off, KTT_TINYGRAD_RAND=host draws tinygrad's own (CPU) random
stream. The package `kohakutt` still imports torch today."""

import os

from kohakutt.tinygrad_backend import close, install, numerics, overrides, use_executor

install(
    executor=os.environ.get("KTT_TINYGRAD_EXECUTOR", "bridge"),
    overrides=os.environ.get("KTT_TINYGRAD_OVERRIDES", "1") != "0",
)

__all__ = ["close", "install", "numerics", "overrides", "use_executor"]
