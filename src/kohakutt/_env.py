"""Process environment defaults, set before the native library loads."""

import os

# TT-Metal's info and warning log lines are off unless TT_LOGGER_LEVEL is set.
os.environ.setdefault("TT_LOGGER_LEVEL", "ERROR")
