"""Run manifests: what every benchmark and hardware result records about its setup."""

import datetime
import os
import platform
import subprocess
from pathlib import Path

import torch

from kohakutt.core.doctor import run_doctor

# Environment variables that change TT-Metal behaviour.
ENV_KEYS = (
    "TT_VISIBLE_DEVICES",
    "TT_METAL_SLOW_DISPATCH_MODE",
    "TT_METAL_RUNTIME_ROOT",
    "TT_METAL_WATCHER",
    "TT_METAL_CACHE",
    "GLIBC_TUNABLES",
)


def collect_manifest(doctor_report=None, **extra):
    """A manifest dict: time, host, versions, TT environment, cards and `extra`.

    doctor_report: a `run_doctor()` report to embed (taken now when None).
    """
    report = run_doctor() if doctor_report is None else doctor_report
    return {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": platform.python_version(),
        "versions": _versions(),
        "env": {k: os.environ[k] for k in ENV_KEYS if k in os.environ},
        "cards": [
            {
                key: card[key]
                for key in (
                    "index",
                    "bus_id",
                    "firmware",
                    "pcie",
                    "aiclk_mhz",
                    "power_w",
                    "asic_temp_c",
                )
            }
            for card in report["cards"]
        ],
        "warnings": report["warnings"],
        **extra,
    }


def _versions():
    """kohakutt git revision, tt-metal revision, torch version."""
    return {
        "kohakutt": _git(Path(__file__).resolve().parents[3]),
        "tt_metal": _git(Path(os.environ.get("TT_METAL_RUNTIME_ROOT", "/nonexistent"))),
        "torch": torch.__version__,
    }


def _git(path):
    """`git describe --always --dirty` of `path`, or None."""
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "describe", "--always", "--dirty", "--tags"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None
