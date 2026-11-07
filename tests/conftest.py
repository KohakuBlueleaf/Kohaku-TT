"""Test setup: `hw` and `dual` tests run only with KTT_HW=1 on the P150A host;
tests/hardware is not collected where ttnn is not installed."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

collect_ignore_glob = [] if importlib.util.find_spec("ttnn") else ["hardware/*"]


def pytest_collection_modifyitems(config, items):
    """Skips hardware tests unless KTT_HW=1."""
    if os.environ.get("KTT_HW") == "1":
        return
    skip = pytest.mark.skip(reason="needs Tenstorrent cards (KTT_HW=1)")
    for item in items:
        if "hw" in item.keywords or "dual" in item.keywords:
            item.add_marker(skip)
