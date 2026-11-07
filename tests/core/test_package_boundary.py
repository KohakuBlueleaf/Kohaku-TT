"""Regression: kohakutt_native is imported only by kohakutt's registration module;
users, examples, tests, benches and tools go through kohakutt."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ALLOWED = {REPO / "src/kohakutt/torch_backend/registration.py"}
IMPORT = re.compile(r"^\s*(import|from)\s+kohakutt_native\b", re.M)


def test_only_registration_imports_native():
    offenders = [
        str(path.relative_to(REPO))
        for root in ("src/kohakutt", "tests", "examples")
        for path in (REPO / root).rglob("*.py")
        if path not in ALLOWED and IMPORT.search(path.read_text())
    ]
    assert not offenders, offenders
