"""The tinygrad backend stays below PyTorch and inside its compat module."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "src/kohakutt/tinygrad_backend"
FILES = [*PACKAGE.rglob("*.py"), REPO / "src/kohakutt/tinygrad.py"]
TORCH = re.compile(
    r"^\s*(import|from)\s+(torch|kohakutt_native"
    r"|kohakutt\.(torch_backend|device|kernels|fusion|attention))\b",
    re.M,
)
TINYGRAD = re.compile(r"^\s*(import|from)\s+tinygrad\b", re.M)


def test_no_torch_side_imports():
    """Only core.py, the binding of the shared bridge, may reach into kohakutt (its
    registration and device session), and nothing else of the torch side."""
    others = [p for p in FILES if p.name != "core.py"]
    offenders = [
        str(p.relative_to(REPO)) for p in others if TORCH.search(p.read_text())
    ]
    assert not offenders, offenders
    core = (PACKAGE / "core.py").read_text()
    reached = {m.group(0).split()[-1] for m in TORCH.finditer(core)}
    assert reached <= {"kohakutt.torch_backend", "kohakutt.device"}, reached


def test_no_torch_calls():
    offenders = [
        str(p.relative_to(REPO))
        for p in FILES
        if re.search(r"torch\.ops|aten::|\.to\(.tt.\)", p.read_text())
    ]
    assert not offenders, offenders


def test_tinygrad_internals_only_in_compat():
    allowed = {PACKAGE / "compat.py"}
    offenders = [
        str(p.relative_to(REPO))
        for p in FILES
        if p not in allowed and TINYGRAD.search(p.read_text())
    ]
    assert not offenders, offenders
