"""Kohaku-TT's compute / system layer: the compiled extension (TT-NN bridge, tt tensor
model, PrivateUse1 aten kernels, trace runtime). Loaded only by kohakutt; no user API.
"""

import os
from pathlib import Path

import torch

LIBRARY = Path(__file__).with_name("_ktt_native.so")
# An installed wheel carries the tt-metal libraries and kernel runtime tree here.
BUNDLE = Path(__file__).with_name("_tt")
SFPI_SYSTEM = Path("/opt/tenstorrent/sfpi")


def cache_dir():
    """Kohaku-TT's per-user cache: KOHAKUTT_CACHE, else the XDG cache directory."""
    if os.environ.get("KOHAKUTT_CACHE"):
        return Path(os.environ["KOHAKUTT_CACHE"])
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "kohakutt"


def sfpi_dir():
    """The RISC-V kernel compiler: KOHAKUTT_SFPI, the cache's copy or the system's;
    None when there is none."""
    choices = [os.environ.get("KOHAKUTT_SFPI"), cache_dir() / "sfpi", SFPI_SYSTEM]
    for path in filter(None, choices):
        if (Path(path) / "compiler" / "bin" / "riscv-tt-elf-g++").exists():
            return Path(path)
    return None


def bundled_root():
    """A runtime root for the bundled tt-metal: links to the wheel's kernel tree and
    to the kernel compiler, in the cache (the wheel's own files are never written)."""
    version = (BUNDLE / "tt-metal-version").read_text().strip()
    root = cache_dir() / f"runtime-{version}"
    source = BUNDLE / "root"
    for entry in source.iterdir():
        if entry.name != "runtime":
            _link(root / entry.name, entry)
    for entry in (source / "runtime").iterdir():
        _link(root / "runtime" / entry.name, entry)
    sfpi = sfpi_dir()
    if sfpi is None:
        raise RuntimeError(
            "kohakutt: the RISC-V kernel compiler (sfpi) was not found; run "
            "`ktt sfpi` to fetch it, or set KOHAKUTT_SFPI to an sfpi directory"
        )
    if sfpi != SFPI_SYSTEM:
        _link(root / "runtime" / "sfpi", sfpi)
    return root


def _link(link, target):
    """Makes `link` a symlink to `target` (replacing a stale one)."""
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() and link.resolve() == target.resolve():
        return
    if link.is_symlink():
        link.unlink()
    link.symlink_to(target, target_is_directory=target.is_dir())


def load():
    """Loads the extension into torch (idempotent); False when it is not built."""
    if not LIBRARY.exists():
        return False
    if not hasattr(torch.ops.ktt, "open"):
        if BUNDLE.exists() and not os.environ.get("TT_METAL_RUNTIME_ROOT"):
            root = str(bundled_root())
            os.environ["TT_METAL_RUNTIME_ROOT"] = root
            os.environ.setdefault("TT_METAL_HOME", root)
        torch.ops.load_library(str(LIBRARY))
    return True
