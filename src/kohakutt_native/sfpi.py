"""Fetches the RISC-V kernel compiler (sfpi) of the bundled tt-metal into the cache."""

import hashlib
import platform
import re
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import kohakutt_native

VERSION_FILE = kohakutt_native.BUNDLE / "root" / "tt_metal" / "sfpi-version"


def release(text, machine, distro):
    """(url, sha256) of the sfpi archive for `machine` and `distro` named by the
    text of tt-metal's sfpi-version file."""
    values = dict(re.findall(r"(?m)^sfpi_(\w+)='([^']*)'", text))
    name = f"sfpi_{values['version']}_{machine}_{distro}.txz"
    url = f"{values['repo']}/releases/download/{values['version']}/{name}"
    return url, values[f"{machine}_{distro}_txz_hash"]


def distro():
    """The sfpi build for this host: "fedora" on Fedora / RHEL-like, else "debian"."""
    try:
        text = Path("/etc/os-release").read_text()
    except OSError:
        return "debian"
    return "fedora" if re.search(r"(?m)^ID(_LIKE)?=.*(fedora|rhel)", text) else "debian"


def fetch(version_file=VERSION_FILE, target=None):
    """Downloads and unpacks the pinned sfpi into `target` (default: the cache)
    unless it is already there; returns its directory."""
    target = Path(target or kohakutt_native.cache_dir())
    if (target / "sfpi" / "compiler" / "bin" / "riscv-tt-elf-g++").exists():
        return target / "sfpi"
    url, sha256 = release(Path(version_file).read_text(), platform.machine(), distro())
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target) as tmp:
        archive = Path(tmp) / "sfpi.txz"
        digest = hashlib.sha256()
        with urllib.request.urlopen(url) as response, archive.open("wb") as out:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != sha256:
            raise RuntimeError(f"kohakutt: {url} does not match its recorded sha256")
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        (Path(tmp) / "sfpi").replace(target / "sfpi")
    return target / "sfpi"
