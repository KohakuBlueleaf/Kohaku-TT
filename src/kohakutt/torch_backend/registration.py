"""Registers the "tt" PyTorch device: loads the native layer, renames PrivateUse1,
installs torch.tt and the Python-side patches. Never opens a device."""

import atexit
from pathlib import Path

import torch

import kohakutt_native
from kohakutt.torch_backend import device_module, patches, varlen


def install():
    """Registers tt once per process; returns whether the native layer is built."""
    if not kohakutt_native.load():
        return False
    if torch._C._get_privateuse1_backend_name() == "tt":
        return True
    torch.utils.rename_privateuse1_backend("tt")
    torch._register_device_module("tt", device_module)
    torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)
    patches.install()
    varlen.install()
    # From interpreter exit on, live tt handles are leaked, not freed (TT-Metal's
    # statics may be gone when they are destroyed).
    atexit.register(torch.ops.ktt.mark_exiting)
    return True


def fetch_sfpi():
    """Fetches the RISC-V kernel compiler of the bundled tt-metal; its directory."""
    from kohakutt_native import sfpi

    return sfpi.fetch()


def native_dir():
    """The native package's directory (the library, include/, cmake/)."""
    return Path(kohakutt_native.__file__).resolve().parent
