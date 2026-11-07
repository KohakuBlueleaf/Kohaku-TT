"""C++ extension libraries: build paths for CMake and loading (guide:
examples/custom_kernel_cpp/)."""

from pathlib import Path

import torch

from kohakutt.torch_backend import registration


def cmake_module():
    """Path of KohakuTT.cmake (kohakutt_add_extension) for an extension's CMakeLists."""
    return str(registration.native_dir() / "cmake" / "KohakuTT.cmake")


def include_dir():
    """Directory holding ktt/extension/{device,torch}.hpp."""
    return str(registration.native_dir() / "include")


def load(path):
    """Loads extension library `path` after the native backend; its ops appear under
    torch.ops.<namespace>."""
    if not registration.install():
        raise RuntimeError("kohakutt: the native backend is not built")
    torch.ops.load_library(str(Path(path).resolve()))
