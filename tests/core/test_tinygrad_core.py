"""The C ABI of _ktt_native.so and its tinygrad binding without a card: it loads, every
symbol resolves, the image is the one torch loaded, calls without a device raise."""

from pathlib import Path

import pytest

pytest.importorskip("tinygrad.engine")

from kohakutt.tinygrad_backend import core

pytestmark = pytest.mark.skipif(not core.available(), reason="_ktt_native.so not built")


def test_library_loads_and_symbols_resolve():
    lib = core.load()
    assert lib.ktt_abi_version() == core.ABI_VERSION


def test_binding_shares_the_torch_extension():
    """The binding's library is the image torch loaded: one bridge per process."""
    import ctypes

    import kohakutt  # noqa: F401  (loads the extension into torch)

    lib = core.load()
    other = ctypes.CDLL(str(core.library_path()))
    assert (
        ctypes.cast(lib.ktt_open, ctypes.c_void_p).value
        == ctypes.cast(other.ktt_open, ctypes.c_void_p).value
    )
    maps = Path("/proc/self/maps").read_text().splitlines()
    assert len({m.split()[-1] for m in maps if "_ktt_native" in m}) == 1


def test_numerics_scope_keeps_the_attention_kernels():
    """The binding's numerics scope inside kohakutt.sdpa_kernel keeps its kernels."""
    from kohakutt import current_numerics, sdpa_kernel

    core.load()
    with sdpa_kernel("math", "flash"):
        with core.numerics(fidelity=3):
            n = current_numerics()
            assert n.fidelity == "HiFi4"
            assert n.attention_kernels == ("flash", "math")
        assert current_numerics().attention_kernels == ("flash", "math")
    assert current_numerics().attention_kernels == ("flash", "own", "ttnn", "math")


def test_enums_match_the_bridge_header():
    assert core.UNARY["neg"] == 0 and core.UNARY["i0"] == 45 and len(core.UNARY) == 46
    assert (
        core.BINARY["add"] == 0 and core.BINARY["lcm"] == 20 and len(core.BINARY) == 21
    )


def test_calls_without_device_raise():
    core.load()
    assert not core.is_open()
    import numpy as np

    with pytest.raises(core.CoreError):
        core.upload(np.zeros((32, 32), np.float32))
