"""tinygrad's DEBUG=2 report of KTT kernels on the host executor (no card): kernel
names, operation and byte estimates, and the measured time of a launch with wait."""

import re

import numpy as np
import pytest

pytest.importorskip("tinygrad.engine")

from tinygrad.helpers import GlobalCounters, ansistrip

from kohakutt.tinygrad_backend import device as D
from kohakutt.tinygrad_backend import install
from kohakutt.tinygrad_backend import overrides as ov
from tinygrad import Context, Tensor, TinyJit, Variable

TIME = re.compile(r" tm +([\d.]+)(us|ms|s) */")
SCALE = {"us": 1e-6, "ms": 1e-3, "s": 1.0}


@pytest.fixture(scope="module", autouse=True)
def numpy_device():
    backend, overrides = D.get_backend(), ov.enabled()
    install(executor="numpy", overrides=True)
    yield
    D.set_backend(backend)
    ov.enable(overrides)


def _kernel_lines(capsys, fn):
    """(name, seconds) of each KTT kernel line DEBUG=2 prints while `fn` runs."""
    capsys.readouterr()
    with Context(DEBUG=2):
        fn()
    out = []
    for line in map(ansistrip, capsys.readouterr().out.splitlines()):
        fields = line.split()
        if line.startswith("*** KTT") and fields[3] != "copy":
            time = TIME.search(line)
            out.append((fields[3], float(time[1]) * SCALE[time[2]]))
    return out


def _host(a):
    return Tensor(np.asarray(a, np.float32), device="KTT").realize()


def test_matmul_line_and_counters(capsys):
    a, b = _host(np.ones((4, 6))), _host(np.ones((6, 3)))
    GlobalCounters.reset()
    lines = _kernel_lines(capsys, lambda: (a @ b).realize())
    assert [name for name, _ in lines] == ["r_4_3_6"]
    assert lines[0][1] > 0
    assert GlobalCounters.global_ops == 2 * 4 * 3 * 6
    assert GlobalCounters.global_mem == (24 + 18 + 12) * 4
    assert GlobalCounters.time_sum_s >= lines[0][1] - 1e-8  # printed to 0.01 us


def test_no_kernel_is_reported_as_test(capsys):
    x, w = _host(np.ones((8, 16))), _host(np.ones((16, 4)))

    def step():
        y = (x @ w).relu()
        (y - y.max(axis=1, keepdim=True)).exp().sum().realize()

    lines = _kernel_lines(capsys, step)
    assert lines and all(name != "test" and t > 0 for name, t in lines)
    assert {name[0] for name in dict(lines)} <= {"E", "r"}


def test_library_kernel_name_and_estimates(capsys):
    x, w = _host(np.ones((1, 2, 5, 5))), _host(np.ones((3, 2, 3, 3)))
    GlobalCounters.reset()
    lines = _kernel_lines(capsys, lambda: x.conv2d(w).realize())
    assert [name for name, _ in lines] == ["ktt_conv2d_1_3_3_3"]
    assert GlobalCounters.global_ops == 2 * 27 * 18
    assert GlobalCounters.global_mem == (27 + 50 + 54) * 4


def test_symbolic_length_in_a_jit(capsys):
    """A Variable length: its name in the kernel name, estimates at the bound value."""
    big, w = _host(np.ones((16, 8))), _host(np.ones((8, 4)))
    length = Variable("T", 1, 16)

    @TinyJit
    def step(n):
        return (big[:n] @ w).realize()

    for n in (3, 4, 5):
        GlobalCounters.reset()
        lines = _kernel_lines(capsys, lambda n=n: step(length.bind(n)))
        assert [name for name, _ in lines] == ["r_T_4_8"], n
        assert GlobalCounters.global_ops == 2 * n * 4 * 8
        assert GlobalCounters.global_mem == (n * 8 + 32 + n * 4) * 4
