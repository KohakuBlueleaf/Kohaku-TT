"""The block matmul written at the metal level (examples/kernels/dsl_metal/gemm_km.py)
on the card: equal bit for bit to the backend's block matmul (ktt_gemm, `KTT_GEMM=1`)
run with the same one-row-per-core plan, over shapes that take one, two and several
K blocks, a short last K block, ragged and batched sizes and w stored [N, K]."""

import functools
import importlib.util
import math
import os
from pathlib import Path

import pytest
import torch

import kohakutt.dsl.metal as km

# The backend's block matmul (read at the process's first matmul).
os.environ["KTT_GEMM"] = "1"

EXAMPLE = Path(__file__).resolve().parents[3] / "examples/kernels/dsl_metal/gemm_km.py"
# (a's shape, N, w stored [N, K]).
SWEEP = [
    ((4096, 1024), 768, False),
    ((4096, 1024), 768, True),
    ((1000, 256), 333, False),
    ((3, 200, 512), 96, True),
    ((16384, 992), 256, False),
    ((2048, 2048), 512, False),
    ((14080, 512), 32, False),
    ((14080, 1024), 64, False),
    ((14080, 2048), 32, True),
]


@functools.cache
def _example():
    spec = importlib.util.spec_from_file_location("gemm_km_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _operands(a_shape, n, w_nk, seed):
    g = torch.Generator().manual_seed(seed)
    k = a_shape[-1]
    a = torch.randn(*a_shape, generator=g).bfloat16()
    w = (torch.randn((n, k) if w_nk else (k, n), generator=g) / 16).bfloat16()
    return a, w


@pytest.mark.hw
def test_gemm_equals_the_backends_block_matmul(native, capfd):
    module, dev = _example(), native.device(0)
    cols, rows = native.compute_grid()
    launcher = km.Launcher(module.traced(cols, rows))
    passes = set()
    for seed, (a_shape, n, w_nk) in enumerate(SWEEP):
        a, w = (t.to(dev) for t in _operands(a_shape, n, w_nk, seed))
        mt = math.prod(a_shape[:-2]) * math.ceil(a_shape[-2] / 32)
        kt, nt = a_shape[-1] // 32, math.ceil(n / 32)
        values = module.plan(mt, kt, nt, cols, rows)
        assert values is not None, a_shape
        passes.add(min(values["nk"], 3))
        forced = [values[k] for k in ("bm", "bn", "nbh", "nbw", "kb", "sh", "sw")]
        os.environ["KTT_GEMM_PLAN"] = ",".join(map(str, [1, *forced]))
        os.environ["KTT_GEMM_TRACE"] = "1"
        try:
            want = torch.nn.functional.linear(a, w) if w_nk else a @ w
            native.synchronize()
        finally:
            del os.environ["KTT_GEMM_PLAN"], os.environ["KTT_GEMM_TRACE"]
        trace = capfd.readouterr().err
        assert "ktt_gemm forced: 1d" in trace, (a_shape, trace[-300:])
        (got,) = launcher.run(
            {"a": a, "w": w},
            {"out": ([*a_shape[:-1], n], torch.bfloat16)},
            {**values, "w_nk": int(w_nk)},
        )
        assert got.shape == want.shape
        assert torch.equal(got.cpu(), want.cpu()), (a_shape, n, w_nk, values)
    assert passes == {1, 2, 3}
