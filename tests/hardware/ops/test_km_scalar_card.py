"""A generated function's fp32 scalar argument and its bf16 rounding node on the card
(tests/km_scalar.py): out = bf16(a * scale) + b, a new scale compiling nothing."""

import pytest
import torch

from km_scalar import run, scalar_program


@pytest.mark.hw
def test_scalar_argument_and_bf16_rounding(native):
    import kohakutt.dsl.metal as km

    dev = native.device(0)
    cols, rows = native.compute_grid()
    launcher = km.Launcher(scalar_program(km.Machine(cols, rows)))
    g = torch.Generator().manual_seed(0)
    a = torch.randn(64, 96, generator=g)
    b = torch.randn(64, 96, generator=g)
    for i, scale in enumerate((1.7, -0.3125)):
        native.synchronize()
        before = native.counters()["kernel_compiles"]
        out = run(launcher, a.to(dev), b.to(dev), scale).cpu()
        assert i == 0 or native.counters()["kernel_compiles"] == before
        rounded = (a * scale).bfloat16().float() + b
        torch.testing.assert_close(out, rounded, rtol=1e-6, atol=1e-6)
        assert not torch.allclose(out, a * scale + b, rtol=1e-6, atol=1e-6)
