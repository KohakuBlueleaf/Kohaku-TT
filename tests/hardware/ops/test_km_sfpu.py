"""Generated vector-unit functions (`km.sfpu_fn`, tests/km_fused.py) on the card: one
generated function equals the chain of tile ops, or the hand-written instruction
code, it replaces, bit for bit; a new shape compiles nothing."""

import pytest
import torch

import kohakutt.dsl.metal as km
from km_chain import run
from km_fused import cases, fused_program

SHAPES = [(256, 512), (3, 96, 160)]


def _counted(native, fn):
    """(result of fn(), counter increases over it), with the stream drained."""
    native.synchronize()
    before = native.counters()
    out = fn()
    native.synchronize()
    after = native.counters()
    return out, {name: after[name] - before[name] for name in after}


@pytest.mark.hw
@pytest.mark.parametrize("case", ["silu", "swiglu", "complex_mul", "chain6"])
def test_one_function_equals_what_it_replaces(native, case):
    dev = native.device(0)
    machine = km.Machine(*native.compute_grid())
    inputs, ops, function = cases()[case]
    row = 2 if case == "chain6" else None
    chain = km.Launcher(fused_program(machine, f"{case}_ops", inputs, ops, row=row))
    fused = km.Launcher(fused_program(machine, f"{case}_fn", inputs, function, row=row))
    g = torch.Generator().manual_seed(4)
    for index, shape in enumerate(SHAPES):
        tensors = [(torch.randn(shape, generator=g) * 2).to(dev) for _ in range(inputs)]
        want = run(chain, tensors)
        got, grown = _counted(native, lambda tensors=tensors: run(fused, tensors))
        if index:
            assert grown["kernel_compiles"] == 0, shape
        assert got.dtype == torch.float32 and got.shape == want.shape
        assert torch.equal(got.cpu(), want.cpu()), (case, shape)
        assert torch.isfinite(got.cpu()).all()


@pytest.mark.hw
def test_six_ops_agree_with_torch(native):
    dev = native.device(0)
    machine = km.Machine(*native.compute_grid())
    inputs, _, function = cases()["chain6"]
    fused = km.Launcher(fused_program(machine, "chain6_fn", inputs, function, row=2))
    g = torch.Generator().manual_seed(5)
    a, b, c, d = (torch.randn(64, 96, generator=g) * 2 for _ in range(4))
    got = run(fused, [t.to(dev) for t in (a, b, c, d)]).cpu()
    # Operand 2 is a row vector per tile: the first row of each 32 rows, repeated.
    row = c.view(2, 32, 96)[:, :1].expand(2, 32, 96).reshape(64, 96)
    want = torch.maximum(a * b + row - d, row).abs() * 0.375
    assert torch.equal(got, want)
