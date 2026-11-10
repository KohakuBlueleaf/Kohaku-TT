"""Regression: in-tree kernels written at the metal level (tests/km_*.py) equal the
backend's own programs bit for bit on the card, and no data size compiles a kernel.
"""

import pytest
import torch

from km_binary import binary_program, plan, register

# (a's shape, b's shape): b broadcast over a's leading dims or over its rows, at
# least 512 tiles (below that the backend runs TT-NN's op).
BINARY_CASES = {
    "heads, block 2": ((8, 256, 320), (256, 320)),
    "heads, block 3": ((2, 12, 256, 320), (256, 320)),
    "heads, block 1": ((7, 320, 320), (320, 320)),
    "heads, ragged": ((5, 300, 500), (300, 500)),
    "rows, block 2": ((640, 1024), (1, 1024)),
    "rows, 3-d": ((4, 320, 512), (1, 512)),
    "rows, block 1": ((800, 672), (1, 672)),
}
OPS = {
    "add": torch.add,
    "sub": torch.sub,
    "mul": torch.mul,
    "div": torch.div,
}
# Complex operands as (a's shape, b's shape) of complex64 values.
COMPLEX_CASES = {
    "heads": ((2, 4, 96, 32), (1, 1, 96, 32)),
    "2-d b": ((2, 4, 96, 32), (96, 32)),
    "same shape": ((2, 4, 96, 32), (2, 4, 96, 32)),
    "ragged": ((3, 2, 50, 13), (50, 13)),
    "model, 9 crops": ((9, 16, 576, 32), (1, 1, 576, 32)),
}


def _operands(a_shape, b_shape, dtype, seed=0):
    """CPU a and b (b away from zero) of `dtype`."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(*a_shape, generator=g).to(dtype)
    b = (torch.rand(*b_shape, generator=g) + 0.5).to(dtype)
    return a, b


def _counted(native, fn):
    """(result of fn(), counter increases over it), with the stream drained."""
    native.synchronize()
    before = native.counters()
    out = fn()
    native.synchronize()
    after = native.counters()
    return out, {name: after[name] - before[name] for name in after}


def _machine(native):
    import kohakutt.dsl.metal as km

    cols, rows = native.compute_grid()
    return km.Machine(cols, rows)


# The sweep runs first: it counts the program builds of core-count classes the
# tests after it would otherwise have built already.
@pytest.mark.hw
def test_binary_sizes_compile_nothing(native):
    ops, dev = register(), native.device(0)
    program = binary_program.trace(
        _machine(native), {"a": "bfloat16", "b": "bfloat16", "out": "bfloat16"}
    )
    seen, blocks, fills = set(), set(), set()
    for round_ in range(2):
        for name, (a_shape, b_shape) in BINARY_CASES.items():
            values = plan(a_shape, b_shape)
            cores = program.core_count(values)
            a, b = (t.to(dev) for t in _operands(a_shape, b_shape, torch.bfloat16))
            _, grown = _counted(native, lambda a=a, b=b: ops.binary_km(a, b, "add"))
            if seen:
                assert grown["kernel_compiles"] == 0, name
                assert grown["program_entries"] == (cores not in seen), name
            if round_:
                assert grown["programs"] == grown["desc_hits"] == 1, name
            seen.add(cores)
            blocks.add(values["block"])
            fills.add(values["fill"])
    assert blocks == {1, 2, 3} and fills == {0, 1} and len(seen) >= 3


@pytest.mark.hw
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("op", list(OPS))
def test_binary_equals_the_backend(native, op, dtype):
    ops, dev = register(), native.device(0)
    for name, (a_shape, b_shape) in BINARY_CASES.items():
        heads = plan(a_shape, b_shape)["fill"] == 0
        if op == "div" and dtype == torch.bfloat16 and heads:
            continue  # the backend runs TT-NN's op for this case
        a, b = (t.to(dev) for t in _operands(a_shape, b_shape, dtype))
        want, grown = _counted(native, lambda a=a, b=b: OPS[op](a, b))
        assert grown["programs"] == 1, name
        got = ops.binary_km(a, b, op)
        assert torch.equal(got.cpu(), want.cpu()), name


@pytest.mark.hw
@pytest.mark.parametrize("conj", [(False, False), (True, False), (False, True)])
def test_complex_mul_equals_the_backend(native, conj):
    ops, dev = register(), native.device(0)
    for name, (a_shape, b_shape) in COMPLEX_CASES.items():
        g = torch.Generator().manual_seed(1)
        a = torch.view_as_complex(torch.randn(*a_shape, 2, generator=g))
        b = torch.view_as_complex(torch.randn(*b_shape, 2, generator=g))
        at, bt = a.to(dev), b.to(dev)
        want = (at.conj() if conj[0] else at) * (bt.conj() if conj[1] else bt)
        a_pairs = torch.view_as_real(a).flatten(-2).to(dev)
        b_pairs = torch.view_as_real(b).flatten(-2).to(dev)
        got = ops.complex_mul_km(a_pairs, b_pairs, *conj)
        assert torch.equal(got.cpu(), torch.view_as_real(want.cpu()).flatten(-2)), name
