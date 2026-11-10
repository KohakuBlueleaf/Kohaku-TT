"""Programs constructed by code (the explicit builder, km.Launcher) on the card: a
generated elementwise chain runs as one program and equals the eager chain, and a
program rebuilt from a decorated one's model equals it."""

import pytest
import torch

from km_binary import binary_program, plan
from km_chain import chain_program, run
from km_rebuild import rebuild

OPS = {"add": torch.add, "sub": torch.sub, "mul": torch.mul, "div": torch.div}
NODES = [("mul", 1), ("add", 2), ("sub", 0), ("div", 1), ("mul", 2), ("add", 1)]
SHAPES = [(4, 160, 256), (1, 96, 640), (9, 64, 96), (2, 3, 100, 70)]


def _machine(native):
    import kohakutt.dsl.metal as km

    cols, rows = native.compute_grid()
    return km.Machine(cols, rows)


def _counted(native, fn):
    """(result of fn(), counter increases over it), with the stream drained."""
    native.synchronize()
    before = native.counters()
    out = fn()
    native.synchronize()
    after = native.counters()
    return out, {name: after[name] - before[name] for name in after}


def _inputs(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.rand(*shape, generator=g) + 0.5 for _ in range(3)]


@pytest.mark.hw
def test_a_generated_chain_equals_the_eager_chain(native):
    import kohakutt.dsl.metal as km

    dev = native.device(0)
    launcher = km.Launcher(chain_program(_machine(native), NODES, 3))
    for seed, shape in enumerate(SHAPES):
        cpu = _inputs(shape, seed)
        xs = [t.to(dev) for t in cpu]
        got, grown = _counted(native, lambda xs=xs: run(launcher, xs))
        assert grown["programs"] == 1, shape
        if seed:
            assert grown["kernel_compiles"] == 0, shape
        eager, want = xs[0], cpu[0]
        for op, index in NODES:
            eager, want = OPS[op](eager, xs[index]), OPS[op](want, cpu[index])
        assert got.shape == eager.shape and got.dtype == torch.float32
        assert torch.equal(got.cpu(), eager.cpu()), shape
        assert torch.allclose(got.cpu(), want, rtol=1e-5, atol=1e-6), shape


@pytest.mark.hw
def test_a_rebuilt_program_equals_the_decorated_one(native):
    import kohakutt.dsl.metal as km

    dev = native.device(0)
    dtypes = {"a": "bfloat16", "b": "bfloat16", "out": "bfloat16"}
    decorated = binary_program.trace(_machine(native), dtypes, op="mul")
    launchers = [km.Launcher(decorated), km.Launcher(rebuild(decorated))]
    assert launchers[0].sources == launchers[1].sources
    for a_shape, b_shape in [((8, 256, 320), (256, 320)), ((640, 1024), (1, 1024))]:
        g = torch.Generator().manual_seed(0)
        a = torch.randn(*a_shape, generator=g).bfloat16().to(dev)
        b = torch.randn(*b_shape, generator=g).bfloat16().to(dev)
        outs = [
            launcher(
                {"a": a, "b": b},
                outputs={"out": (list(a_shape), torch.bfloat16)},
                runtime=plan(a_shape, b_shape),
            )[0].cpu()
            for launcher in launchers
        ]
        assert torch.equal(outs[0], outs[1]), a_shape
        assert torch.equal(outs[0], (a * b).cpu()), a_shape
