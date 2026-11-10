"""Regression: metal-level programs (kohakutt.dsl.metal) on the card. They equal the
hand-written examples bit for bit, and a new data size compiles no kernel: program
builds happen once per core-count class.
"""

import functools
import importlib.util
import math
from pathlib import Path

import pytest
import torch

EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "kernels"

# RoPE inputs (batch, heads, tokens) at dim 64: tiles = batch * heads *
# ceil(tokens / 32) * 2. (1, 5, 32) is 10 tiles on 12 cores: two take no work.
ROPE_SHAPES = [
    (2, 4, 64),
    (1, 4, 128),
    (3, 4, 64),
    (1, 2, 256),
    (1, 1, 32),
    (1, 2, 32),
    (1, 5, 32),
    (2, 8, 64),
    (9, 16, 576),
]
# SwiGLU inputs (rows, cols); (160, 64) is 10 tiles on 12 cores.
SWIGLU_SHAPES = [(256, 256), (512, 128), (300, 250), (32, 64), (160, 64), (1024, 4096)]


def _cores(native, units):
    """The core count a program splitting `units` runs on."""
    import kohakutt.dsl.metal as km

    classes = km.Machine(*native.compute_grid()).classes()
    return next((c for c in classes if c >= units), classes[-1])


@functools.cache
def _example(name):
    """Example module `name`; None when another test registered its op first."""
    if f"kohakutt_example::{name}" in torch._C._dispatch_get_all_op_names():
        return None
    folder = "dsl_metal" if name.endswith("_km") else "metalium"
    path = EXAMPLES / folder / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rope_inputs(dev, batch, heads, tokens, dim=64):
    gen = torch.Generator().manual_seed(batch * 1000 + heads * 100 + tokens)
    x = torch.randn(batch, heads, tokens, dim, generator=gen).bfloat16()
    angles = torch.rand(tokens, dim // 2, generator=gen) * 6.2831853
    table = torch.stack([angles.cos(), angles.sin()], dim=-1).reshape(tokens, dim)
    return x.to(dev), table.contiguous().to(dev)


def _counted(native, fn):
    """(result of fn(), counter increases over it), with the stream drained."""
    native.synchronize()
    before = native.counters()
    out = fn()
    native.synchronize()
    after = native.counters()
    return out, {name: after[name] - before[name] for name in after}


@pytest.mark.hw
def test_rope_equals_hand_written_and_sizes_compile_nothing(native):
    for name in ("rope", "rope_km"):
        _example(name)
    ops = torch.ops.kohakutt_example
    dev = native.device(0)

    seen = set()
    for shape in ROPE_SHAPES:
        batch, heads, tokens = shape
        cores = _cores(native, batch * heads * math.ceil(tokens / 32) * 2)
        x, table = _rope_inputs(dev, *shape)
        out, grown = _counted(native, lambda x=x, table=table: ops.rope_km(x, table))
        if seen:
            assert grown["kernel_compiles"] == 0, shape
            assert grown["program_entries"] == (cores not in seen), shape
        assert torch.equal(out.cpu(), ops.rope(x, table).cpu()), shape
        seen.add(cores)
        _, again = _counted(native, lambda x=x, table=table: ops.rope_km(x, table))
        assert again["kernel_compiles"] == again["program_entries"] == 0, shape
        assert again["programs"] == again["desc_hits"] == 1, shape
    assert len(seen) >= 6


@pytest.mark.hw
def test_swiglu_equals_hand_written_and_sizes_compile_nothing(native):
    for name in ("swiglu", "swiglu_km"):
        _example(name)
    ops = torch.ops.kohakutt_example
    dev = native.device(0)
    gen = torch.Generator().manual_seed(0)

    def inputs(rows, cols):
        a = torch.randn(rows, cols, generator=gen).bfloat16()
        b = torch.randn(rows, cols, generator=gen).bfloat16()
        return a.to(dev), b.to(dev)

    seen = set()
    for rows, cols in SWIGLU_SHAPES:
        cores = _cores(native, math.ceil(rows / 32) * math.ceil(cols / 32))
        a, b = inputs(rows, cols)
        out, grown = _counted(native, lambda a=a, b=b: ops.swiglu_km(a, b))
        if seen:
            assert grown["kernel_compiles"] == 0, (rows, cols)
            assert grown["program_entries"] == (cores not in seen), (rows, cols)
        assert torch.equal(out.cpu(), ops.swiglu(a, b).cpu()), (rows, cols)
        seen.add(cores)
    assert len(seen) >= 5


@pytest.mark.hw
def test_pinned_nocs_scratch_and_semaphore(native):
    import kohakutt.dsl.metal as km
    import kohakutt.kernels as kk

    @km.program("copy_pinned")
    def copy_program(p, x: km.TensorRef, out: km.Out):
        start, count = p.split(p.runtime("tiles"))
        cb_x, cb_out = p.cb(0, like=x), p.cb(16, like=out)
        p.scratch(256)
        p.semaphore(initial=1)

        @p.dm0(noc=1)
        def reader(k):
            for i in k.range(count):
                slot = cb_x.reserve(1)
                k.read(x.page(start + i), slot)
                k.reads_done()
                cb_x.push(1)

        @p.compute
        def compute(c):
            c.startup(cb_x, cb_x, cb_out)
            c.init("copy", cb_x)
            for _ in c.range(count):
                cb_x.wait(1)
                with c.registers():
                    c.copy(cb_x, 0, 0)
                    cb_out.reserve(1)
                    c.pack(0, cb_out)
                    cb_out.push(1)
                cb_x.pop(1)

        @p.dm1(noc=0)
        def writer(k):
            for i in k.range(count):
                cb_out.wait(1)
                k.write(cb_out, out.page(start + i))
                k.flushed()
                cb_out.pop(1)
            k.writes_done()

    @km.op("kohakutt_test::copy_pinned(Tensor x) -> Tensor", copy_program)
    def copy(x):
        return km.Call(
            inputs={"x": x},
            outputs={"out": (list(x.shape), x.dtype)},
            runtime={"tiles": kk.tile_count(x)},
        )

    x = torch.randn(3, 96, 160).bfloat16()
    out = torch.ops.kohakutt_test.copy_pinned(x.to(native.device(0)))
    assert torch.equal(out.cpu(), x)
