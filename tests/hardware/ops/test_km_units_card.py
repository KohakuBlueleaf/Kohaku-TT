"""The compute constructs of tests/km_units.py on the card: block matmul, table ops,
a broadcast unpack under a runtime branch, packs at a page and packer accumulation
agree with the same arithmetic in torch, and sizes compile nothing."""

import pytest
import torch

from km_units import run, units_program


def _reference(x, w, g, flag, bias):
    """(out, acc) in fp32 on the CPU."""
    y = torch.relu(x.float() @ w.float()) + bias
    out = y * g.float()[:1] if flag else y - 1.0
    return out, out[:, :32] + out[:, 32:]


def _every_op(machine, dtype):
    """A program applying every op of the op table once to a tile of x."""
    import kohakutt.dsl.metal as km
    from kohakutt.dsl.metal.ops import OPS

    p = km.Program(f"every_op_{dtype}", machine)
    x, out = p.input("x", dtype), p.output("out", dtype)
    start, count = p.split(p.runtime("tiles"))
    cb_x, cb_out = p.cb(0, like=x), p.cb(16, like=out)
    k = p.kernel("dm0")
    with k.loop(count) as i:
        slot = k.reserve(cb_x, 1)
        k.read(x.page(start + i), slot)
        k.reads_done()
        k.push(cb_x, 1)
    fp32 = [cb_x] if dtype == "float32" else []
    c = p.kernel("compute", fp32_dest=True, unpack_fp32=fp32)
    c.startup(cb_x, None, cb_out)
    with c.loop(count):
        c.wait(cb_x, 1)
        c.acquire()
        c.init("copy", cb_x)
        for slot in range(4):
            c.copy(cb_x, 0, slot)
        for name, entry in OPS.items():
            if entry.call is None:
                continue
            c.init(name)
            c.op(name, *range(entry.slots), values=[0.5] * entry.values)
        c.reserve(cb_out, 1)
        c.pack(0, cb_out)
        c.release()
        c.push(cb_out, 1)
        c.pop(cb_x, 1)
    w = p.kernel("dm1")
    with w.loop(count) as i:
        w.wait(cb_out, 1)
        w.write(cb_out, out.page(start + i))
        w.flushed()
        w.pop(cb_out, 1)
    w.writes_done()
    return p


@pytest.mark.hw
def test_every_table_op_compiles_and_runs(native):
    import kohakutt.dsl.metal as km

    dev = native.device(0)
    cols, rows = native.compute_grid()
    launcher = km.Launcher(_every_op(km.Machine(cols, rows), "float32"))
    x = torch.rand(64, 96).to(dev)
    (out,) = launcher(
        {"x": x}, outputs={"out": ([64, 96], torch.float32)}, runtime={"tiles": 6}
    )
    assert out.cpu().shape == (64, 96)


@pytest.mark.hw
@pytest.mark.parametrize("flag", [1, 0])
def test_units_agree_with_torch(native, flag):
    import kohakutt.dsl.metal as km

    dev = native.device(0)
    cols, rows = native.compute_grid()
    launcher = km.Launcher(units_program(km.Machine(cols, rows)))
    for seed, row_tiles in enumerate([3, 40, 7]):
        g_ = torch.Generator().manual_seed(seed)
        x = torch.randn(row_tiles * 32, 64, generator=g_).bfloat16()
        w = (torch.randn(64, 64, generator=g_) / 8).bfloat16()
        g = (torch.rand(32, 64, generator=g_) + 0.5).bfloat16()
        native.synchronize()
        before = native.counters()
        out, acc = run(launcher, x.to(dev), w.to(dev), g.to(dev), flag, 0.25)
        out, acc = out.cpu().float(), acc.cpu()
        grown = native.counters()["kernel_compiles"] - before["kernel_compiles"]
        assert seed == 0 or grown == 0, row_tiles
        want_out, want_acc = _reference(x, w, g, flag, 0.25)
        assert torch.allclose(out, want_out, rtol=2e-2, atol=2e-2), row_tiles
        # acc is summed from the fp32 registers, before out is rounded to bf16.
        assert torch.allclose(acc, want_acc, rtol=2e-2, atol=3e-2), row_tiles
