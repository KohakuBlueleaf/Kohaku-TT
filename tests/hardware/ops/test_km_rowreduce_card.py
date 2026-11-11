"""Row max and row sum of register slots (tests/km_rowreduce.py) on the card, and
ktt_flash's step built on them (max, b = 639 - a m, P = flash_exp)."""

import pytest
import torch

from km_rowreduce import A, rowreduce_program

ITEMS = 4


def _launch(native, **kw):
    import kohakutt.dsl.metal as km

    dev = native.device(0)
    machine = km.Machine(*native.compute_grid())
    launcher = km.Launcher(rowreduce_program(machine, **kw))
    g = torch.Generator().manual_seed(14)
    s = torch.randn(32 * ITEMS, 96, generator=g) * 8
    prev = (torch.randn(32 * ITEMS, 1, generator=g) * 8).expand(-1, 32).contiguous()
    outputs = {"out": ([32 * ITEMS, 32], torch.float32)}
    if kw.get("flash"):
        outputs["p"] = ([32 * ITEMS, 96], torch.float32)
    got = launcher(
        {"s": s.to(dev), "prev": prev.to(dev)},
        outputs=outputs,
        runtime={"items": ITEMS},
    )
    return s, prev, [t.cpu() for t in got]


def _columns_equal(out):
    bits = out.view(torch.int32)
    return torch.equal(bits, bits[:, :1].expand_as(bits))


@pytest.mark.hw
def test_row_max_and_sum(native):
    s, prev, (out,) = _launch(native, op="max")
    want = torch.maximum(s.max(dim=1, keepdim=True).values, prev[:, :1])
    assert torch.equal(out, want.expand_as(out))
    s, prev, (out,) = _launch(native, op="sum")
    assert _columns_equal(out)
    want = s.double().sum(dim=1, keepdim=True) + prev[:, :1].double()
    torch.testing.assert_close(out[:, :1].double(), want, rtol=1e-5, atol=1e-4)
    s, prev, (out,) = _launch(native, op="sum", fresh=True)
    assert _columns_equal(out)
    want = s.double().sum(dim=1, keepdim=True)
    torch.testing.assert_close(out[:, :1].double(), want, rtol=1e-5, atol=1e-4)


@pytest.mark.hw
def test_flash_step_in_registers(native):
    s, prev, (b, probs) = _launch(native, op="max", flash=True)
    m = torch.maximum(s.max(dim=1, keepdim=True).values, prev[:, :1]).double()
    assert _columns_equal(b)
    torch.testing.assert_close(b[:, :1].double(), 639 - A * m, rtol=0, atol=1e-4)
    exact = torch.exp2(A * (s.double() - m))
    error = (probs.double() - exact).abs() / exact
    assert error[exact > 1e-30].max() < 1.8e-3
