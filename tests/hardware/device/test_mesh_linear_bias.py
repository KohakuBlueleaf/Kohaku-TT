"""Regression: on the 1x2 mesh a Linear's bias runs inside each chip's matmul, costing
no op of its own, when it is placed as the output's last dim (a column-split weight
with its bias shard, a replicated Linear on split rows) or the output is a partial sum
(row-split weight: half the bias per chip); otherwise it is added once after the
matmul."""

import pytest
import torch


def _ops(kohakutt):
    torch.tt.synchronize()
    return kohakutt.counters()["ops"]


@pytest.mark.hw
@pytest.mark.dual
def test_mesh_linear_bias(native_module):
    kohakutt = native_module
    rows, narrow, wide = 128, 512, 2048
    # name: (in, out, marks of the Linear, placement of x, bias inside the matmul:
    # None leaves the op count unchecked)
    cases = {
        "column-split": (narrow, wide, kohakutt.colwise, kohakutt.replicate, True),
        "rows split": (
            narrow,
            wide,
            kohakutt.replicate,
            lambda x: kohakutt.shard(x, 0),
            True,
        ),
        "column-split, bias replicated": (
            narrow,
            wide,
            lambda m: kohakutt.place(
                m, weight="S(0)", **({} if m.bias is None else {"bias": "R"})
            ),
            kohakutt.replicate,
            None,
        ),
        "row-split": (wide, narrow, kohakutt.rowwise, kohakutt.replicate, True),
        "row-split, x split on rows": (
            wide,
            narrow,
            kohakutt.rowwise,
            lambda x: kohakutt.shard(x, 0),
            None,
        ),
    }
    got = {}
    mesh = kohakutt.mesh([0, 1], shape=(1, 2))
    try:
        for seed, (name, (k, n, mark, place_x, inside)) in enumerate(cases.items()):
            torch.manual_seed(seed)
            plain = torch.nn.Linear(k, n, bias=False).to(torch.bfloat16)
            biased = torch.nn.Linear(k, n).to(torch.bfloat16)
            with torch.no_grad():
                biased.weight.copy_(plain.weight)
                biased.bias.copy_(4 * torch.randn(n))
            x = torch.randn(rows, k).bfloat16()
            want = (
                x.double() @ biased.weight.double().t() + biased.bias.double()
            ).detach()
            plain, biased = mark(plain).to(mesh), mark(biased).to(mesh)
            xd = place_x(x.to(mesh))
            with torch.no_grad():
                for _ in range(2):
                    plain(xd), biased(xd)
                start = _ops(kohakutt)
                plain(xd)
                middle = _ops(kohakutt)
                y = biased(xd)
                end = _ops(kohakutt)
            got[name] = (inside, middle - start, end - middle, y.cpu().double(), want)
    finally:
        kohakutt.close()
    for name, (inside, plain_ops, biased_ops, y, want) in got.items():
        error = (y - want).abs().max().item()
        assert error <= 0.01 * want.abs().max().item(), (name, error)
        if inside:
            assert biased_ops == plain_ops, (name, plain_ops, biased_ops)
        elif inside is False:
            assert biased_ops > plain_ops, (name, plain_ops, biased_ops)
