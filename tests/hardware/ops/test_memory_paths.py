"""Regression: data-movement fast paths stay on them (program counts bounded, no host
transfers after warm-up) and stay bit-exact against CPU."""

import pytest
import torch
import torch.nn.functional as F

import kohakutt


def cost(fn):
    """(result, programs, host transfers) of fn() after one warm-up call."""
    fn()
    c0 = kohakutt.counters()
    out = fn()
    c1 = kohakutt.counters()
    xfer = c1["uploads"] + c1["downloads"] - c0["uploads"] - c0["downloads"]
    return out, c1["ops"] - c0["ops"], xfer


@pytest.mark.hw
def test_row_slice_views_do_not_flatten(native):
    x = torch.randn(512, 256).bfloat16()
    y = torch.randn(512, 256).bfloat16()
    xt, yt = x.to("tt"), y.to("tt")
    out, programs, xfer = cost(lambda: xt + yt[:1])
    assert programs <= 2 and xfer == 0
    assert torch.equal(out.cpu(), x + y[:1])
    out, programs, xfer = cost(lambda: xt[:256].repeat(2, 1))
    assert programs <= 2 and xfer == 0
    assert torch.equal(out.cpu(), x[:256].repeat(2, 1))


@pytest.mark.hw
def test_sliced_permuted_grad_view_resolves(native):
    # The ViT patch-embed gradient: [B, 1 + HW, C] base, CLS row skipped, read as
    # [B, C, H, W].
    base = torch.randn(2, 1 + 49, 64).bfloat16()
    view = base[:, 1:].transpose(1, 2).reshape(2, 64, 7, 7)
    tview = base.to("tt")[:, 1:].transpose(1, 2).reshape(2, 64, 7, 7)
    out, programs, xfer = cost(lambda: tview.contiguous())
    assert programs <= 4 and xfer == 0
    assert torch.equal(out.cpu(), view.contiguous())


@pytest.mark.hw
def test_index_select_embedding_path(native):
    table = torch.randn(1000, 64).bfloat16()
    idx = torch.randint(0, 1000, (96,))
    tt_table, tt_idx = table.to("tt"), idx.to("tt")
    out, programs, xfer = cost(lambda: tt_table.index_select(0, tt_idx))
    assert programs <= 2 and xfer == 0
    assert torch.equal(out.cpu(), table.index_select(0, idx))
    tt_table.mul_(2)  # a new version: the cached row-major copy is rebuilt
    assert torch.equal(F.embedding(tt_idx, tt_table).cpu(), F.embedding(idx, table * 2))


@pytest.mark.hw
def test_fused_qkv_head_split(native):
    b, t, h, d = 2, 64, 4, 32
    qkv = torch.randn(b, t, 3 * h * d).bfloat16()
    cpu = [x.transpose(1, 2) for x in qkv.view(b, t, 3, h, d).unbind(2)]
    tt = [x.transpose(1, 2) for x in qkv.to("tt").view(b, t, 3, h, d).unbind(2)]
    out, programs, xfer = cost(
        lambda: F.scaled_dot_product_attention(*tt, is_causal=True)
    )
    assert programs <= 2 and xfer == 0
    assert all(torch.equal(a.contiguous(), c.cpu()) for a, c in zip(cpu, tt))


@pytest.mark.hw
def test_patchify_conv_path(native):
    x = torch.randn(2, 3, 64, 64).bfloat16().to("tt").requires_grad_()
    w = (torch.randn(32, 3, 16, 16) * 0.05).bfloat16().to("tt").requires_grad_()
    out, programs, xfer = cost(lambda: F.conv2d(x, w, stride=16))
    assert out.shape == (2, 32, 4, 4) and programs <= 10 and xfer == 0
    grad = torch.randn(2, 16 + 1, 32).bfloat16().to("tt")[:, 1:].transpose(1, 2)
    grad = grad.reshape(2, 32, 4, 4)
    _, programs, xfer = cost(lambda: out.backward(grad, retain_graph=True))
    assert programs <= 30 and xfer == 0


@pytest.mark.hw
def test_fills_exact(native):
    g = torch.randn(64, 96).bfloat16().to("tt")
    g[0, 0] = float("nan")
    assert torch.equal(g.zero_().cpu(), torch.zeros(64, 96).bfloat16())
    assert torch.equal(g.fill_(3.5).cpu(), torch.full((64, 96), 3.5).bfloat16())
    _, programs, xfer = cost(lambda: g.zero_())
    assert programs == 1 and xfer == 0


@pytest.mark.hw
def test_fp32_nll_on_device(native):
    logits = torch.randn(256, 512).to("tt").requires_grad_()
    target = torch.randint(0, 512, (256,)).to("tt")

    def step():
        loss = F.cross_entropy(logits, target)
        loss.backward()
        return loss

    _, programs, xfer = cost(step)
    assert xfer == 0 and programs <= 40
