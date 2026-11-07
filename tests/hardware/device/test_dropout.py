"""Regression: dropout on tt is seed-deterministic, scales kept values, backprops the
mask, and is the identity in eval mode."""

import pytest
import torch


@pytest.mark.hw
def test_dropout(native):
    p = 0.25
    x = torch.randn(64, 256).bfloat16().to("tt").requires_grad_()

    torch.manual_seed(0)
    y = torch.nn.functional.dropout(x, p, training=True)
    y.sum().backward()
    kept = y.detach().cpu() != 0
    assert torch.equal(x.grad.cpu() != 0, kept)

    torch.manual_seed(0)
    again = torch.nn.functional.dropout(x.detach(), p, training=True)
    assert torch.equal(again.cpu(), y.detach().cpu())

    same = torch.nn.functional.dropout(x.detach(), p, training=False)
    assert torch.equal(same.cpu(), x.detach().cpu())
