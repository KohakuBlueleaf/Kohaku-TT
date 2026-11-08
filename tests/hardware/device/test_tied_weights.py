"""Regression: Module.to("tt") keeps tied parameters tied, so a tied embedding / head
weight receives both gradient contributions."""

import pytest
import torch
import torch.nn.functional as F


class TiedLM(torch.nn.Module):
    """Embedding whose weight is also the output head's weight."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(32, 8)
        self.head = torch.nn.Linear(8, 32, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, tokens):
        return self.head(self.embed(tokens))


@pytest.mark.hw
def test_to_tt_keeps_ties(native):
    torch.manual_seed(0)
    model = TiedLM().bfloat16()
    tokens = torch.tensor([[1, 2, 3, 2]])
    model(tokens).float().square().mean().backward()
    want = model.embed.weight.grad.clone()

    model.zero_grad()
    model.to("tt")
    assert model.head.weight is model.embed.weight
    model(tokens.to("tt")).float().square().mean().backward()
    got = model.embed.weight.grad.cpu()
    # Rows of tokens absent from the input get the head's contribution only.
    assert got[10].abs().sum() > 0
    cosine = F.cosine_similarity(got.float().flatten(), want.float().flatten(), dim=0)
    assert cosine > 0.999
