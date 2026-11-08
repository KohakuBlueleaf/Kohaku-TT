"""Regression: the opt-in CPU fallback runs a missing op (igamma) on CPU and counts it;
KTT_CPU_FALLBACK is set for this test only."""

import pytest
import torch


@pytest.mark.hw
def test_cpu_fallback_opt_in(native, monkeypatch):
    a = torch.rand(8, 16) * 4 + 0.1
    x = torch.rand(8, 16) * 4
    with pytest.raises(NotImplementedError, match=native.UNSUPPORTED):
        torch.igamma(a.to("tt"), x.to("tt"))

    monkeypatch.setenv("KTT_CPU_FALLBACK", "1")
    monkeypatch.setenv("KTT_FALLBACK_WARN", "0")
    native.reset_fallback_counts()
    got = torch.igamma(a.to("tt"), x.to("tt")).cpu()
    assert torch.equal(got, torch.igamma(a, x))
    assert native.fallback_counts() == {"aten::igamma.out": 1}
