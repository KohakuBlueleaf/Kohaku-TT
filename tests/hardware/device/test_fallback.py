"""Regression: the opt-in CPU fallback runs a missing op (special.bessel_j0) on CPU
and counts it; KTT_CPU_FALLBACK is set for this test only."""

import pytest
import torch


@pytest.mark.hw
def test_cpu_fallback_opt_in(native, monkeypatch):
    a = torch.randn(8, 16) * 4
    with pytest.raises(NotImplementedError, match=native.UNSUPPORTED):
        torch.special.bessel_j0(a.to("tt"))

    monkeypatch.setenv("KTT_CPU_FALLBACK", "1")
    monkeypatch.setenv("KTT_FALLBACK_WARN", "0")
    native.reset_fallback_counts()
    got = torch.special.bessel_j0(a.to("tt")).cpu()
    assert torch.equal(got, torch.special.bessel_j0(a))
    assert native.fallback_counts() == {"aten::special_bessel_j0.out": 1}
