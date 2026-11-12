"""Regression: with KTT_CUSTOM_KERNELS=0 (read once per process: run this file under
it), index_copy_ runs the selection route only (no own program, no private copy) and
still equals PyTorch."""

import os

import pytest
import torch

import kohakutt


@pytest.mark.hw
@pytest.mark.skipif(
    os.environ.get("KTT_CUSTOM_KERNELS") != "0", reason="needs KTT_CUSTOM_KERNELS=0"
)
def test_index_copy_without_own_kernels(native):
    g = torch.Generator().manual_seed(0)
    src = torch.randn(1, 2, 1, 64, generator=g).bfloat16()
    s = src.to("tt")
    # A first tensor warms the selection route's own caches.
    warm = torch.randn(1, 2, 60, 64, generator=g).bfloat16().to("tt")
    warm.index_copy_(2, torch.tensor([1]).to("tt"), s)
    ref = torch.randn(1, 2, 60, 64, generator=g).bfloat16()
    cache = ref.to("tt")
    calls = []
    for pos in (5, 59):
        j = torch.tensor([pos]).to("tt")
        kohakutt.synchronize()
        before = kohakutt.counters()
        cache.index_copy_(2, j, s)
        kohakutt.synchronize()
        after = kohakutt.counters()
        calls.append(
            (after["programs"] - before["programs"], after["ops"] - before["ops"])
        )
        ref.index_copy_(2, torch.tensor([pos]), src)
    # No own program, and the first write queues no extra private copy.
    assert [c[0] for c in calls] == [0, 0]
    assert calls[0][1] == calls[1][1]
    assert torch.equal(cache.cpu(), ref)
