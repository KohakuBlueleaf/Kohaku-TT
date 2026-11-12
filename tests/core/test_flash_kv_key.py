"""ktt_flash's descriptor key leaves K / V's key count out with key lengths, so one
cached program serves every KV length (the reader takes the count per call)."""

import pytest
import torch

import kohakutt

pytestmark = pytest.mark.skipif(
    not kohakutt.AVAILABLE, reason="needs the native build (torch.ops.ktt)"
)


def key(sk, lengths, b=2, hk=8, d=64):
    return torch.ops.ktt._debug_flash_kv_key([b, hk, sk, d], lengths)


def test_lengths_share_one_key():
    assert key(100, True) == key(107, True) == key(4096, True)


def test_other_dims_still_count():
    assert key(100, True) != key(100, True, hk=2)
    assert key(100, True) != key(100, True, d=128)
    assert key(100, True) != key(100, True, b=1)


def test_without_lengths_the_key_count_counts():
    assert key(100, False) != key(107, False)
    assert key(100, False) != key(100, True)
