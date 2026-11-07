"""Regression: small HF Llama / Qwen2 (random weights) run on tt with no fallback."""

import pytest

from kohakutt.bench.models.hf_check import check


@pytest.mark.hw
@pytest.mark.parametrize("kind", ["llama", "qwen2"])
def test_hf_model_on_tt(native, kind):
    record = check(kind)
    assert record["generate_equal"]
    assert record["fallbacks"] == {}
    assert record["logits_pcc"] > 0.999
