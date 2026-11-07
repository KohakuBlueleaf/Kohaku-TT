"""Regression: HF architectures under torch.compile (ktt_tt_aot) equal eager on tt."""

import pytest

from kohakutt.bench.models import model_zoo
from kohakutt.torch_backend import compiler as tt_compile


@pytest.mark.hw
@pytest.mark.parametrize("kind", ["gpt2", "t5", "vit", "convnext", "whisper"])
def test_compiled_zoo_matches_eager(native, kind):
    tt_compile.register()
    record = model_zoo.compiled_vs_eager(kind)
    tt_compile.release()
    assert "error" not in record, record.get("error")
    assert record["logits_equal"]
    done, total = record["grads_equal"].split("/")
    assert done == total
    assert record["fallbacks"] == {}
