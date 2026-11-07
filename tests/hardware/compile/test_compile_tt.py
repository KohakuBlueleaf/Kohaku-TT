"""Regression: torch.compile on tt: compiled inference and training match eager."""

import pytest
import torch._dynamo

from kohakutt.bench.capture import compile_tt_check
from kohakutt.torch_backend import compiler as tt_compile


@pytest.mark.hw
def test_compiled_inference_matches_eager(native):
    tt_compile.register()
    torch._dynamo.reset()
    assert compile_tt_check.inference()["eq_eager"]
    tt_compile.release()


@pytest.mark.hw
@pytest.mark.parametrize("kind", ["mlp", "cnn"])
def test_compiled_training_matches_eager(native, kind):
    tt_compile.register()
    torch._dynamo.reset()
    eager, _ = compile_tt_check.train(None, 4, kind)
    compiled, _ = compile_tt_check.train("ktt_tt_aot", 4, kind)
    tt_compile.release()
    assert compiled == eager
