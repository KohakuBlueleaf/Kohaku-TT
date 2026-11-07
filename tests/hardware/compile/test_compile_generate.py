"""Regression: HF static-cache greedy generate with the forward compiled by ktt_tt
equals eager generate on tt (the trace updates the KV cache in place)."""

import pytest
import torch
import torch._dynamo
from transformers import Qwen2ForCausalLM

from kohakutt.bench.models.census import hf_config
from kohakutt.torch_backend import compiler as tt_compile


@pytest.mark.hw
@pytest.mark.xfail(
    reason="P-1: generate's per-step allocations (growing input_ids) made after the "
    "decode graph's capture can be overwritten by its replays",
    strict=False,
)
def test_compiled_static_cache_generate_matches_eager(native):
    tt_compile.register()
    torch._dynamo.reset()
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(hf_config("qwen")).bfloat16().eval().to("tt")
    model.generation_config.cache_implementation = "static"
    ids = torch.randint(0, 512, (1, 8), generator=torch.Generator().manual_seed(1))
    ids = ids.to("tt")
    with torch.no_grad():
        eager = model.generate(ids, max_new_tokens=12, do_sample=False).cpu()
        model.forward = torch.compile(model.forward, backend="ktt_tt", dynamic=False)
        # Twice: the second generate makes a new cache (copied into the trace's).
        first = model.generate(ids, max_new_tokens=12, do_sample=False).cpu()
        second = model.generate(ids, max_new_tokens=12, do_sample=False).cpu()
    tt_compile.release()
    assert torch.equal(first, eager)
    assert torch.equal(second, eager)
    assert native.fallback_counts() == {}
