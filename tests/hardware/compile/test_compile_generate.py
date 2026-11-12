"""Regression: HF static-cache greedy generate with the forward compiled by ktt_tt
equals eager generate on tt (the trace updates the KV cache in place); greedy_decode
feeds tokens back on the device: no upload per token, one download per read_every."""

import pytest
import torch
import torch._dynamo
from transformers import Qwen2ForCausalLM, StaticCache

import kohakutt
from kohakutt.bench.models.census import hf_config
from kohakutt.torch_backend import compiler as tt_compile


@pytest.mark.hw
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


@pytest.mark.hw
def test_greedy_decode_feeds_tokens_on_the_device(native):
    tt_compile.register()
    torch._dynamo.reset()
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(hf_config("qwen")).bfloat16().eval().to("tt")
    ids = torch.randint(0, 512, (1, 8), generator=torch.Generator().manual_seed(1))
    ids = ids.to("tt")
    with torch.no_grad():
        eager = model.generate(ids, max_new_tokens=12, do_sample=False)[:, 8:].cpu()
    cache = StaticCache(config=model.config, max_cache_len=40)
    forward = torch.compile(model, backend="ktt_tt", dynamic=False)
    for _ in range(3):  # two eager calls per graph, then its trace
        got = kohakutt.greedy_decode(forward, ids, cache, 12, read_every=4)
        assert torch.equal(got, eager)
    torch.tt.synchronize()
    before = kohakutt.counters()
    got = kohakutt.greedy_decode(forward, ids, cache, 12, read_every=4)
    after = kohakutt.counters()
    assert torch.equal(got, eager)
    assert after["uploads"] - before["uploads"] == 2  # the prompt's positions
    assert after["downloads"] - before["downloads"] == 3
    # an end token mid-way: the same tokens and padding as HF's generate
    end = int(eager[0, 4])
    with torch.no_grad():
        want = model.generate(
            ids, max_new_tokens=12, do_sample=False, eos_token_id=end, pad_token_id=0
        )[:, 8:].cpu()
    got = kohakutt.greedy_decode(
        forward, ids, cache, 12, eos_token_id=end, pad_token_id=0, read_every=4
    )
    assert torch.equal(got, want)
    tt_compile.release()
    assert native.fallback_counts() == {}
