"""greedy_decode on a left-padded batch with an attention mask, the forward compiled by
ktt_tt: the same tokens as HF's eager generate with that mask on tt."""

import pytest
import torch
import torch._dynamo
from transformers import Qwen2ForCausalLM, StaticCache

import kohakutt
from kohakutt.bench.models.census import hf_config
from kohakutt.torch_backend import compiler as tt_compile


@pytest.mark.hw
def test_left_padded_batch_matches_hf(native):
    tt_compile.register()
    torch._dynamo.reset()
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(hf_config("qwen")).bfloat16().eval().to("tt")
    ids = torch.randint(1, 512, (2, 8), generator=torch.Generator().manual_seed(1))
    mask = torch.ones_like(ids)
    mask[0, :3] = 0
    ids = ids.masked_fill(mask == 0, 3)
    with torch.no_grad():
        want = model.generate(
            ids.to("tt"),
            attention_mask=mask.to("tt"),
            max_new_tokens=6,
            do_sample=False,
            pad_token_id=3,
            cache_implementation="static",
        )[:, 8:].cpu()
    cache = StaticCache(config=model.config, max_cache_len=16)
    forward = torch.compile(model, backend="ktt_tt", dynamic=False)
    for _ in range(4):  # eager calls, capture, replays
        got = kohakutt.greedy_decode(
            forward, ids.to("tt"), cache, 6, pad_token_id=3, attention_mask=mask
        )
        assert torch.equal(got, want)
    tt_compile.release()
    assert native.fallback_counts() == {}
