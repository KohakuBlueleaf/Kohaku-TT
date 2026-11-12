"""Slow regressions (KTT_SLOW=1, local Gemma-4 E2B weights): teacher-forced static-cache
decode and a 77-token prefill compiled by ktt_tt stay as close to a CPU reference as
eager (decode: Python-int positions; prefill: masked attention at head dim 512)."""

import os

import pytest
import torch
import torch._dynamo

from kohakutt.torch_backend import compiler as tt_compile

MODEL = "google/gemma-4-E2B"
PROMPT, STEPS = 77, 24


def _decode(model, dev, ids, tokens=None):
    """Last-token fp32 logits [STEPS, vocab] of a decode fed `tokens` (greedy when
    None)."""
    from transformers import StaticCache

    cache = StaticCache(config=model.config, max_cache_len=PROMPT + STEPS + 1)
    x, pos, out = ids, torch.arange(PROMPT), []
    with torch.no_grad():
        for i in range(STEPS):
            logits = model(
                input_ids=x.to(dev),
                past_key_values=cache,
                position_ids=pos[None].to(dev),
                cache_position=pos.to(dev),
                use_cache=True,
            ).logits[0, -1]
            out.append(logits.float().cpu())
            token = out[-1].argmax() if tokens is None else tokens[i]
            x, pos = token.view(1, 1), torch.tensor([PROMPT + i])
    return torch.stack(out)


def _median_error(logits, ref):
    errors = (logits - ref).norm(dim=-1) / ref.norm(dim=-1)
    return errors.median().item()


@pytest.mark.hw
@pytest.mark.skipif(os.environ.get("KTT_SLOW") != "1", reason="slow: set KTT_SLOW=1")
def test_compiled_gemma_decode_matches_eager_error(native):
    transformers = pytest.importorskip("transformers")
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, local_files_only=True
        ).eval()
    except OSError:
        pytest.skip(f"{MODEL} is not in the local cache")
    ids = torch.randint(
        0, 20000, (1, PROMPT), generator=torch.Generator().manual_seed(0)
    )
    ref = _decode(model, "cpu", ids)
    tokens = ref.argmax(-1)
    model = model.to("tt")
    eager = _median_error(_decode(model, "tt", ids, tokens), ref)
    tt_compile.register()
    torch._dynamo.reset()
    compiled = torch.compile(model, backend="ktt_tt")
    error = _median_error(_decode(compiled, "tt", ids, tokens), ref)
    tt_compile.release()
    assert error <= 2 * eager, (error, eager)


@pytest.mark.hw
@pytest.mark.skipif(os.environ.get("KTT_SLOW") != "1", reason="slow: set KTT_SLOW=1")
def test_compiled_gemma_prefill_matches_eager_error(native):
    transformers = pytest.importorskip("transformers")
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float32, local_files_only=True
        ).eval()
    except OSError:
        pytest.skip(f"{MODEL} is not in the local cache")
    ids = torch.randint(
        0, 20000, (1, PROMPT), generator=torch.Generator().manual_seed(0)
    )
    with torch.no_grad():
        ref = model(input_ids=ids, use_cache=False).logits.double()
    model = model.to(torch.bfloat16).to("tt")

    def error(net):
        with torch.no_grad():
            out = net(input_ids=ids.to("tt"), use_cache=False).logits.cpu().double()
        return ((out - ref).norm() / ref.norm()).item()

    eager = error(model)
    tt_compile.register()
    torch._dynamo.reset()
    compiled = torch.compile(model, backend="ktt_tt")
    errors = [error(compiled) for _ in range(4)]  # eager twice, capture, replay
    tt_compile.release()
    assert max(errors) <= 1.5 * eager, (errors, eager)
