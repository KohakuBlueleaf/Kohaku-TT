"""The compile graph pass on CPU: per HF architecture, the RMSNorm and RoPE chains of a
one-layer decode step are rewritten (Dynamo and AOT graphs) and logits stay close."""

import pytest
import torch
import torch._dynamo
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd

import kohakutt  # noqa: F401  (registers torch.ops.ktt)
from kohakutt.torch_backend.chains import fuse_chains

transformers = pytest.importorskip("transformers")

SMALL = dict(
    vocab_size=128,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    max_position_embeddings=256,
)
# Norms of one layer plus the final one; RoPE on q and k.
FAMILIES = {
    "Qwen2Config": {"rms_norm": 3, "rotary": 2},
    "Qwen3Config": {"rms_norm": 5, "rotary": 2},
    "GemmaConfig": {"rms_norm": 3, "rotary": 2},
    "Gemma3TextConfig": {"rms_norm": 7, "rotary": 2},
    "LlamaConfig": {"rms_norm": 3, "rotary": 2},
}


def _decode(family, dtype, level, monkeypatch, mode="all"):
    monkeypatch.setenv("KTT_FUSE_CHAINS", mode)
    counts = []

    def fused(gm, example_inputs):
        got = fuse_chains(gm).ktt_chains
        counts.append({k: got[k] for k in ("rms_norm", "rotary")})
        return make_boxed_func(gm.forward) if level == "aot" else gm.forward

    backend = aot_autograd(fw_compiler=fused) if level == "aot" else fused
    torch._dynamo.reset()
    cfg = getattr(transformers, family)(**SMALL)
    torch.manual_seed(0)
    net = transformers.AutoModelForCausalLM.from_config(cfg, dtype=dtype).eval()
    ids = torch.randint(0, 100, (1, 4), generator=torch.Generator().manual_seed(1))
    logits = []
    for run in (net, torch.compile(net, backend=backend)):
        cache = transformers.StaticCache(config=cfg, max_cache_len=16)
        pos = torch.arange(4)
        step = dict(past_key_values=cache, use_cache=True)
        with torch.no_grad():
            net(input_ids=ids, position_ids=pos[None], cache_position=pos, **step)
            out = run(
                input_ids=ids[:, :1],
                position_ids=torch.tensor([[4]]),
                cache_position=torch.tensor([4]),
                **step,
            )
        logits.append(out.logits)
    return counts, logits


@pytest.mark.parametrize("level", ["dynamo", "aot"])
@pytest.mark.parametrize("family", list(FAMILIES))
def test_chains_rewritten(monkeypatch, family, level):
    counts, (eager, compiled) = _decode(family, torch.float32, level, monkeypatch)
    assert counts == [FAMILIES[family]]
    torch.testing.assert_close(compiled, eager, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("family", ["Qwen2Config", "Gemma3TextConfig"])
def test_bf16_casts_absorbed(monkeypatch, family):
    counts, (eager, compiled) = _decode(family, torch.bfloat16, "dynamo", monkeypatch)
    assert counts == [FAMILIES[family]]
    assert compiled.dtype == eager.dtype == torch.bfloat16
    torch.testing.assert_close(compiled, eager, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "mode, want", [("0", (0, 0)), ("norm", (3, 0)), ("norm,mask,gqa", (3, 0))]
)
def test_knob(monkeypatch, mode, want):
    counts, _ = _decode("Qwen2Config", torch.float32, "dynamo", monkeypatch, mode)
    assert [(c["rms_norm"], c["rotary"]) for c in counts] == [want]
