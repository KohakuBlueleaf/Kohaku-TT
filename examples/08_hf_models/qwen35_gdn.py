"""Qwen3.5 / 3.8 (GDN series): the gated delta net layers' single-token decode step on
one metal-level kernel (examples/kernels/dsl_metal/gdn_decode_km.py), the rest of the
layer as transformers runs it. Prefill is unchanged; KTT_GDN_DECODE=0 turns it off.
    python qwen35_gdn.py --model Qwen/Qwen3.5-4B --prompt 16 --new 16"""

import argparse
import importlib.util
import os
import time
import types
from pathlib import Path

import torch
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3_5 import modeling_qwen3_5 as m35

import kohakutt

KERNEL = Path(__file__).resolve().parents[1] / "kernels/dsl_metal/gdn_decode_km.py"


def _kernel():
    """The kernel's torch op (registered on the first load of its example)."""
    if not hasattr(torch.ops.kohakutt_example, "gdn_decode_km"):
        spec = importlib.util.spec_from_file_location("gdn_decode_km", KERNEL)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
    return torch.ops.kohakutt_example.gdn_decode_km


def enabled():
    """False when KTT_GDN_DECODE=0 (read per call)."""
    return os.environ.get("KTT_GDN_DECODE", "1") != "0"


def gdn_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    """Qwen3_5GatedDeltaNet.forward with the cached decode step on the kernel: the
    recurrent state is updated in place (the cache's own tensor); otherwise the
    transformers forward."""
    n, seq_len, _ = hidden_states.shape
    layer = self.layer_idx
    decode = (
        enabled()
        and seq_len == 1
        and cache_params is not None
        and not cache_params.is_recording_past(layer)
        and hidden_states.device.type == "tt"
    )
    conv_state = cache_params.get_conv_state(layer) if decode else None
    state = cache_params.get_recurrent_state(layer) if decode else None
    if conv_state is None or state is None or state.dtype != torch.float32:
        return self._ktt_forward(hidden_states, cache_params, attention_mask, **kwargs)
    hidden_states = m35.apply_mask_to_padding_states(hidden_states, attention_mask)
    mixed_qkv = m35.causal_conv1d_update(
        self.in_proj_qkv(hidden_states).transpose(1, 2),
        conv_state,
        self.conv1d.weight.squeeze(1),
        self.conv1d.bias,
        self.activation,
    ).transpose(1, 2)
    z = self.in_proj_z(hidden_states).reshape(-1, self.head_v_dim)
    b, a = self.in_proj_b(hidden_states), self.in_proj_a(hidden_states)
    o = _kernel()(state, mixed_qkv, a, b, self.A_log, self.dt_bias)
    o = self.norm(o.reshape(-1, self.head_v_dim), z).reshape(n, 1, -1)
    return self.out_proj(o)


def install(model):
    """Routes every GDN layer's cached decode step of `model` to the kernel; returns
    the number of layers."""
    count = 0
    for module in model.modules():
        if isinstance(module, m35.Qwen3_5GatedDeltaNet):
            module._ktt_forward = module.forward
            module.forward = types.MethodType(gdn_forward, module)
            count += 1
    _kernel()
    return count


def load(name, dev):
    """The text model of checkpoint `name` (a Qwen3.5 image-text checkpoint) on
    `dev`, and its config."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    config = AutoConfig.from_pretrained(name)
    config = getattr(config, "text_config", config)
    model = Qwen3_5ForCausalLM.from_pretrained(
        name,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        key_mapping={r"^model\.language_model\.": "model."},
    )
    return model.eval().to(dev), config


def greedy(model, config, ids, new, dev):
    """`new` greedy tokens after `ids` with a DynamicCache: (tokens, ms per step)."""
    cache = DynamicCache(config=config)
    tokens, steps = [], []
    x, pos = ids, torch.arange(ids.shape[1])
    with torch.no_grad():
        for _ in range(new):
            start = time.perf_counter()
            logits = model(
                input_ids=x.to(dev),
                past_key_values=cache,
                position_ids=pos[None].to(dev),
                cache_position=pos.to(dev),
                use_cache=True,
            ).logits
            x = logits[:, -1:].argmax(-1).cpu()
            steps.append((time.perf_counter() - start) * 1e3)
            tokens.append(int(x))
            pos = pos[-1:] + 1
    return tokens, steps


def main():
    """Greedy decoding with and without the kernel; prints both token lists and the
    ms per decode step."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--prompt", type=int, default=16)
    parser.add_argument("--new", type=int, default=16)
    args = parser.parse_args()
    dev = kohakutt.device(0)
    try:
        model, config = load(args.model, dev)
        ids = torch.randint(
            1000, 100000, (1, args.prompt), generator=torch.Generator().manual_seed(0)
        )
        print(f"GDN layers on the kernel: {install(model)}")
        for flag in ("0", "1"):
            os.environ["KTT_GDN_DECODE"] = flag
            tokens, steps = greedy(model, config, ids, args.new, dev)
            rest = sorted(steps[2:]) or steps
            label = "kernel" if flag == "1" else "transformers"
            print(f"  {label}: {rest[len(rest) // 2]:.1f} ms per step, tokens {tokens}")
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
