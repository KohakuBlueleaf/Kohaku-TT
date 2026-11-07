"""Experiment: small HF Llama / Qwen2 (random weights) on tt vs CPU, with the opt-in
CPU fallback counting every op still missing a tt kernel.

Greedy generate (tokens equal?), forward logits statistics, and one training step.
    KTT_CPU_FALLBACK=1 KTT_FALLBACK_WARN=0 python -m kohakutt.bench.models.hf_check
"""

import argparse
import json

import torch
from transformers import LlamaForCausalLM, Qwen2ForCausalLM

import kohakutt
from kohakutt.bench.models.census import hf_config
from kohakutt.core.stats import diff_stats


def build(kind):
    """Seeded bf16 random-weight model of `kind` on CPU."""
    torch.manual_seed(0)
    cls = LlamaForCausalLM if kind == "llama" else Qwen2ForCausalLM
    return cls(hf_config(kind)).bfloat16().eval()


def check(kind):
    """Record for `kind`: logits stats, generate equality, train step, fallbacks."""
    kohakutt.reset_fallback_counts()
    cpu = build(kind)
    tt = build(kind).to("tt")
    tokens = torch.randint(0, 512, (2, 32), generator=torch.Generator().manual_seed(1))
    record = {"model": kind}
    with torch.no_grad():
        ref = cpu(tokens).logits.double()
        got = tt(tokens.to("tt")).logits.cpu().double()
    record["logits"] = diff_stats(got, ref)
    with torch.no_grad():
        g_cpu = cpu.generate(tokens[:1, :8], max_new_tokens=8, do_sample=False)
        g_tt = tt.generate(tokens[:1, :8].to("tt"), max_new_tokens=8, do_sample=False)
    record["generate_equal"] = bool(torch.equal(g_cpu, g_tt.cpu()))

    # One training step on tt.
    tt.train()
    opt = torch.optim.AdamW(tt.parameters(), lr=1e-3)
    t = tokens.to("tt")
    loss = torch.nn.functional.cross_entropy(tt(t).logits.flatten(0, 1), t.flatten())
    loss.backward()
    opt.step()
    record["train_loss"] = round(loss.item(), 4)
    record["fallbacks"] = kohakutt.fallback_counts()
    return record


def main():
    """Runs both models on device 0 and prints one record each."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["llama", "qwen2"])
    args = parser.parse_args()
    kohakutt.open(0)
    try:
        for kind in args.models:
            try:
                print(json.dumps(check(kind)), flush=True)
            except Exception as error:  # noqa: BLE001  (reported per model)
                message = str(error).splitlines()[0][:300]
                print(json.dumps({"model": kind, "error": message}), flush=True)
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
