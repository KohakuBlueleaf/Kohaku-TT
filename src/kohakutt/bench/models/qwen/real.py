"""Experiment: real Qwen2.5-0.5B (bf16) on tt vs CPU: greedy decode and per-step time.

Reports token agreement with CPU over a greedy continuation, first-divergence step,
next-token logits statistics, decode ms per token (eager tt), and CPU-fallback calls.
    python -m kohakutt.bench.models.qwen.real [--tokens 32]  (HF_HOME set)
"""

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import kohakutt
from kohakutt.core.stats import diff_stats

MODEL = "Qwen/Qwen2.5-0.5B"
PROMPT = "The Tenstorrent Blackhole chip is"


def main():
    """Decodes greedily on CPU and on tt and prints one comparison record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--cache", choices=["dynamic", "static"], default="dynamic")
    args = parser.parse_args()
    gen_kwargs = {"max_new_tokens": args.tokens, "do_sample": False}
    if args.cache == "static":
        gen_kwargs["cache_implementation"] = "static"

    tok = AutoTokenizer.from_pretrained(MODEL)
    cpu = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        ref_logits = cpu(ids).logits[0, -1].double()
        ref = cpu.generate(ids, max_new_tokens=args.tokens, do_sample=False)

    kohakutt.open(0)
    try:
        kohakutt.reset_fallback_counts()
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
        model = model.to("tt")
        with torch.no_grad():
            logits = model(ids.to("tt")).logits[0, -1].cpu().double()
            model.generate(ids.to("tt"), **{**gen_kwargs, "max_new_tokens": 2})
            torch.tt.synchronize()
            start = time.perf_counter()
            out = model.generate(ids.to("tt"), **gen_kwargs)
            torch.tt.synchronize()
            ms = (time.perf_counter() - start) * 1e3 / args.tokens

            # Same decode again: every program is cached now (steady-state eager).
            ops_before = kohakutt.counters()["ops"]
            start = time.perf_counter()
            model.generate(ids.to("tt"), **gen_kwargs)
            torch.tt.synchronize()
            steady_ms = (time.perf_counter() - start) * 1e3 / args.tokens
            ops_per_token = (kohakutt.counters()["ops"] - ops_before) / args.tokens
        out = out.cpu()
        fallbacks = kohakutt.fallback_counts()
    finally:
        kohakutt.close()

    new_ref, new_tt = ref[0, ids.shape[1] :], out[0, ids.shape[1] :]
    same = (new_ref == new_tt).tolist()
    first_diff = same.index(False) if False in same else None
    print(
        json.dumps(
            {
                "model": MODEL,
                "new_tokens": args.tokens,
                "tokens_equal_until": first_diff,
                "agreement": round(sum(same) / len(same), 3),
                "next_token_logits": diff_stats(logits, ref_logits),
                "top1_equal": int(logits.argmax()) == int(ref_logits.argmax()),
                "decode_ms_per_token": round(ms, 1),
                "steady_decode_ms_per_token": round(steady_ms, 1),
                "device_ops_per_token": round(ops_per_token, 1),
                "cpu_text": tok.decode(new_ref),
                "tt_text": tok.decode(new_tt),
                "fallbacks": fallbacks,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
