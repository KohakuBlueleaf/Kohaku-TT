"""Greedy decoding with a static KV cache, a language model from transformers on a chip
(eager, or torch.compile with --compile): tokens per second, text against the CPU.
    python lm_decode.py --model Qwen/Qwen3-1.7B --new 32 --compile
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

import kohakutt

PROMPT = (
    "The Tenstorrent Blackhole card is an accelerator. In this document we describe"
)


def greedy(model, ids, new, dev):
    """`new` greedy tokens after `ids`; a StaticCache keeps every step's shapes fixed.
    Returns (tokens, prefill seconds, per-step seconds)."""
    n = ids.shape[1]
    cache = StaticCache(config=model.config, max_cache_len=n + new)
    tokens, steps = [], []
    x, pos = ids, torch.arange(n)
    for i in range(new):
        start = time.perf_counter()
        logits = model(
            input_ids=x.to(dev),
            past_key_values=cache,
            position_ids=pos[None].to(dev),
            cache_position=pos.to(dev),
            use_cache=True,
        ).logits
        x = logits[:, -1:].argmax(-1).cpu()  # the read synchronizes the step
        steps.append(time.perf_counter() - start)
        tokens.append(int(x))
        pos = torch.tensor([n + i])
    return tokens, steps[0], steps[1:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--new", type=int, default=32, help="tokens to generate")
    parser.add_argument(
        "--compile", action="store_true", help="torch.compile the model"
    )
    args = parser.parse_args()

    dev = kohakutt.device(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model = model.eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        ref, _, _ = greedy(model, ids, args.new, "cpu")

    model = model.to(dev)
    run = torch.compile(model, backend="ktt_tt") if args.compile else model
    with torch.no_grad():
        tokens, prefill, steps = greedy(run, ids, args.new, dev)
    steady = sorted(steps[len(steps) // 4 :])  # skip the first steps' program builds
    median = steady[len(steady) // 2]
    same = next((i for i, (a, b) in enumerate(zip(tokens, ref)) if a != b), len(ref))
    mode = "compiled" if args.compile else "eager"
    print(
        f"{mode}: prefill {prefill:.2f} s ({ids.shape[1]} tokens), first step "
        f"{steps[0]:.2f} s, then {median * 1e3:.1f} ms/token ({1 / median:.1f} tok/s)"
    )
    print(f"text equal to the CPU for {same}/{len(ref)} tokens")
    print("tt: ", repr(tok.decode(tokens)))
    print("cpu:", repr(tok.decode(ref)))
    kohakutt.close()


if __name__ == "__main__":
    main()
