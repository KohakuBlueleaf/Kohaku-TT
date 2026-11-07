"""Experiment: real Qwen2.5-0.5B greedy generate on tt with a static KV cache, eager vs
the forward compiled with ktt_tt (HF's recipe): tokens vs CPU, ms per token.
    python -m kohakutt.bench.models.qwen.generate
"""

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import kohakutt
from kohakutt.torch_backend import compiler as tt_compile


def generate_record(model, ids, want, new_tokens, tok):
    """Record of greedy generate on tt: tokens equal to `want`, timings, text."""
    with torch.no_grad():
        start = time.perf_counter()
        model.generate(ids, max_new_tokens=new_tokens, do_sample=False)
        first = time.perf_counter() - start
        start = time.perf_counter()
        out = model.generate(ids, max_new_tokens=new_tokens, do_sample=False)
        torch.tt.synchronize()
        steady = time.perf_counter() - start
    got = out[0, ids.shape[1] :].cpu()
    return {
        "tokens_equal_cpu": int((got == want).sum()),
        "of": new_tokens,
        "first_run_s": round(first, 1),
        "ms_per_token": round(steady * 1e3 / new_tokens, 1),
        "text": tok.decode(got),
    }


def main():
    """Runs eager and compiled generate on device 0; prints one JSON record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prompt", default="The Tenstorrent Blackhole is")
    parser.add_argument("--tokens", type=int, default=24)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(args.prompt, return_tensors="pt").input_ids
    cpu = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        out = cpu.generate(ids, max_new_tokens=args.tokens, do_sample=False)
    want = out[0, ids.shape[1] :]

    kohakutt.open(0)
    tt_compile.register()
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
        model = model.eval().to("tt")
        model.generation_config.cache_implementation = "static"
        record = {"eager": generate_record(model, ids.to("tt"), want, args.tokens, tok)}
        model.forward = torch.compile(model.forward, backend="ktt_tt", dynamic=False)
        record["compiled"] = generate_record(
            model, ids.to("tt"), want, args.tokens, tok
        )
        record["fallbacks"] = kohakutt.fallback_counts()
        print(json.dumps(record, indent=1))
    finally:
        tt_compile.release()
        kohakutt.close()


if __name__ == "__main__":
    main()
