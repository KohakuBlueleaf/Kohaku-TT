"""Experiment: Qwen2.5-0.5B greedy decode with a static KV cache, one captured trace per
token (forward + argmax on device), vs CPU greedy tokens and eager tt time.
    python -m kohakutt.bench.capture.decode_graph [--tokens 32]  (HF_HOME set)
"""

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

import kohakutt
from kohakutt import capture
from kohakutt.torch_backend.trace import footprint

MODEL = "Qwen/Qwen2.5-0.5B"
PROMPT = "The Tenstorrent Blackhole chip is"


def main():
    """Decodes on CPU (reference) and on tt by trace replay; prints one record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPT, return_tensors="pt").input_ids
    cpu = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        ref = cpu.generate(ids, max_new_tokens=args.tokens, do_sample=False)[
            0, ids.shape[1] :
        ]

    kohakutt.open(0)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
        model = model.eval().to("tt")
        length = ids.shape[1]
        cache = StaticCache(config=model.config, max_cache_len=length + args.tokens + 8)
        with torch.no_grad():
            # Prefill eagerly; the first new token comes from its last logits.
            out = model(
                ids.to("tt"),
                past_key_values=cache,
                cache_position=torch.arange(length).to("tt"),
                use_cache=True,
            )
            first = out.logits[:, -1].argmax(-1, keepdim=True)

            def step(token, position):
                """One decode step: next token id (1, 1) on device."""
                logits = model(
                    token,
                    past_key_values=cache,
                    cache_position=position,
                    position_ids=position.unsqueeze(0),
                    use_cache=True,
                ).logits
                return logits[:, -1].argmax(-1, keepdim=True)

            position = torch.tensor([length]).to("tt")
            graph = capture(step, first, position, warmup=3)
            kept = footprint(graph)

            # Replay from the prefill state: positions length, length + 1, ...
            tokens, token = [], first
            torch.tt.synchronize()
            start = time.perf_counter()
            # Each token is read to host right away: replay outputs on device are only
            # guaranteed until the next replay (docs/concepts/trace.md, known issues).
            for i in range(args.tokens):
                position = torch.tensor([length + i]).to("tt")
                tokens.append(token.cpu())
                token = graph(token, position)
            torch.tt.synchronize()
            ms = (time.perf_counter() - start) * 1e3 / args.tokens
            got = torch.cat(tokens, dim=1)[0]
        graph.release()
    finally:
        kohakutt.close()

    print("ref ids", ref.tolist(), flush=True)
    print("tt  ids", got.tolist(), flush=True)
    same = (got == ref).tolist()
    print(
        json.dumps(
            {
                "model": MODEL,
                "tokens": args.tokens,
                "agreement": round(sum(same) / len(same), 3),
                "tokens_equal_until": same.index(False) if False in same else None,
                "replay_ms_per_token": round(ms, 2),
                "trace_held_buffers": graph.held,
                "trace_footprint": kept,
                "tt_text": tok.decode(got.clamp(0, len(tok) - 1)),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
