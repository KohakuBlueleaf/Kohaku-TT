"""A language-model prefill from transformers captured once with `kohakutt.capture` and
replayed as one device trace, against the same forward run eagerly.
    python lm_capture.py --model Qwen/Qwen3-0.6B --tokens 256
"""

import argparse

import torch
from common import random_tokens, rel_l2, report, timed
from transformers import AutoModelForCausalLM

import kohakutt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()

    dev = kohakutt.device(0)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model = model.eval().to(dev)
    ids = random_tokens(args.tokens).to(dev)

    # keep the default use_cache: use_cache=False reads the host inside the capture
    def prefill(input_ids):
        with torch.no_grad():
            return model(input_ids=input_ids).logits

    eager, first, median = timed(lambda: prefill(ids), dev)
    report("eager", first, median)
    replay = kohakutt.capture(prefill, ids)
    out, first, median = timed(lambda: replay(ids), dev)
    report("captured", first, median, rel_l2(out, eager), against="eager")

    # a replay reads its inputs from the captured buffers: new ids, same shape
    other = random_tokens(args.tokens, seed=1).to(dev)
    err = rel_l2(replay(other), prefill(other))
    print(f"new input ids, same length: rel L2 replay vs eager {err:.2e}")
    replay.release()
    kohakutt.close()


if __name__ == "__main__":
    main()
