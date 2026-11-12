"""Language-model prefill (one forward over a prompt, logits out) from transformers on a
chip, eager and with torch.compile, timed, against fp32 on the CPU.
    python lm_prefill.py --model Qwen/Qwen3-0.6B --tokens 256
"""

import argparse

import torch
from common import random_tokens, rel_l2, report, timed, to_bf16
from transformers import AutoModelForCausalLM

import kohakutt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()

    dev = kohakutt.device(0)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    ids = random_tokens(args.tokens)
    with torch.no_grad():
        ref = model(input_ids=ids, use_cache=False).logits

    model = to_bf16(model).to(dev)
    ids_tt = ids.to(dev)
    compiled = torch.compile(model, backend="ktt_tt")

    def check(name, net):
        """Times net's prefill; reports the error and the next-token agreement."""

        def prefill():
            with torch.no_grad():
                return net(input_ids=ids_tt, use_cache=False).logits

        out, first, median = timed(prefill, dev)
        report(name, first, median, rel_l2(out, ref))
        agree = (out.cpu().argmax(-1) == ref.argmax(-1)).float().mean().item()
        print(f"{'':9s} next-token argmax equal to the CPU at {agree:.1%} of positions")

    check("eager", model)
    check("compiled", compiled)
    kohakutt.close()


if __name__ == "__main__":
    main()
