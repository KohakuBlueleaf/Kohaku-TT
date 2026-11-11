"""Fine-tuning a language model from transformers on a chip with AdamW: LoRA adapters on
the attention q / v projections, or (--last N) the last N decoder layers. The model
learns one fixed batch, so the loss falls fast.   python lora.py --last 2
"""

import argparse
import math
import time

import torch
import torch.nn.functional as F
from common import random_tokens
from torch import nn
from transformers import AutoModelForCausalLM

import kohakutt


class LoRA(nn.Module):
    """base(x) + scale * B A x with a frozen `base` Linear; B starts at zero."""

    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base, self.scale = base, alpha / rank
        a = torch.randn(rank, base.in_features) / math.sqrt(base.in_features)
        self.A = nn.Parameter(a)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, x):
        delta = (x.float() @ self.A.t()) @ self.B.t()
        return self.base(x) + (self.scale * delta).to(x.dtype)


def lm_loss(model, ids):
    logits = model(input_ids=ids[:, :-1]).logits
    return F.cross_entropy(logits.float().flatten(0, 1), ids[:, 1:].flatten())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--last", type=int, default=0, help="train the last N layers")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    dev = kohakutt.device(0)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval().requires_grad_(False)
    ids = random_tokens(args.seq + 1, args.batch)
    with torch.no_grad():
        cpu_loss = lm_loss(model, ids).item()  # the same bf16 weights on the CPU

    if args.last:
        model.model.layers[-args.last :].requires_grad_(True)
        model.model.norm.requires_grad_(True)
    else:
        for layer in model.model.layers:
            attn = layer.self_attn
            attn.q_proj, attn.v_proj = LoRA(attn.q_proj), LoRA(attn.v_proj)
    model = model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in params) / 1e6:.2f} M")
    opt = torch.optim.AdamW(params, lr=1e-4 if args.last else 1e-3)
    ids = ids.to(dev)
    for step in range(args.steps):
        start = time.perf_counter()
        loss = lm_loss(model, ids)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        value = loss.item()
        ms = (time.perf_counter() - start) * 1e3
        extra = f"  (CPU {cpu_loss:.4f})" if step == 0 else ""
        print(f"step {step:2d} loss {value:.4f}  {ms:8.1f} ms{extra}")
    kohakutt.close()


if __name__ == "__main__":
    main()
