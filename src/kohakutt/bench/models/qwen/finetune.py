"""Experiment: real Qwen2.5-0.5B fine-tuned for a few AdamW steps on one fixed batch;
loss and grad norm per step in CPU-bf16 eager, tt eager or tt compiled (ktt_tt_aot).
    python -m kohakutt.bench.models.qwen.finetune --mode tt  (HF_HOME set)
"""

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import kohakutt
from kohakutt.torch_backend import compiler as tt_compile

TEXT = (
    "Tenstorrent Blackhole cards run PyTorch through Kohaku-TT: eager kernels, traces "
    "and torch.compile share one device runtime."
)


def finetune(mode, model_name, steps, lr):
    """Record of `steps` AdamW steps in `mode` (cpu | tt | compiled)."""
    device = "cpu" if mode == "cpu" else "tt"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    model = model.to(device).train()
    ids = tok(TEXT, return_tensors="pt").input_ids.to(device)
    run = torch.compile(model, backend="ktt_tt_aot") if mode == "compiled" else model
    opt = torch.optim.AdamW(model.parameters(), lr=lr, foreach=True)
    losses, norms, times = [], [], []
    for _ in range(steps):
        start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = run(input_ids=ids, labels=ids).loss
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        opt.step()
        losses.append(round(loss.item(), 4))
        norms.append(round(norm.item(), 2))
        times.append(round((time.perf_counter() - start) * 1e3))
    return {
        "mode": mode,
        "tokens": ids.shape[1],
        "losses": losses,
        "grad_norms": norms,
        "ms_per_step": times,
    }


def main():
    """Runs one mode and prints one JSON record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cpu", "tt", "compiled"], default="tt")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    if args.mode == "cpu":
        print(json.dumps(finetune(args.mode, args.model, args.steps, args.lr)))
        return
    kohakutt.open(0)
    tt_compile.register()
    try:
        record = finetune(args.mode, args.model, args.steps, args.lr)
        record["fallbacks"] = kohakutt.fallback_counts()
        print(json.dumps(record))
    finally:
        tt_compile.release()
        kohakutt.close()


if __name__ == "__main__":
    main()
