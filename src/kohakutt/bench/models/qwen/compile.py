"""Experiment: real Qwen2.5-0.5B, one training step (forward + backward) compiled with
ktt_tt_aot vs eager on tt: loss / logits / grads equality, ms per step, memory.
    python -m kohakutt.bench.models.qwen.compile
"""

import argparse
import json
import time

import torch
import torch._dynamo
from transformers import AutoModelForCausalLM

import kohakutt
from kohakutt.torch_backend import compiler as tt_compile


def step(run, model, ids):
    """(loss, logits) of one forward + backward of `run` on token ids."""
    model.zero_grad(set_to_none=True)
    out = run(input_ids=ids, labels=ids)
    out.loss.backward()
    return out.loss.detach(), out.logits.detach()


def grads(model):
    """{param name: grad on CPU}."""
    return {n: p.grad.cpu() for n, p in model.named_parameters() if p.grad is not None}


def timed_ms(fn, repeat=3):
    """Mean ms per call of `fn`."""
    torch.tt.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.tt.synchronize()
    return round((time.perf_counter() - start) * 1e3 / repeat, 1)


def main():
    """Runs the comparison on device 0 and prints one JSON record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--seq", type=int, default=32)
    args = parser.parse_args()

    kohakutt.open(0)
    tt_compile.register()
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
        model = model.eval().to("tt")
        gen = torch.Generator().manual_seed(0)
        ids = torch.randint(0, 1000, (1, args.seq), generator=gen).to("tt")

        start = time.perf_counter()
        loss_e, logits_e = step(model, model, ids)
        grads_e = grads(model)
        eager_first = time.perf_counter() - start

        compiled = torch.compile(model, backend="ktt_tt_aot")
        start = time.perf_counter()
        # The first EAGER_CALLS calls run eagerly; the last one replays the trace.
        for _ in range(tt_compile.EAGER_CALLS + 1):
            loss_c, logits_c = step(compiled, model, ids)
        grads_c = grads(model)
        compile_first = time.perf_counter() - start
        same = sum(torch.equal(grads_c[n], grads_e[n]) for n in grads_e if n in grads_c)

        record = {
            "model": args.model,
            "tied_embeddings": model.lm_head.weight is model.model.embed_tokens.weight,
            "loss_eager": round(loss_e.item(), 4),
            "loss_compiled": round(loss_c.item(), 4),
            "logits_equal": bool(torch.equal(logits_c.cpu(), logits_e.cpu())),
            "grads_equal": f"{same}/{len(grads_e)}",
            "eager_first_s": round(eager_first, 1),
            "compiled_first_s": round(compile_first, 1),
            "eager_ms": timed_ms(lambda: step(model, model, ids)),
            "compiled_ms": timed_ms(lambda: step(compiled, model, ids)),
            "fallbacks": kohakutt.fallback_counts(),
            "peak_mib": round(torch.tt.max_memory_allocated() / 2**20),
        }
        print(json.dumps(record, indent=1))
    finally:
        tt_compile.release()
        kohakutt.close()


if __name__ == "__main__":
    main()
