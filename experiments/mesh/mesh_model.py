"""A Hugging Face causal LM forward on a 1x2 mesh, with automatic placement (default)
against replicated placement (KTT_MESH_AUTO=0, the single-card kernels).

Prints next-token logits error statistics against a CPU float64 model, next to the
CPU bf16 model's; warm forward time per run; the auto-placed run's conversions and
gather fallbacks per op (torch.tt.mesh_stats / mesh_fallback_ops); and the
placements of the model's largest parameters. Needs HF_HOME with the model downloaded.

    python experiments/mesh/mesh_model.py
    python experiments/mesh/mesh_model.py --model Qwen/Qwen2.5-0.5B --seq 128 --repeat 5
"""

import argparse
import os
import time

import torch
from transformers import AutoModelForCausalLM

import kohakutt


def _forward_ms(model, ids, repeat, warmup):
    with torch.no_grad():
        for _ in range(warmup):
            model(ids)
        torch.tt.synchronize()
        start = time.perf_counter()
        for _ in range(repeat):
            logits = model(ids).logits
        torch.tt.synchronize()
    return logits, (time.perf_counter() - start) * 1e3 / repeat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--warmup", type=int, default=3, help="untimed calls (placements settle)"
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    cpu = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).eval()
    vocab = cpu.config.vocab_size
    ids = torch.randint(0, vocab, (1, args.seq))
    with torch.no_grad():
        base = cpu(ids).logits[0, -1].double()
        ruler = cpu.double()(ids).logits[0, -1]
    del cpu

    kohakutt.mesh([0, 1], shape=(1, 2))
    try:
        results = {}
        for mode in ("replicated", "auto"):
            os.environ["KTT_MESH_AUTO"] = "0" if mode == "replicated" else "1"
            model = AutoModelForCausalLM.from_pretrained(
                args.model, dtype=torch.bfloat16
            )
            model = model.eval().to("tt")
            if mode == "auto":
                big = sorted(model.named_parameters(), key=lambda kv: -kv[1].numel())[
                    :6
                ]
                for name, p in big:
                    print(f"param {name} {tuple(p.shape)}: {kohakutt.placement(p)}")
            torch.tt.reset_mesh_stats()
            logits, ms = _forward_ms(model, ids.to("tt"), args.repeat, args.warmup)
            stats = {k: v for k, v in torch.tt.mesh_stats().items() if v}
            fallbacks = torch.tt.mesh_fallback_ops()
            results[mode] = logits[0, -1].cpu().double()
            print(
                f"{mode}: forward {ms:.1f} ms "
                f"(seq {args.seq}, warm, {args.repeat} runs)"
            )
            print(f"    conversions {stats}")
            print(f"    fallbacks per op {fallbacks}")
            del model
        for mode, got in results.items():
            s = kohakutt.diff_stats(got, ruler)
            print(f"logits {mode:10s} {kohakutt.format_stats(s)}")
        print(
            f"logits cpu bf16   "
            f"{kohakutt.format_stats(kohakutt.diff_stats(base, ruler))}"
        )
        print(
            "auto vs replicated top-1 equal: "
            f"{int(results['auto'].argmax()) == int(results['replicated'].argmax())}"
        )
    finally:
        os.environ.pop("KTT_MESH_AUTO", None)
        kohakutt.close()


if __name__ == "__main__":
    main()
