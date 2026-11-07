"""ViT-B/16 inference / training over batch sizes and real Qwen2.5-0.5B prefill on one
stack per process: median ms, throughput, TFLOPS (FLOPs counted on CPU). Run as a file:
    python src/kohakutt/bench/perf/workloads.py --stack kohakutt|xla  (HF_HOME set)
"""

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F
import transformers as tf
from torch.utils.flop_counter import FlopCounterMode

QWEN = "Qwen/Qwen2.5-0.5B"


def vit(batch, train):
    """make() -> (model, inputs, step, items per step) for ViT-B/16 at 224 px."""

    def make():
        torch.manual_seed(0)
        model = tf.ViTForImageClassification(tf.ViTConfig(num_labels=1000)).bfloat16()
        inputs = (
            torch.randn(batch, 3, 224, 224).bfloat16(),
            torch.randint(0, 1000, (batch,)),
        )
        return model, inputs, train_step if train else infer_step, batch

    return make


def qwen_prefill(tokens):
    """make() -> (model, inputs, step, items per step) for a Qwen2.5-0.5B prefill."""

    def make():
        model = tf.AutoModelForCausalLM.from_pretrained(
            QWEN, dtype=torch.bfloat16
        ).eval()
        ids = torch.randint(0, 32000, (1, tokens))
        return model, (ids,), lm_step, tokens

    return make


def infer_step(run, inputs, opt):
    """One no-grad image forward; returns the logits."""
    with torch.no_grad():
        return run(pixel_values=inputs[0]).logits


def train_step(run, inputs, opt):
    """One image forward, cross-entropy backward and AdamW update; returns the loss."""
    loss = F.cross_entropy(run(pixel_values=inputs[0]).logits.float(), inputs[1])
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=False)
    return loss.detach()


def lm_step(run, inputs, opt):
    """One no-grad, no-cache token forward; returns the logits."""
    with torch.no_grad():
        return run(input_ids=inputs[0], use_cache=False).logits


WORKLOADS = {
    **{f"vit_infer b{b}": vit(b, False) for b in (1, 8, 10, 32, 64)},
    **{f"vit_train b{b}": vit(b, True) for b in (8, 32)},
    **{f"qwen_prefill {t}": qwen_prefill(t) for t in (128, 512)},
}


def cpu_flops(make):
    """FLOPs of one step, counted on CPU in fp32."""
    model, inputs, step, _ = make()
    model = model.float()
    inputs = tuple(t.float() if t.is_floating_point() else t for t in inputs)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    counter = FlopCounterMode(display=False)
    with counter:
        step(model, inputs, opt)
    return counter.get_total_flops()


def stack_setup(stack):
    """(device, modes, sync) of `stack`."""
    if stack == "xla":
        # Optional dependency: torch-xla with the TT pjrt plugin (tt-xla venv).
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr

        xr.set_device_type("TT")

        def sync(device):
            torch_xla.sync()
            xm.wait_device_ops()

        return torch_xla.device(), ["lazy"], sync

    # Optional dependency: Kohaku-TT itself (this repo's venv).
    import kohakutt

    def sync(device):
        kohakutt.synchronize(device)

    return kohakutt.device(0), ["eager", "compiled"], sync


def run(name, make, stack, device, mode, sync, steps, warmup):
    """One JSON record of workload `name` in `mode`."""
    flops = cpu_flops(make)
    model, inputs, step, items = make()
    model = model.to(device)
    inputs = tuple(t.to(device) for t in inputs)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    run_model = model
    if mode == "compiled":
        training = name.startswith("vit_train")
        run_model = torch.compile(model, backend="ktt_tt_aot" if training else "ktt_tt")
    # Each step's output stays alive through the sync: a lazy stack only computes
    # results that are still referenced.
    for _ in range(warmup):
        out = step(run_model, inputs, opt)
        if stack == "xla":
            sync(device)
    sync(device)
    samples = []
    for _ in range(steps):
        start = time.perf_counter()
        out = step(run_model, inputs, opt)
        sync(device)
        samples.append(time.perf_counter() - start)
    del out
    ms = statistics.median(samples) * 1e3
    return {
        "stack": stack,
        "mode": mode,
        "workload": name,
        "ms": round(ms, 2),
        "items_per_s": round(items / (ms / 1e3), 1),
        "tflops": round(flops / (ms / 1e3) / 1e12, 2),
    }


def main():
    """Runs the workloads on one stack; prints one JSON line per (workload, mode)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", choices=["kohakutt", "xla"], required=True)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--modes", nargs="+")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    device, modes, sync = stack_setup(args.stack)
    for name, make in WORKLOADS.items():
        if args.only and not any(name.startswith(o) for o in args.only):
            continue
        for mode in args.modes or modes:
            try:
                record = run(
                    name, make, args.stack, device, mode, sync, args.steps, args.warmup
                )
            except Exception as error:  # noqa: BLE001  (a failing workload is reported)
                record = {
                    "stack": args.stack,
                    "mode": mode,
                    "workload": name,
                    "error": str(error).splitlines()[0][:300],
                }
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
