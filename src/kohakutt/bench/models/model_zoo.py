"""Experiment: tiny random-weight HF models (several architectures) on tt: eager vs
CPU / fp64 statistics and fallbacks, or (--compile) ktt_tt_aot vs eager on tt.
    KTT_CPU_FALLBACK=1 KTT_FALLBACK_WARN=0 python -m kohakutt.bench.models.model_zoo
"""

import argparse
import copy
import json
import time
import traceback

import torch
import torch._dynamo
import transformers as tf

import kohakutt
from kohakutt.core.stats import diff_stats
from kohakutt.torch_backend import compiler as tt_compile

TEXT = dict(vocab_size=512)
ATTN = dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4)


def make(kind):
    """(seeded bf16 eval model, example inputs dict on CPU) for architecture `kind`."""
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    tokens = torch.randint(0, 512, (2, 16), generator=gen)
    images = torch.randn(2, 3, 32, 32, generator=gen).bfloat16()
    if kind == "bert":
        cfg = tf.BertConfig(**TEXT, **ATTN, intermediate_size=128, num_labels=3)
        return tf.BertForSequenceClassification(cfg), {"input_ids": tokens}
    if kind == "gpt2":
        cfg = tf.GPT2Config(**TEXT, n_embd=64, n_layer=2, n_head=4, n_positions=64)
        return tf.GPT2LMHeadModel(cfg), {"input_ids": tokens}
    if kind == "t5":
        cfg = tf.T5Config(
            **TEXT, d_model=64, d_ff=128, d_kv=16, num_layers=2, num_heads=4
        )
        return tf.T5ForConditionalGeneration(cfg), {
            "input_ids": tokens,
            "decoder_input_ids": tokens[:, :8],
        }
    if kind == "vit":
        cfg = tf.ViTConfig(
            **ATTN, intermediate_size=128, image_size=32, patch_size=8, num_labels=10
        )
        return tf.ViTForImageClassification(cfg), {"pixel_values": images}
    if kind == "resnet":
        cfg = tf.ResNetConfig(
            embedding_size=16, hidden_sizes=[16, 32], depths=[1, 1], num_labels=10
        )
        return tf.ResNetForImageClassification(cfg), {"pixel_values": images}
    if kind == "convnext":
        cfg = tf.ConvNextConfig(
            num_stages=2,
            hidden_sizes=[16, 32],
            depths=[1, 1],
            num_labels=10,
            image_size=32,
        )
        return tf.ConvNextForImageClassification(cfg), {"pixel_values": images}
    if kind == "whisper":
        cfg = tf.WhisperConfig(
            **TEXT,
            d_model=64,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=128,
            decoder_ffn_dim=128,
            num_mel_bins=16,
            max_source_positions=16,
            max_target_positions=32,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            decoder_start_token_id=1,
        )
        features = torch.randn(2, 16, 32, generator=gen).bfloat16()
        return tf.WhisperForConditionalGeneration(cfg), {
            "input_features": features,
            "decoder_input_ids": tokens[:, :8],
        }
    raise ValueError(kind)


def logits_and_grads(model, inputs, device, dtype=None):
    """(logits on CPU as fp64, {param name: grad on CPU as fp64}) on `device`; floating
    inputs cast to `dtype` when given."""
    model = model.to(device)

    def move(v):
        cast = dtype if dtype is not None and v.is_floating_point() else v.dtype
        return v.to(device=device, dtype=cast)

    moved = {k: move(v) for k, v in inputs.items()}
    logits = model(**moved).logits
    logits.float().square().mean().backward()
    grads = {
        n: p.grad.cpu().double()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    return logits.detach().cpu().double(), grads


def check(kind):
    """Record for `kind`: logits / grad statistics and fallback counts, or the error."""
    kohakutt.reset_fallback_counts()
    model, inputs = make(kind)
    model = model.bfloat16().eval()
    record = {"model": kind}
    try:
        # fp64 reference from the same bf16 weights and inputs.
        reference = copy.deepcopy(model).double()
        ref_logits, ref_grads = logits_and_grads(
            reference, inputs, "cpu", torch.float64
        )
        # Baseline: the same bf16 model executed by PyTorch on CPU.
        cpu_logits, cpu_grads = logits_and_grads(copy.deepcopy(model), inputs, "cpu")
        got_logits, got_grads = logits_and_grads(model, inputs, "tt")
    except Exception as error:  # noqa: BLE001  (experiment: record and continue)
        record["error"] = f"{type(error).__name__}: {str(error).splitlines()[0][:200]}"
        record["where"] = traceback.format_exc().splitlines()[-3].strip()[:200]
        record["fallbacks"] = kohakutt.fallback_counts()
        return record
    record["logits"] = diff_stats(got_logits, ref_logits)
    record["cpu_bf16_logits"] = diff_stats(cpu_logits, ref_logits)
    # Parameters whose true gradient is zero (fp64 reference at roundoff level, e.g.
    # attention key biases) are counted, not compared.
    scale = max(g.abs().max().item() for g in ref_grads.values())
    zero = [n for n, g in ref_grads.items() if g.abs().max().item() < 1e-9 * scale]
    grads = {
        n: diff_stats(got_grads[n], ref_grads[n])
        for n in ref_grads
        if n in got_grads and n not in zero
    }
    record["grads"] = len(got_grads)
    record["zero_grad_params"] = len(zero)

    def median(values):
        values = sorted(values)
        return values[len(values) // 2] if values else None

    # Median over parameters of each gradient's rel error median, tt and CPU bf16.
    record["grad_median_rel_median"] = {
        "tt": median(s["rel_median"] for s in grads.values()),
        "cpu_bf16": median(
            diff_stats(cpu_grads[n], ref_grads[n])["rel_median"]
            for n in grads
            if n in cpu_grads
        ),
    }
    # The parameter whose tt gradient has the lowest cosine, with its CPU bf16 stats.
    worst = min(grads, key=lambda n: grads[n]["cosine"]) if grads else None
    if worst:
        record["grad_lowest_cosine"] = {
            "param": worst,
            "tt": grads[worst],
            "cpu_bf16": (
                diff_stats(cpu_grads[worst], ref_grads[worst])
                if worst in cpu_grads
                else None
            ),
        }
    record["fallbacks"] = kohakutt.fallback_counts()
    return record


def tt_step(run, model, inputs):
    """(logits, {param name: grad}) on CPU of one forward + backward of `run` on tt."""
    model.zero_grad(set_to_none=True)
    logits = run(**inputs).logits
    logits.float().square().mean().backward()
    grads = {
        n: p.grad.cpu().double()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    return logits.detach().cpu().double(), grads


def timed_ms(fn, repeat=3):
    """Mean ms per call of `fn` after one untimed call."""
    fn()
    torch.tt.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.tt.synchronize()
    return round((time.perf_counter() - start) * 1e3 / repeat, 1)


def compiled_vs_eager(kind):
    """Record for `kind` compiled with ktt_tt_aot vs eager on tt: logits / grads
    equality, ms per step, fallbacks (needs the backends registered)."""
    torch._dynamo.reset()
    kohakutt.reset_fallback_counts()
    model, inputs = make(kind)
    model = model.bfloat16().eval().to("tt")
    inputs = {k: v.to("tt") for k, v in inputs.items()}
    record = {"model": kind}
    try:
        eager_logits, eager_grads = tt_step(model, model, inputs)
        compiled = torch.compile(model, backend="ktt_tt_aot")
        # The first EAGER_CALLS calls run eagerly; the last one captures the trace (its
        # results are its own eager pass's, no replay).
        for _ in range(tt_compile.EAGER_CALLS + 1):
            logits, grads = tt_step(compiled, model, inputs)
    except Exception as error:  # noqa: BLE001  (experiment: record and continue)
        record["error"] = f"{type(error).__name__}: {str(error).splitlines()[0][:200]}"
        record["fallbacks"] = kohakutt.fallback_counts()
        tt_compile.release()
        return record
    same = [
        n for n in eager_grads if n in grads and torch.equal(grads[n], eager_grads[n])
    ]
    record["logits_equal"] = bool(torch.equal(logits, eager_logits))
    record["grads_equal"] = f"{len(same)}/{len(eager_grads)}"
    record["eager_ms"] = timed_ms(lambda: tt_step(model, model, inputs))
    record["compiled_ms"] = timed_ms(lambda: tt_step(compiled, model, inputs))
    record["fallbacks"] = kohakutt.fallback_counts()
    # This model's traces leave the device's trace region for the next one.
    tt_compile.release()
    return record


def main():
    """Runs every requested architecture on device 0 and prints one record each
    (eager vs CPU / fp64, or with --compile: compiled vs eager on tt)."""
    parser = argparse.ArgumentParser()
    kinds = ["bert", "gpt2", "t5", "vit", "resnet", "convnext", "whisper"]
    parser.add_argument("--models", nargs="+", default=kinds)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    kohakutt.open(0)
    tt_compile.register()
    try:
        for kind in args.models:
            record = compiled_vs_eager(kind) if args.compile else check(kind)
            print(json.dumps(record), flush=True)
    finally:
        tt_compile.release()
        kohakutt.close()


if __name__ == "__main__":
    main()
