---
title: Result zoo
summary: Every model run on Kohaku-TT 0.1.0 with its time, error and the same run on tt-xla - vision, diffusion, language models, training, attention call shapes - and the problems found.
tags:
  - reference
  - results
  - benchmark
---

# Result zoo

What runs, how fast, how accurate, and what does not: every model tried on release
0.1.0, with the same run on tt-xla where it runs; headline rows re-measured on the
current build come first, and "now" in an older row gives its current figure. Curated comparisons and kernel-level
numbers are in [measured](measured.md).

## Conditions

- One Blackhole P150A per run, Kohaku-TT 0.1.0, bf16 weights and activations, eager
  unless a column says otherwise, warm (the first call is listed separately where
  measured). Local weights from the Hugging Face cache; Transformers SDPA attention.
- tt-xla: pjrt-plugin-tt 1.4.0 with torch-xla 2.9. Vision and diffusion models use
  `torch.compile(model, backend="tt")`: in torch-xla lazy mode every vision model gave
  non-finite or wrong output, fp32 included. Language models use lazy mode (outputs
  checked against CPU) unless a row says compiled.
- Vision error: relative L2 of the output against the CPU fp32 run of the same weights
  and input. Language-model error: relative L2 of the logits against CPU bf16; "text"
  is greedy tokens equal to CPU bf16's. A number from a wrong output is marked and
  does not count.
- The host was shared (load average 15 to 75). Host-bound numbers (small batch-1
  forwards, decode, first calls) are slower and noisier than on an idle host: the
  DINOv3 ConvNeXt-tiny forward is 28.7 ms here and 14.1 ms in [measured](measured.md).
  Compare within a row.

## Current build: re-measured headline rows

Measured after release 0.1.0 on the current tree: one P150A, bf16, wall-clock of
synchronized calls (median of 5 to 30 calls after warm-up), batch 1 unless noted. Decode:
greedy, a 77-token prompt, `StaticCache` of 77 + 64 entries, position advancing; compiled
is `torch.compile(model, backend="ktt_tt", dynamic=False)`. Captured is
`kohakutt.capture`. Rows marked "to be revisited" are unverified or inconsistent with an
earlier figure and are listed in the notes below the tables.

### Language models: decode

| Model | Eager ms / tok/s | Compiled ms / tok/s | Compiled tokens equal to eager |
|---|---|---|---|
| Qwen2.5-0.5B | 67.1 / 14.9 | 8.3 / 120 | 24 of 24 |
| Qwen3-1.7B | 106.0 / 9.4 | 18.0 / 55.5 | 24 of 24 |
| Qwen2.5-3B | 103.5 / 9.7 | 26.2 / 38.1 | differs at token 9, a bf16 tie (to be revisited) |
| Qwen3.5-4B, gated delta net decode kernel | 108.8 / 9.2 | 43.9 / 22.8 | equal to CPU, 17 tokens |
| Qwen3.5-4B, transformers' gated delta net | 187.8 / 5.3 | | |

Compiled decode at larger batch (tokens/s summed over the batch; ratio to batch 1):

| Model | Batch 1 | Batch 4 | Batch 16 |
|---|---|---|---|
| Qwen2.5-0.5B | 8.3 ms, 120 tok/s | 11.1 ms, 359 (3.0x) | 27.3 ms, 585 (4.9x) |
| Qwen3-1.7B | 18.0 ms, 55.5 | 23.7 ms, 169 (3.0x) | 46.7 ms, 343 (6.2x) |
| Qwen2.5-3B | 26.2 ms, 38.1 | 33.7 ms, 119 (3.1x) | |

Batch 1 is 2 to 4 times the time it takes to stream the weights (bf16 at 512 GB/s), and
the scaling flattens well before the matmul compute bound (to be revisited).

### Language models: prefill

Random token ids, full logits. Captured outputs were not compared with eager in this
run.

| Model | Tokens x batch | Eager ms | Captured ms | Tokens/s (eager) |
|---|---|---|---|---|
| Qwen2.5-0.5B | 512 | 62.2 | 30.7 | 8.2k |
| Qwen2.5-0.5B | 512 x 4 | 68.5 | 64.0 | 29.9k |
| Qwen2.5-0.5B | 2048 | 68.7 | 65.8 | 29.8k |
| Qwen3-1.7B | 512 | 93.0 | 61.0 | 5.5k |
| Qwen3-1.7B | 512 x 4 | 137.8 | 142.8 | 14.9k |
| Qwen3-1.7B | 2048 | 144.3 | 148.9 | 14.2k |
| Qwen2.5-3B | 512 | 94.3 | 70.4 | 5.4k |
| Qwen2.5-3B | 512 x 4 | 185.4 | 189.0 | 11.0k |
| Qwen2.5-3B | 2048 | 185.3 | 188.7 | 11.0k |
| Qwen3.5-4B | 512 | 2356 | 2387 | 217 |
| Qwen3.5-4B | 2048 | 9522 (1 call) | 9744 (1 call) | 215 |
| Gemma-4 E4B | 197 | 216.8 | 144.4 | 909 |

Qwen3.5-4B prefill runs transformers' reference chunked gated delta rule (12k device ops
at 512 tokens, 25k at 2048); there is no prefill kernel for it yet (to be revisited).

### Training, vision and diffusion

| Model | Setup | Eager | Captured |
|---|---|---|---|
| Qwen2.5-0.5B LoRA (q, v, r 8), seq 64, batch 2 | AdamW; captured with `capturable=True` | 245 ms | 102 ms (to be revisited) |
| PixAI tagger v1.0, 1008 px | batch 1 | 518 ms | 522 ms |
| PixAI tagger v1.0, 1008 px | batch 4 | 2027 ms | 2046 ms |
| SDXL UNet (Kohaku-XL-Zeta), 1024 px, one CFG step | batch 2 | 334.5 ms | 333.4 ms |
| SDXL UNet, 1024 px | batch 1 / batch 4 | 190.3 / 690.0 ms | 188.9 / 688.3 ms |
| SDXL VAE decode, 1024 px | batch 1 | 665.4 ms | 663.5 ms |
| MOSS-Image CFG step, 1024 x 1024, 300 / 8 text tokens | bf16 weights | 1154.5 ms | 1166.6 ms |
| MOSS-Image CFG step, 1024 x 1024, 300 / 8 text tokens | bfp8 DiT linears, HiFi2 | 1182.5 ms | 1200.1 ms |

Output checks against CPU fp32: SDXL UNet relative L2 5.4e-3, SDXL VAE 1.6e-2, MOSS-Image
CFG step 0.167 (bf16) and 0.239 (bfp8; the guidance scale amplifies the per-forward noise
floor, see [performance](../concepts/performance.md)).

### To be revisited

- Qwen2.5-3B compiled decode: the token stream differs from eager at one position where
  eager's top two logits are tied in bf16 (gap 0.0 against one bf16 step compiled).
- Qwen2.5-0.5B LoRA captured step: 102 ms here; 69 ms was published before. Not
  investigated.
- PixAI tagger: batch 4 takes 4x batch 1 and capture gains nothing. Batch 16's first
  call takes longer than 35 s and its setup is not cached across processes.
- Qwen3.5-4B prefill: about 215 tokens/s at any length (no gated delta net prefill
  kernel).
- Compiled decode batch scaling: batch 1 is 2 to 4 times the weight-streaming time and
  the gain flattens (Qwen2.5-0.5B 4.9x at batch 16, Qwen2.5-3B 3.1x at batch 4).
- MOSS-Image 1024 x 1024 CFG step: 1.18 s in the bfp8 configuration, 26 % slower than
  the 0.935 s published before; bfp8 is not faster than bf16. Not investigated.

## Vision encoders, inference at 224 px

| Model | Batch | Kohaku-TT ms | Error | tt-xla ms | Error | tt-xla / Kohaku-TT |
|---|---|---|---|---|---|---|
| DINOv3 ViT-S+/16 | 1 | 37.5 | 2.8e-2 | 70.5 | 4.1e-2 | 1.9x |
| DINOv3 ViT-S+/16 | 8 | 34.9 | 3.5e-2 | 65.8 | 4.5e-2 | 1.9x |
| DINOv3 ViT-B/16 | 1 | 36.3 | 2.3e-2 | 68.2 | 5.4e-2 | 1.9x |
| DINOv3 ViT-B/16 | 8 | 50.4 | 2.3e-2 | 68.9 | 5.6e-2 | 1.4x |
| DINOv3 ViT-L/16 | 1 | 72.9 | 2.5e-2 | fails (output readback) | | |
| DINOv3 ViT-L/16 | 8 | 126.2 | 2.8e-2 | 127.1 | 8.2e-2 | 1.0x |
| DINOv3 ConvNeXt-tiny | 1 | 28.7 | 3.9e-2 | 1411 | 4.3e-2 | 49x |
| DINOv3 ConvNeXt-tiny | 8 | 55.6 | 3.6e-2 | 1388 | 6.7e-2 | 25x |
| DINOv3 ConvNeXt-small | 1 | 37.0 | 2.7e-2 | 2762 | 4.2e-2 | 75x |
| DINOv3 ConvNeXt-small | 8 | 78.8 | 3.3e-2 | 2285 | 5.4e-2 | 29x |
| DINOv3 ConvNeXt-base | 1 | 34.9 | 4.8e-2 | 3983 | 6.9e-2 | 114x |
| DINOv3 ConvNeXt-base | 8 | 102.9 | 4.3e-2 | 4030 | 7.6e-2 | 39x |
| DINOv3 ConvNeXt-large | 1 | 43.2 | 3.8e-2 | 11240 | 8.3e-2 | 260x |
| DINOv3 ConvNeXt-large | 8 | 134.4 | 4.1e-2 | 11821 | 1.0e-1 | 88x |
| facebook/convnext-tiny-224 | 1 | 21.1 | 9.6e-3 | 1394 | 1.3e-2 | 66x |
| facebook/convnext-tiny-224 | 8 | 55.0 | 1.0e-2 | 1365 | 1.4e-2 | 25x |
| facebook/convnext-base-224 | 1 | 40.1 | 1.9e-2 | 4011 | 3.4e-2 | 100x |
| facebook/convnext-base-224 | 8 | 102.2 | 2.3e-2 | 4360 | 4.0e-2 | 43x |
| facebook/convnext-large-224 | 1 | 37.4 | 1.2e-2 | 11496 | 2.1e-2 | 307x |
| facebook/convnext-large-224 | 8 | 133.6 | 1.1e-2 | 12070 | 2.2e-2 | 90x |

Kohaku-TT runs all 20 points; tt-xla runs 19, with errors 1.5 to 2.5 times
Kohaku-TT's and first calls of 9 to 61 s.

Other sizes on Kohaku-TT: ViT-B/16 at 512 px 53.0 ms (tt-xla 68.0), at 1008 px 195 ms;
ConvNeXt-tiny at 512 px 30.4 ms (tt-xla 1454), at 1024 px 74.9 ms; batch 32 at 224 px
183 ms (ViT-B) and 204 ms (ConvNeXt-tiny); batch 128 755 ms and 710 ms (5.9 and 5.5 ms
per image), no memory limit reached.

## Other vision and diffusion models, batch 1

| Model | Input | Kohaku-TT ms (first call s) | Error | tt-xla ms (first call s) | tt-xla result |
|---|---|---|---|---|---|
| pixai-tagger-v1.0 | 1008 px | 531 (5.4); now 518 | 9.2e-3 | 5543 (114) | 3.3e-2 |
| sd-vae-ft-mse decode | 512 px | 226 (1.3) | 2.2e-2 | 967 (25) | wrong output |
| sd-vae-ft-mse encode | 512 px | 119 (3.0) | 2.0e-2 | 569 (15) | wrong output |
| sdxl-vae decode | 512 px | 226 (1.2) | 2.6e-2 | 972 (37) | non-finite |
| sdxl-vae encode | 512 px | 119 (2.8) | 1.0e-1 | 570 (16) | wrong output |
| sdxl-vae decode | 1024 px | 803 (7.8); now 665 | 5.9e-2 | 3114 (99) | non-finite |
| sdxl-vae encode | 1024 px | 407 (26.7) | 8.7e-2 | 1570 (105) | wrong output |
| Kohaku-XL-Zeta UNet, CFG (batch 2) | 512 px | 326 (31) | 5.8e-3 | compile did not finish in 280 s | |
| Kohaku-XL-Zeta UNet, CFG (batch 2) | 1024 px | 632 (37); now 334.5 | 5.6e-3 | not run | |
| sam3 vision encoder | 1008 px | 2057 (16) | 1.1e-1 | 5447 (147) | 2.4e-1 |
| VideoMAE base | 16 x 224 px | fails: a gather the kernel does not cover | | 40 (11) | 2.2e-2 |

pixai-tagger takes 1008 px only (the model asserts its grid). VideoMAEv2 needs a
package neither environment had and was not run.

## Language models

### Prefill, 77 tokens

| Model | Kohaku-TT ms | Error | First call s | tt-xla ms | Error | First call s |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 54.7 | 0.025 | 9.0 | 220.5 | 0.035 | 8.0 |
| Qwen2.5-1.5B | 84.4 | 0.026 | 7.9 | 234 to 250 | 0.07 | 20 to 36 |
| Qwen2.5-3B | 111.6 | 0.027 | 0.5 | 308.8 | 0.052 | 20.6 |
| Qwen3-0.6B | 115.5 | 0.033 | 6.3 | 225.1 | 0.050 | 22.1 |
| Qwen3-1.7B | 112.4 | 0.068 | 0.5 | 226.9 | 0.087 | 12.6 |
| Gemma-4 E2B (197 tokens) | 532 (191 at 77) | 0.015 | 7.8 | 581 | 0.030 | 138.6 |
| Gemma-4 E4B (197 tokens) | 261.5; now 216.8 | 0.030 | 4.9 | 723.8 | 0.039 | 132.2 |
| Gemma-4 12B (197 tokens; tt-xla 32) | 354.8 | 0.064 | 11.3 | 764.3 | 0.030 | 81.1 |

Lengths that are not multiples of 32 (77, 197) run with no padding by the user. Time is
flat from 32 to 1024 tokens (host dispatch bound) and follows compute above: Qwen2.5-3B
114 / 185 / 344 ms at 1024 / 2048 / 4096 tokens. Batch 2, 4 and 8 of 77 tokens cost 75
to 82 ms on Qwen2.5-0.5B (tt-xla batch 4: 213 ms). A padded batch with an attention
mask costs about twice the unmasked call (Qwen2.5-0.5B 116 ms, tt-xla 211 ms).

### Prefill: eager, compiled, captured (Kohaku-TT)

| Model | Tokens | Eager ms | `torch.compile(backend="ktt_tt")` ms | `kohakutt.capture` ms |
|---|---|---|---|---|
| Qwen2.5-0.5B | 77 | 54.7 | 25.7 | 21.2 |
| Qwen2.5-0.5B | 197 | 56.9 | 34.7 | 29.1 |
| Qwen2.5-0.5B | 512 | 56.6; now 62.2 | | 47.5; now 30.7 |
| Qwen2.5-3B | 77 | 111.6 | 65.7 | 58.7 |
| Qwen2.5-3B | 512 | 111.9; now 94.3 | fails on a second length | 119.5; now 70.4 |

A captured forward is bit-identical to eager. tt-xla compiled: Qwen2.5-0.5B at 77
tokens 113.5 ms (lazy 220.5).

### Greedy decode, 77-token prompt, static KV cache

| Model | Kohaku-TT eager tok/s | Kohaku-TT compiled tok/s | tt-xla tok/s | Text equal to CPU |
|---|---|---|---|---|
| Qwen2.5-0.5B | 7.8; now 14.9 | 35; now 120 | 2.9 | yes |
| Qwen2.5-1.5B | 4.6 | 21.0 | 2.5 | yes |
| Qwen2.5-3B | 3.9; now 9.7 | 14.5; now 38.1 (to be revisited) | 2.0 | yes |
| Qwen3-0.6B | 3.8 | 19.1 | 2.5 | yes |
| Qwen3-1.7B | 4.5; now 9.4 | 15.4; now 55.5 | 2.5 | yes |
| Gemma-4 E2B | 4.0 | wrong from the 6th token | not run (first call 142 s) | eager: yes |
| Gemma-4 E4B | 2.9 | | did not finish in 600 s | differs at two bf16 ties |
| Gemma-4 12B | 2.2 | | | yes |

tt-xla's compiled static-cache decode fails to build (`ttir.fill_cache`). With
Hugging Face's default growing cache, Kohaku-TT builds one attention program per new
KV length: about 1 s per new token position and about 30 s for a new prompt length
the first time, then cached on disk (0.3 to 0.4 s per token in a later process).
`model.generate()` warm: 79 ms per token (Qwen2.5-0.5B, default cache).

### Gemma-4 12B on one card (32 GB)

| Prompt tokens x batch | First call s | ms |
|---|---|---|
| 32 | 8.2 | 302 |
| 197 | 11.3 | 355 |
| 512 | 21.3 | 347 |
| 1024 | 29.6 | 484 |
| 2048 | 14.2 | 864 |
| 4096 | 36.5 | 1855 |
| 197 x 4 | 28.8 | 523 |
| 197 x 8 | 43.8 | 978 |
| 8192, `logits_to_keep=1` | 24.5 | 3935 |
| 8192, full logits | | out of memory (the 4.29 GB logits) |

## Training

| Task | CPU | Kohaku-TT | tt-xla |
|---|---|---|---|
| Transformer from scratch (4 layers, d 256), 200 AdamW steps: final loss | 0.0090 | 0.0090 | wrong from the first step, then inf |
| Same, ms per step | 417 | 49.6 (fused AdamW 33.6, captured 28.3) | 7700 |
| Qwen2.5-0.5B LoRA (q, v, r 8): ms per step | 43000 | 271 (captured 69); now 245 (captured 102, to be revisited) | 82600, follows CPU |
| Qwen2.5-0.5B last two layers: ms per step | 150000 | 109 (captured 34) | 13400, follows CPU |

Kohaku-TT's loss curves follow CPU's: the 200-step curve to three digits, the LoRA
curve within bf16 rounding. tt-xla's failure is its fp32 `layer_norm`, wrong for most
inputs (a hand-written normalisation is exact); Qwen and Gemma use RMSNorm.

Training step of vision models on Kohaku-TT (forward and backward, 224 px, batch 1):

| Model | ms | Loss and gradients against CPU fp32 (CPU bf16) |
|---|---|---|
| DINOv3 ConvNeXt-tiny | 95 | 7.3e-2 (7.8e-2) |
| facebook/convnext-tiny-224 | 93 | 3.6e-2 |
| DINOv3 ViT-S+/16 | 136 | 0.79 (0.15): see problems |
| DINOv3 ViT-B/16 | 127 | 0.84: see problems |
| pixai-tagger-v1.0, 1008 px | 1583 | 0.23 |
| sam3 vision encoder, 1008 px | 4174 | no reference |

## Capture and compile on vision models (Kohaku-TT, ms)

| Model | Batch | Eager | `ktt_tt` compile | `kohakutt.capture` |
|---|---|---|---|---|
| DINOv3 ViT-S+/16 | 1 | 37.5 | 56.6 | 13.0 |
| DINOv3 ViT-B/16 | 8 | 50.4 | 56.6 | 55.1 |
| DINOv3 ConvNeXt-tiny | 1 | 28.7 | 35.9 | 12.2 |

## Attention call shapes

`F.scaled_dot_product_attention`, bf16, 14 heads (2 key heads for GQA), head dim 64;
every inference call is correct (error 0.004 to 0.011 against CPU fp32).

| Call | ms | Route |
|---|---|---|
| causal or unmasked, MHA or GQA, S 32 to 512 | 0.18 to 0.31 | flash kernel |
| decode, one query, no mask | 0.19 to 0.24 | flash kernel |
| bool key-padding mask `[B, 1, 1, S]` | 1.3 to 2.3 | off the flash kernel |
| causal with Sq != Sk | 3.8 to 4.2 | math composite |
| fp32, or head dim 80 | 2.1 to 2.4 | math composite |
| training, GQA causal, S 64 / 77 / 197 | 1.6 / 7.6 / 8.0 | forward and backward |

## Where the time goes (per-op profile, Kohaku-TT)

| Case | Largest shares |
|---|---|
| DINOv3 ViT-B/16, 1008 px | `mul` 43 %, attention 12 %, `linear` 11 %, `neg` 10 %: the rotary embedding written as slice / neg / cat / mul / add is 59 % |
| sam3 vision encoder | `mul` 21 %, casts 16 %, `neg` 14.5 %, convolution 13 % |
| pixai-tagger | fp32 casts of its complex rotary embedding 38 %, attention 18 %, `linear` 13 % |
| DINOv3 ConvNeXt-tiny, batch 8 | `linear` 34 %, convolution 27 %, layer norm 13 % |
| SDXL UNet, 512 px | group norm 24 %, `linear` 19.5 %, convolution 13 %, attention 9 % |
| sdxl-vae decode, 1024 px | convolution 72 %, nearest upsample 13 %, group norm 7.7 % |
| Qwen2.5-0.5B, one decode token | 1682 ops: KV-cache write (`index_copy_`) 20 %, `mul` 10 %, `linear` 9 %, `add` 7 %, attention 4 % |

## tinygrad frontend

| Model | Result |
|---|---|
| GPT, d 128, 2 layers, 20 AdamW steps | runs with no host kernel; loss 6.438 to 5.696 (tinygrad CPU 6.440 to 5.522) |
| GPT, d 256, 4 layers | 2132 ms per step eager, 148 ms with `TinyJit` |
| Convnet (3 conv, 2 batch norm, linear), 5 steps | 560 to 600 ms per step (tinygrad CPU 1250 to 1700); first loss 7 % above CPU's |

## Problems found

Kohaku-TT:

- ViT training gradients are far from CPU (0.79 to 0.84 overall against 0.15 for CPU
  bf16), concentrated in the first block and the patch embedding; ConvNeXt gradients
  are at bf16 level.
- Compiled (`ktt_tt`) Gemma-4 E2B now matches eager (teacher-forced decode equal at
  every step; prefill at 77 tokens 0.057 against 0.055); its decode runs eagerly where
  the position ints change (see [limitations](limitations.md)).
- `ktt_tt` compile of Qwen2.5-3B fails on a second sequence length; `ktt_tt_aot` fails
  on attention with grouped heads at ratio 7 (a LoRA step of Qwen2.5-0.5B).
- `kohakutt.capture` fails when the Hugging Face forward is called with
  `use_cache=False`, and for Gemma.
- `torch.compile` is slower than eager on ViT and ConvNeXt inference; capture is the
  fast form.
- Changing `KTT_FLASH` inside a running process is supported: each call reads it when
  it is queued and a backward follows its forward's kernel. The wrong masked forward
  seen then was TT-NN's masked attention at lengths not a multiple of 32, now fixed;
  the segfault was not reproduced.
- VideoMAE base fails on a gather outside the gather kernel's coverage.
- A new KV length or prompt length builds a new attention program (see decode).
- sdxl-vae encode and the sam3 encoder have errors near 0.1, unattributed; the
  tinygrad convnet's first loss is 7 % off.
- A bool key-padding mask leaves the flash kernel in real model code.

tt-xla 1.4.0:

- Lazy mode gives wrong or non-finite outputs for every vision model tried.
- fp32 `layer_norm` is wrong unless the input is zero-mean, unit-scale with affine
  weights; a LayerNorm transformer does not train.
- The diffusers VAEs are wrong in compiled mode; the SDXL UNet compile did not finish.
- ConvNeXt models take 1.4 to 12 s per forward, nearly independent of batch.
- Parallel jobs share one descriptor file in the temp directory and fail with
  `Error code: 13`; a private `TMPDIR` per job avoids it.
