# MOSS-Image on tt

Text-to-image with [MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base),
written as plain PyTorch modules and run on a Tenstorrent card: a Qwen3.5 text encoder
(gated DeltaNet + attention layers), a 42-block DiT sampled with Euler flow matching and
classifier-free guidance, and the FLUX.2 VAE decoder. No model library is used for compute
(the tokenizer comes from `transformers`); every op is an aten op with a tt kernel.

```bash
# one image
python examples/06_moss_image/generate.py --model /path/to/MOSS-Image-Base \
    --prompt "a lighthouse on a rocky coast at sunset" --size 1024 --out out.png

# a sweep: every prompt file x cfg x seed, one process per chip
for chip in 0 1; do
    TT_VISIBLE_DEVICES=$chip python examples/06_moss_image/generate.py \
        --model /path/to/MOSS-Image-Base --prompts prompts/ --negative negative.txt \
        --cfgs 3 4.5 6 --seeds 0 1 --shard $chip/2 --out-dir images/ &
done; wait
```

`--model` is the checkpoint directory in the Hugging Face layout (`text_encoder/`,
`transformer/`, `vae/`, `tokenizer/`). A prompt file sets its image size with a first
line `WxH` or a `canvas=WxH` anywhere in the text; otherwise `--size` / `--width` /
`--height` apply.

The models load once. The text encoder encodes every prompt (and the negative prompt)
in one batch and is freed; then the DiT and the VAE load. Each prompt's (cfg, seed) jobs
sample as one batch, and every image is decoded and saved as soon as its batch is done:
a PNG, its latent (`.pt`) and its settings (`.json`).

| option | default | meaning |
|---|---|---|
| `--prompts DIR` | | every `*.txt` in DIR is a prompt |
| `--negative TEXT_OR_FILE` | empty | the negative prompt |
| `--cfgs`, `--seeds` | 3.0, 0 | guidance scales and seeds; every combination runs |
| `--steps` | 28 | sampler steps |
| `--batch` | all | most (cfg, seed) rows per DiT batch |
| `--weights` | `bfp8` | DiT and text encoder linear weights: `bfp8` (BFLOAT8_B) or `bf16` |
| `--te-device` | `tt` | `cpu`: run the text encoder in fp32 on the host |
| `--text-cache FILE` | | text states read from FILE when it holds every prompt, else written there after encoding (shards of a sweep share one) |
| `--shard i/n` | `0/1` | this process's share of the jobs |
| `--fidelity`, `--approx`, `--attention` | HiFi2, off, exact | numerics scope for the run ([numerics](../../docs/design/numerics.md)) |
| `--compile` | off | the DiT trunk through `torch.compile` |

## Files

| file | contents |
|---|---|
| `text_encoder.py` | Qwen3.5 text model: gated attention (GQA, qk-norm, partial RoPE), gated DeltaNet (causal conv, chunked delta rule with a triangular solve), SiLU MLP |
| `dit.py` | MOSS DiT: text refinement blocks, joint image + text blocks with shared modulation, qk-RMSNorm, 4-axis RoPE, SiTU FFN |
| `vae.py` | FLUX.2 VAE decoder: batch-norm de-normalization, 2 x 2 unpatchify, ResNet / attention / upsample blocks |
| `generate.py` | the pipeline: encode, sample, decode, save |

## Precision

| part | weights | activations | math inside ops |
|---|---|---|---|
| text encoder | bfp8 | bf16 | fp32 vector math |
| DiT | bfp8 | bf16 (residual stream included) | fp32 inside `ktt.rms_norm`, `ktt.rotary` and the fused SiTU / gated-residual chains |
| VAE | bf16 | bf16 | fp32 group norm |

Matmuls accumulate in fp32 at HiFi2. Error of the DiT against fp32 on the CPU is in
[performance](../../docs/design/performance.md).

## Speed (one P150A)

| | time |
|---|---|
| DiT forward, 1360 x 768, 1615 text tokens | 0.652 s |
| CFG step (cond + uncond), 1024 x 1024, 300 text tokens | 0.935 s |
| CFG step, 1360 x 768, 1615 text tokens | 1.104 s |
| VAE decode, 1312 x 800 (repeat size / first decode at a size) | 1.1 s / about 1 min |
| text encoder, 5 prompts up to 1615 tokens (tt) | about 1 min |

The first decode at a new image size pays TT-NN's host setup for new shapes
([TT-NN issues](../../docs/reference/ttnn-issues.md)); later images of that size decode
warm.

## Guidance

The DiT in bf16 carries a noise floor of a few percent per forward (any rounding
difference moves the velocity by 4.5% on the CPU in bf16, 12% on tt), and
classifier-free guidance scales it with the guidance scale
([performance](../../docs/design/performance.md)). cfg 1.5-4.5 with an empty negative
prompt gives clean images; higher scales and long negative prompts show oversaturation.
