# Hugging Face models on tt

Models as they come from `transformers` and `diffusers` (their own classes, and remote
code for the tagger), moved to the chip with `model.to(dev)`. One short script per use;
each loads its weights by repo id from the local Hugging Face cache (`HF_HOME`), runs on
chip 0 in bf16 and checks itself against the CPU where that is cheap.

```bash
cd examples/08_hf_models
python vision.py --compile          # DINOv3 ViT-S+: eager, captured, compiled
python vision_train.py              # ConvNeXt-T: training steps with AdamW
python vae.py                       # sd-vae-ft-mse: encode and decode 512 px
python tagger.py                    # PixAI tagger v1.0, 1008 px
python lm_prefill.py                # Qwen2.5-0.5B prefill: eager and compiled
python lm_capture.py                # the same prefill captured and replayed
python lm_decode.py                 # Qwen3-0.6B greedy decoding, static KV cache
python lm_decode.py --compile
python lora.py                      # Qwen2.5-0.5B LoRA fine-tuning
python lora.py --last 2             # ... or the last two decoder layers
python profile_ops.py               # the per-op time of one forward
```

Every script takes `--model <repo id>`; the defaults are the smallest models the
examples were run with. Times printed are a first call (which builds the device
programs, cached on disk for later runs) and the median of three more calls, each
synchronized. Measured numbers for these models and their conditions are in
[results](../../docs/reference/results.md).

| file | shows | prints |
|---|---|---|
| `vision.py` | an image encoder (`AutoModel`: DINOv3 ViT / ConvNeXt, `facebook/convnext-*`) eager, under `kohakutt.capture`, and with `--compile` under `torch.compile(backend="ktt_tt")` | per mode: first call, median ms, relative L2 error against the fp32 CPU output |
| `vision_train.py` | forward, backward and AdamW on an image encoder | the first step's loss next to the CPU's, the gradient error against an fp32 CPU step, then loss and ms per step |
| `vae.py` | a diffusers `AutoencoderKL`: `encode(...).latent_dist.mode()` and `decode(...).sample` | time and error for encode and decode |
| `tagger.py` | the PixAI tagger's own remote code (`stubs.py` stands in for `timm` / `torchvision` when they are not installed) | time, error, how many of the CPU's top-10 tags the chip also has |
| `lm_prefill.py` | `AutoModelForCausalLM` over a prompt of random tokens, eager and `torch.compile(backend="ktt_tt")` | time, logits error, the share of positions whose argmax equals the CPU's |
| `lm_capture.py` | the prefill captured once and replayed as a device trace, then replayed with new token ids | eager and replay times, replay error against eager |
| `lm_decode.py` | a greedy loop with `StaticCache`, `position_ids` and `cache_position` on the device; `--compile` compiles the model | prefill and first-step time, ms per token and tokens/s, how many tokens equal the CPU's (same bf16 weights), both texts |
| `lora.py` | LoRA adapters (rank 8) on `q_proj` / `v_proj`, or `--last N` decoder layers trainable, AdamW, one fixed batch | trainable parameters, loss and ms per step, the CPU's first loss |
| `profile_ops.py` | `kohakutt.tools.profile.OpProfile` around one forward | the top aten overloads by time, calls, programs per call |
| `common.py` | the shared helpers: seeded inputs, relative error, a synchronized timer | |

## Limitations you will meet

- **Capture and `use_cache=False`.** `kohakutt.capture` of a causal LM forward called
  with `use_cache=False` fails with `host download inside a trace capture`: that path
  builds the mask with a host read. Keep the default `use_cache` under capture
  (`lm_capture.py`); eager and compiled prefill take `use_cache=False` fine.
- **Use a static KV cache for decoding.** With the default `DynamicCache` (also what
  `model.generate()` uses unless told otherwise) the KV length grows by one every token,
  and every new length is a new shape: the attention program is built again, about a
  second per token the first time a length is seen. `StaticCache` (or
  `generate(..., cache_implementation="static")`) keeps the shapes fixed, so only the
  first steps build programs.
- **`torch.compile` is not faster than eager on the vision models.** For a single image,
  DINOv3 and ConvNeXt run about as fast compiled as eager (or slower), while
  `kohakutt.capture` replays several times faster. The same holds for a single prefill
  (`lm_prefill.py`: try `lm_capture.py` instead). Compilation pays off on the decode
  loop, where each step is a small forward and eager time is per-op host work.
- **First calls are slow.** The first call of every new shape builds device programs
  (seconds, up to a minute for a first training step); they are cached on disk, so a
  second run of the same script starts much faster.
- **The PixAI tagger takes 1008 px only.** Its remote code asserts a fixed 1008 x 1008
  grid; other sizes fail in the model code. At that size (5184 tokens) a forward takes
  seconds, and much of it goes to the model's complex-number rotary embedding rather
  than to matmuls or attention.
- **Tokenizers.** `lm_decode.py` needs the model's tokenizer files in the cache; the
  other language-model scripts use random token ids and need only the weights.
