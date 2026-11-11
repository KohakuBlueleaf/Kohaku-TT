# Examples

Every example is ordinary PyTorch: `import kohakutt`, take a chip as a torch device
(`dev = kohakutt.device(0)`, or `kohakutt.mesh([0, 1])` for two), then `.to(dev)` as usual.

| file | shows |
|---|---|
| `00_hardware.py` | the cards on this host (read-only), chip 0 as `tt:0`, its compute grid |
| `01_tensors.py` | tensors, math, indexing, views, in-place updates, autograd |
| `02_training.py` | a CNN trained with AdamW: an unchanged training loop |
| `03_trace.py` | capture a whole training step once, replay it as one device trace |
| `04_compile.py` | `torch.compile(model, backend="ktt_tt_aot")` for training |
| `05_mesh.py` | two chips as one device (`kohakutt.mesh([0, 1])`): sharded weights, placement marks, collectives, a captured replay |
| `06_moss_image/` | MOSS-Image text-to-image (text encoder, DiT sampler, VAE) as plain PyTorch modules on `tt` |
| `07_sdxl/` | the SDXL UNet and VAE decoder as plain PyTorch modules loading a diffusers checkpoint, checked against diffusers on the CPU, run on `tt` |
| `08_hf_models/` | Hugging Face models as they come (`transformers`, `diffusers`), one short script per use: vision encoder inference and training, VAE, a tagger, language-model prefill, static-cache decoding, capture, LoRA fine-tuning |
| `tinygrad/` | the same chip from tinygrad (`import kohakutt.tinygrad`, device `"KTT"`): tensors, training, `TinyJit` |
| `kernels/` | your own kernels as torch ops, three ways (see its README): hand-written TT-Metalium from Python (`metalium/`), the kernel DSL's metal level (`dsl_metal/`), a C++ extension library (`cpp_extension/`) |

```bash
export TT_METAL_RUNTIME_ROOT=$TT_METAL      # see the top-level README for the setup
python examples/01_tensors.py
```
