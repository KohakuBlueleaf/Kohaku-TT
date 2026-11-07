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
| `05_mesh.py` | two chips as one device (`kohakutt.mesh([0, 1])`, tensors replicated) |
| `06_moss_image/` | MOSS-Image text-to-image (text encoder, DiT sampler, VAE) as plain PyTorch modules on `tt` |
| `custom_kernel/` | how to write a tt kernel: a hand-written TT-Metalium program (SwiGLU) as a torch op via `kohakutt.kernels`, conventions, checks |
| `custom_kernel_cpp/` | the same kernel as a C++ extension library (`ktt/extension/*.hpp`, `kohakutt_add_extension`) |

```bash
export TT_METAL_RUNTIME_ROOT=$TT_METAL      # see the top-level README for the setup
python examples/01_tensors.py
```
