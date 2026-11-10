# tinygrad examples

Every example is ordinary tinygrad: `import kohakutt.tinygrad` registers the device
`"KTT"` (chip 0), then tensors and models go there with `device="KTT"` or `.to("KTT")`.
They need tinygrad v0.14.x importable and the same build and TT-Metal setup as the torch
examples (the torch extension carries the bridge both frontends share).

| file | shows | prints |
|---|---|---|
| `01_tensors.py` | tensors, math, slicing, padding, concatenation, views, cumsum, gradients | the values of each step and the largest difference to the CPU backend for `tanh` |
| `02_training.py` | a small MLP trained with tinygrad's AdamW: an unchanged tinygrad training loop | first-batch loss and gradients against the CPU backend, loss every 10 steps, eval accuracy, logits of the trained weights against the CPU backend (relative) and the share of equal predictions |
| `03_jit.py` | a transformer block (causal attention + MLP) under `TinyJit`: two tracing calls, then replays | per call, the largest difference to the CPU output relative to its largest value |

```bash
export TT_METAL_RUNTIME_ROOT=$TT_METAL      # see the top-level README for the setup
export PYTHONPATH=/path/to/tinygrad-v0.14.0:$PWD/src
python examples/tinygrad/01_tensors.py
python examples/tinygrad/02_training.py
python examples/tinygrad/03_jit.py
```

Results match the CPU backend up to the card's default matmul precision (bf16-level,
about 1e-2 relative for matmul-heavy code; `with kohakutt.tinygrad.numerics(fidelity=3):`
narrows it). `Tensor.randn(..., device="KTT")` is drawn on the card: reproducible under
`Tensor.manual_seed`, but not the CPU's numbers for the same seed
(`KTT_TINYGRAD_RAND=host` draws those, on the host). The examples make their weights on
the CPU and move them, so both backends start from the same parameters.
