---
title: tinygrad add-on
summary: The card as the tinygrad device "KTT": what runs where, numerics, how to compare against the CPU backend, and the current limits.
tags:
  - tinygrad
  - frontend
---

# tinygrad add-on

`import kohakutt.tinygrad` registers chip 0 as the tinygrad (v0.14.x) device `"KTT"`.
Code stays plain tinygrad: tensors go there with `device="KTT"` or `.to("KTT")`, and
`Tensor`, `nn`, `nn.optim`, `Context(TRAINING=1)` and `TinyJit` work unchanged. Runnable
examples: [examples/tinygrad](../examples/tinygrad/README.md).

## What runs where

Each tinygrad kernel is read as a graph of bridge operations (the same TT-NN ops and
Kohaku-TT programs the torch backend uses) and queued on the card. Library-level calls
are replaced when their arguments fit (`KTT_TINYGRAD_OVERRIDES=0` turns this off):

| tinygrad call | on the card |
|---|---|
| `scaled_dot_product_attention` (no dropout; `is_causal` or no mask) | fused attention, forward and backward; fp32 inputs are cast to bf16 for it |
| `Tensor.layernorm`, `nn.RMSNorm` | fused norm, forward and backward |
| `conv2d` (4-D, fp32 / bf16) | TT-NN conv; the backward kernels take bf16 |
| `max_pool2d` with stride = kernel, no padding, sizes divisible by the kernel | reshape + max, forward and backward |
| `Tensor.rand` (fp32 / bf16), and through it `randn`, `uniform`, `randint`, `dropout` | drawn on the card ([random numbers](#random-numbers)) |

A kernel the reader does not take runs on the host (numpy) and is counted; the counts
are in `kohakutt.tinygrad_backend.executor.FALLBACKS` by reason. The small models
shipped as checks (MLP, convnet, ViT, GPT; forward and AdamW training) run with no host
kernel.

## Random numbers

`Tensor.rand(..., device="KTT")` in float32 or bfloat16 is one draw of the card's
uniform generator (TT-NN's per-core generators; Philox4x32-10 on each element's index
under `KTT_PHILOX=1`). `randn` (Box-Muller), `uniform`, `randint`, `normal`, the
initialisers built on them and `dropout` run their arithmetic on the card as well, and
`dropout` keeps its mask inside one kernel. No host kernel runs and nothing is uploaded.

Each draw is keyed by a hash of `Tensor.manual_seed`'s seed and the number of draws run
since it was set, taken when the kernel runs: the same program after the same
`manual_seed` gives the same numbers, and a `TinyJit` replay draws new ones. The values
are not the CPU backend's for the same seed. `KTT_TINYGRAD_RAND=host` (read at each
call) runs tinygrad's own generator instead, which gives the CPU backend's stream: its
threefry kernel on uint64 words runs on the host (counted fallbacks `dtype dtypes.uint`
and `op BITCAST`) and its bits are uploaded. Other float dtypes, other devices and
`KTT_TINYGRAD_OVERRIDES=0` take that route too.

Measured on the card (one draw after a warm-up, until the result is on the device;
statistics of the 4096 x 4096 draws):

| call | on the card | `KTT_TINYGRAD_RAND=host` |
|---|---|---|
| `rand(1024, 1024)` | 3.8 ms | 101 ms |
| `randn(1024, 1024)` | 6.0 ms | 224 ms |
| `rand(4096, 4096)` | 2.5 ms | 1765 ms |
| `randn(4096, 4096)` | 10.0 ms | 3533 ms |
| `uniform(4096, 4096, low=-2, high=3)` | 4.1 ms | 1835 ms |
| `dropout(0.25)` of 4096 x 4096 | 4.9 ms | 2910 ms |

`rand`: mean 0.4999, standard deviation 0.2887, range [0, 1) (bfloat16: largest value
0.9961). `randn`: mean 0.0003, standard deviation 1.0001, range -5.2 to 5.2.
`dropout(0.25)`: share of zeros 0.2499, kept values 1 / 0.75.

## Numerics

Results match tinygrad's CPU backend up to the card's own precision: fp32 matmuls run at
fidelity HiFi2 by default (about 1e-2 relative), elementwise and reductions are about
1e-7. `with kohakutt.tinygrad.numerics(fidelity=3):` raises the matmul fidelity for the
kernels queued inside it. Fidelity does not change the bf16 kernels (fused attention, the
conv backward), so models that use them keep a bf16-sized error (about 4e-3 relative)
at any fidelity. Measured on the card, ViT (d=128, 2 blocks, patch conv), error
against the CPU backend relative to the largest reference value:

| fidelity | forward | gradients, median over parameters | gradients, worst parameter |
|---|---|---|---|
| 1 (HiFi2, default) | 9.1e-3 | 1.3e-2 | 2.6e-2 |
| 2 (HiFi3) | 4.3e-3 | 2.9e-3 | 5.1e-3 |
| 3 (HiFi4) | 3.4e-3 | 2.5e-3 | 6.6e-3 |
| 3, overrides off (fp32 attention and conv) | 2.7e-3 | 2.4e-3 | 4.0e-3 |

The error falls with fidelity; what remains at fidelity 3 is the bf16 attention and conv
kernels and the device's fp32 arithmetic.

### Comparing a training run against the CPU backend

Compare losses or gradients, not parameters after an Adam step. Adam divides each
gradient element by its own running magnitude, so an element whose true gradient is zero
or near zero moves by about `lr` in the direction of its rounding noise, on any backend.
The attention key bias is one such parameter (softmax ignores a shift shared by a row).
Measured on the card, GPT (d=128, 2 blocks), 2 steps against the CPU backend:

| optimizer | largest weight difference / largest weight |
|---|---|
| SGD, lr 0.1 | 2.4e-2 |
| AdamW, lr 1e-2 | 1.2 (elements stepped `lr` in opposite directions) |

The step-1 gradients differ by 0.4e-2 to 3.4e-2 of their largest element, and sign flips
occur only in elements below 1.2e-2 of it. The same AdamW comparison on the host executor
(numpy, fp32, gradients equal to 1e-6) still differs by 0.27, all in the key bias.
[examples/tinygrad/02_training.py](../examples/tinygrad/02_training.py) prints both
checks that hold: first-batch loss and gradients, and the trained weights' logits on
both backends.

## Limitations

- Random numbers drawn on the card are not the CPU backend's stream for the same seed
  (`KTT_TINYGRAD_RAND=host` gives it, at host speed).
- Device dtypes are float32, bfloat16 and int32. Kernels on float16, int64 and other
  dtypes run on the host (counted fallbacks) or fail.
- tinygrad's memory planner is off (`NO_MEMORY_PLANNER=1` is set on import): every
  buffer keeps its own device allocation, so peak device memory is higher than tinygrad
  would plan.
- torch must be installed even for tinygrad-only code: `import kohakutt.tinygrad`
  imports the `kohakutt` package, which imports torch, and the add-on calls the card
  through the C functions of the torch extension `_ktt_native.so`, loaded into torch.
- One card (chip 0); no mesh.
- Overlapping, padded or non-dividing `max_pool2d` (stride < kernel, padding, a size
  the kernel does not divide) keeps tinygrad's own formulation. One backward kernel,
  which adds each window's gradient back into the input (overlapping, gated reads), is
  readable only through the gather route, which is capped at 262144 gathered elements;
  already a 2 x 16 x 32 x 32 input exceeds it, so the kernel runs on the host (fallback
  `gather of more than 262144 elements`).
