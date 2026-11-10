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
| `max_pool2d`, `avg_pool2d` with stride = kernel, no padding, sizes divisible by the kernel | reshape + max / mean, forward and backward |
| other 2-D `max_pool2d` / `avg_pool2d` on fp32 / bf16 (any kernel, stride, padding, dilation, `ceil_mode`, `count_include_pad`) | one kernel of strided slices per window tap; the backward adds each window's share back into the input with one scatter-add ([pooling](#pooling)) |
| `Tensor.rand` (fp32 / bf16), and through it `randn`, `uniform`, `randint`, `dropout` | drawn on the card ([random numbers](#random-numbers)) |

A kernel the reader does not take runs on the host (numpy) and is counted; the counts
are in `kohakutt.tinygrad_backend.executor.FALLBACKS` by reason. The small models
shipped as checks (MLP, convnets with 2x2 and with 3x3 stride-2 padded pooling, ViT,
GPT; forward and AdamW training) run with no host kernel.

## Pooling

A pool over the last two dims whose windows overlap, are padded, dilated or do not
divide the input is one kernel: the input is padded (with the dtype's lowest value for
max, zero for average), each of the kernel's taps is one strided slice, and the taps are
combined by `maximum` or by a sum divided by tinygrad's window count (all cells,
`count_include_pad=False`: the input cells only, `ceil_mode`: without the extra cells).
The gradient kernel follows tinygrad's rule: a max window's gradient is split evenly
among the taps equal to its maximum, an average window's is divided by its count. Each
tap's shares are added back at the padded positions they were read from by one
scatter-add (`ktt_scatter`), and the padding is cut off. `return_indices=True`, integer
dtypes, other dims and `KTT_TINYGRAD_OVERRIDES=0` keep tinygrad's formulation.

Measured on the card against the CPU backend (fp32, 2 x 16 x 32 x 32, no host kernel):
max pooling, forward and gradients, is equal for kernel 3 stride 2 padding 1, kernel 2
stride 1, kernel 2 stride 2 padding 1, kernel 2 on a 31 x 31 input, kernel 3 stride 1
padding 1 dilation 2, kernel 3 stride 2 `ceil_mode`, and with ties (after `relu`);
average pooling is within 1.1e-7 (forward) and 1.7e-7 (gradients) relative. The
convnet with 3x3 stride-2 padded max and average pooling trains for three AdamW steps in
249 card kernels and none on the host, losses 2.3796, 2.1610, 1.9902 against the CPU
backend's 2.3804, 2.1617, 1.9891.

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
threefry kernel on uint64 words runs on the host (counted fallbacks `dtype uint32` and
`op BITCAST`) and its bits are uploaded. Other float dtypes, other devices and
`KTT_TINYGRAD_OVERRIDES=0` take that route too (and `KTT_TINYGRAD_DTYPES=error` stops
it: its kernels are on uint32).

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

## Dtypes

The card holds float32, bfloat16 and int32. A tensor of any dtype can be made on "KTT",
moved there and read back (`Tensor(array, device="KTT")`, `.to("KTT")`, `.numpy()`):
its bytes are kept as they are, and no kernel runs. A kernel with a buffer or a value of
another dtype that the card cannot keep runs on the host (numpy) with the CPU backend's
results; it is counted in `executor.FALLBACKS["dtype <name>"]` and the first one of each
dtype warns (`kohakutt.tinygrad.DtypeWarning`). `KTT_TINYGRAD_DTYPES=error` raises
`kohakutt.tinygrad.DtypeError`, naming the dtype and the device dtypes, instead of
running it. A failure of the host evaluator itself raises
`kohakutt.tinygrad.HostKernelError`, naming the kernel and why it left the card.

| dtype | kernels on "KTT" |
|---|---|
| float32, bfloat16, int32 | on the card (bfloat16 kernels that fall back for another reason run on the host in bf16) |
| float16, float64 | on the host; a float16 matmul sums in float16 there (about 1e-3 relative from the CPU backend) |
| int8, int16, int64, uint8, uint16, uint32, uint64 | on the host; a float cast to an unsigned type out of its range is undefined and differs from the CPU backend |
| bool | a comparison or mask used inside one kernel (`where`, a sum, `tril` masks) on the card; a kernel that stores or loads a bool buffer on the host |
| fp8 (`float8_e4m3`, `float8_e5m2` and their fnuz forms) | `DtypeError`: neither a device dtype nor modelled by the host evaluator |

Common computations, measured on the card against the CPU backend:

| computation | runs on |
|---|---|
| `(x > 0).where(x, 0)` and other comparisons used in the same kernel | the card |
| accuracy `(logits.argmax(1) == labels).mean()`, int32 labels | the card |
| `sparse_categorical_crossentropy` with int32 labels | the card |
| `Tensor.ones(t, t).tril().cast(dtypes.bool)` used as a mask | the card |
| `x[index]` with an int32 index | the card |
| `x > 0` kept as a tensor | the host (`dtype bool`) |
| a numpy bool array uploaded and used in `where` / `masked_fill` | the host (`dtype bool`) |
| int64 labels or indices (numpy's default integer) | the host (`dtype int64`) |
| float16 matmul | the host (`dtype float16`) |

`Tensor([0, 1, 2])` from Python integers is int32; a numpy integer array is int64 unless
cast (`Tensor(labels.astype(np.int32))` keeps those kernels on the card).

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
- Device dtypes are float32, bfloat16 and int32; kernels on other dtypes run on the
  host or raise `DtypeError` ([dtypes](#dtypes)).
- tinygrad's memory planner is off (`NO_MEMORY_PLANNER=1` is set on import): every
  buffer keeps its own device allocation, so peak device memory is higher than tinygrad
  would plan.
- torch must be installed even for tinygrad-only code: `import kohakutt.tinygrad`
  imports the `kohakutt` package, which imports torch, and the add-on calls the card
  through the C functions of the torch extension `_ktt_native.so`, loaded into torch.
- One card (chip 0); no mesh.
- Pooling outside the forms above (integer dtypes, `return_indices=True`, 1-D or 3-D
  windows) keeps tinygrad's formulation; its backward kernel, which adds each window's
  gradient back into the input, is readable only through the gather route (at most
  262144 gathered elements) and above that runs on the host (fallback `gather of more
  than 262144 elements`).
