---
title: Numerics
summary: "The math settings tt ops run with, how a program chooses them, and how they are checked."
tags:
  - concepts
  - numerics
---

# Numerics

The math settings tt ops run with, how a program chooses them, and how they are checked.

## Settings

| Setting | Values | Acts on |
|---|---|---|
| `fidelity` | `LoFi` < `HiFi2` < `HiFi3` < `HiFi4` | the passes of the FPU's multiplies: matmul, linear, attention, convolution (as matmuls), FPU reductions and norms |
| `approx` | off, on | the SFPU's approximations: exp, log, sqrt, rsqrt, tanh, sigmoid, and the exp of attention's softmax |
| `strict` | off, on | on: kernel requirements are ignored, every kernel runs the scope's values |
| `attention` | `exact`, `centered`, `centered_bfp8` | `scaled_dot_product_attention`'s Q / K: as given; K minus its mean over the sequence (each query's scores shift by one constant, which softmax removes); centered K and Q / K stored as bfp8 |

Always on, not settings: fp32 accumulation in the destination registers; fp32 unpack to
the destination registers for fp32 inputs (the FPU's source registers hold about 19
bits).

The FPU multiplies a 5-bit slice of one operand's mantissa by a 7-bit slice of the
other's per pass. `LoFi` takes one pass (the high slices), `HiFi2` adds the low slice of
the first operand, `HiFi3` and `HiFi4` add the low slices of the second. A bf16 operand's
mantissa (7 bits and the implicit one) fits the 7-bit slice, so HiFi2 covers bf16 x bf16
products; the higher fidelities matter for fp32 operands and for block-float operands
with long shared-exponent ranges.

## Scopes

```python
with kohakutt.numerics(fidelity="HiFi3", approx=False, strict=False):
    ...

remove = kohakutt.numerics_for(module, fidelity="LoFi")  # wraps module.forward
kohakutt.current_numerics()  # Numerics(fidelity="HiFi2", approx=False, strict=False)
```

- A scope is per thread and nests; settings left out are inherited.
- Autograd runs the gradient ops of `backward()` on its own device thread. A thread in a
  backward pass with no scope of its own reads the most recently opened scope of any thread,
  so `loss.backward()` called inside a scope runs its gradient ops with that scope. Other
  threads without a scope read the process default.
- An op reads the scope on the calling thread when it is queued and carries the value to
  the stream worker, so work queued in a scope runs with it after the scope closed.
- The process default is HiFi2, exact; `KTT_FIDELITY` (alias `KTT_MATMUL_FIDELITY`) and
  `KTT_APPROX=1` change it.
- `kohakutt.capture` keeps the scope it captured under; `torch.compile` (`ktt_tt`,
  `ktt_tt_aot`) captures one trace per scope.

## Attention kernels

```python
with kohakutt.sdpa_kernel("flash"):  # or several: sdpa_kernel("flash", "math")
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

`kohakutt.sdpa_kernel(*kernels)` is a numerics scope (`attention_kernels`) that runs the
attention calls queued in it on the kernels it names, as `torch.nn.attention.sdpa_kernel`
does on CUDA. It nests, follows a call to the stream worker, into `kohakutt.capture` and
into each `torch.compile` scope like the other settings.

| Kernel | What | Takes |
|---|---|---|
| `flash` | Kohaku-TT's flash kernel ([ktt_flash](../reference/kernels/flash.md)), forward and training backward; `sdpa_varlen` | bf16, head dims whose buffers fit L1 (training backward: multiples of 32 up to 256) |
| `own` | Kohaku-TT's earlier attention program | bf16 without mask; inference only |
| `ttnn` | TT-NN's SDPA; tt-train's training kernels | bf16; training: S a multiple of 32 |
| `math` | PyTorch's math composite on tt ops | every call |

- In a scope, a call runs on the first allowed kernel in the table's order that takes
  it; `own` runs a non-causal call with equal heads only when `ttnn` is not allowed (by
  default TT-NN takes those). Outside a scope every kernel is allowed and the routing is
  the default ([limitations](../reference/limitations.md)).
- A call none of the allowed kernels takes raises a `RuntimeError` that names the call
  (shapes, dtypes, mask kind, causal) and each allowed kernel's refusal (for example
  `flash: its circular buffers do not fit L1 (head dims 1536 / 1536, ...)`). The device kernels' L1 checks
  run once per shapes and scope on the stream (one sync).
- Training: an unmasked call with the default scale and Sq == Sk trains on `flash` (where
  its programs fit) or `ttnn` (S a multiple of 32), forward and backward. Other calls
  with gradients run their forward on a device kernel and their backward on `math`, so
  the scope must allow `math` too.
- The backward runs on its forward's kernel family whatever scope is open at
  `backward()`: the forward's scope decided it, and checked that it was allowed.
- `kohakutt.counters()` counts the calls per kernel: `attention_flash`, `attention_own`,
  `attention_ttnn`, `attention_math` (forward) and `attention_backward_flash`,
  `attention_backward_ttnn`, `attention_backward_math` (the recomputed backward of a
  device forward; a call that runs on `math` from the start differentiates through its
  ops and is not counted again).

## Requirements

A kernel bounds the scope: a fidelity floor (it needs the precision, or the precision
costs it nothing), a fidelity ceiling (more passes buy nothing), `exact` (no
approximations). The effective setting is the scope's, clamped to them, unless the scope
is strict.

| Kernel | Requirement | Reason |
|---|---|---|
| reductions (sum, max), softmax | floor HiFi4 | memory-bound; fp32 inputs lose their low mantissa bits below HiFi4 |
| fused RMSNorm (bf16) | floor HiFi4 | memory-bound |
| interpolation, adaptive pooling, `index_copy` (matmuls with selection / averaging matrices) | floor HiFi4 | they copy or average values: a bf16 value's 8 mantissa bits need the operand's second slice |
| FFT (DFT matrices), triangular solve (series inverse) | floor HiFi4 | fp32 operands; errors compound |
| own unary (`ktt_unary`) | follows `approx` | SFPU only; fidelity unused |
| own broadcast binary (`ktt_binary`) | exact | SFPU division's reciprocal |
| matmul, linear, attention | none | the scope's values |

User kernels state theirs: `kk.Kernel(min_fidelity=, max_fidelity=, exact=)` in Python,
`register_program(name, build, Requirements{...})` in C++; the builder receives the
effective values.

## Checking

Numerics are reported as statistics, never as a pass / fail threshold: the max, median
and mean of the absolute and relative error and the cosine similarity of a result against
a CPU float64 ruler (`kohakutt.diff_stats`), beside the same statistics of a CPU
simulation of the same precision (fp32 math on the same low-precision inputs, rounded as
the device rounds). Equal statistics mean the kernel is as exact as its precision allows;
`KernelAudit` reports the same statistics per aten call, per
module and per model.
