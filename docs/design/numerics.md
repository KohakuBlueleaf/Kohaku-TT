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
- An op reads the scope on the calling thread when it is queued and carries the value to
  the stream worker, so work queued in a scope runs with it after the scope closed.
- The process default is HiFi2, exact; `KTT_FIDELITY` (alias `KTT_MATMUL_FIDELITY`) and
  `KTT_APPROX=1` change it.
- `kohakutt.capture` keeps the scope it captured under; `torch.compile` (`ktt_tt`,
  `ktt_tt_aot`) captures one trace per scope.

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
