---
title: Block-float weights and scaled matmul
summary: "Tenstorrent cores natively read TT block floats: BFLOAT8_B and BFLOAT4_B tiles, where each group of 16 values shares one 8-bit exponent and keeps a sign + 7-..."
tags:
  - concepts
  - block-float
---

# Block-float weights and scaled matmul

Tenstorrent cores natively read TT block floats: BFLOAT8_B and BFLOAT4_B tiles, where
each group of 16 values shares one 8-bit exponent and keeps a sign + 7-bit (bfp8) or
3-bit (bfp4) mantissa. A bfp8 tile is 1088 bytes (vs 2048 for bf16), a bfp4 tile 576.
The matmul engine takes them directly, and mixes them with bf16 / fp32 operands.

## Storage format, not a dtype

PyTorch has no dtype for these formats, so Kohaku-TT treats them as a storage format of
a bf16 tensor:

```python
w = linear.weight.data  # bf16 tt tensor
kohakutt.block_float_(w, "bfp8")  # stored as BFLOAT8_B, in place
kohakutt.storage_format(w)  # "bfp8"
w.dtype  # torch.bfloat16
w.cpu()  # the block-rounded values, as bf16
kohakutt.block_float_linears(model, "bfp8")  # every nn.Linear weight of a model
```

- **Matmul operands are used as stored.** `F.linear`, `torch.mm` and `F.scaled_mm`
  pass a block-float weight (or its transpose) straight to the TT matmul,
  next to a bf16 / fp32 activation. No activation conversion and no dequantized weight
  copy happen.
- **Everything else reads the bf16 value.** Any other op (or a view that is not the
  whole weight) reads the tensor through an on-device conversion to bf16.
- **A write ends the block format.** An in-place write into the tensor stores bf16 from
  then on.

## `F.scaled_mm`

`torch.nn.functional.scaled_mm` (which dispatches to `aten::_scaled_mm_v2`) runs on tt:

```python
one = torch.ones(())
y = F.scaled_mm(x, w.t(), one, F.ScalingType.TensorWise, one, F.ScalingType.TensorWise)
```

- Operands are bf16 / fp32 tensors, either of which may be block-float storage. The
  per-16 block scales live inside that storage, so they are not passed separately.
- `scale_a` / `scale_b` are extra decode scales: TensorWise (one value) or RowWise (M
  values for `mat_a`, N for `mat_b`), applied to the fp32 product. A host scale of 1 is
  skipped. Other recipes and scale swizzles raise `KohakuTTUnsupportedOperator`.
- `bias` is added in fp32; the result is cast to `output_dtype` (bf16 by default).
- `use_fast_accum=True` runs the matmul at LoFi fidelity; otherwise the
  `KTT_MATMUL_FIDELITY` setting applies (HiFi4 by default). Accumulation is fp32.

## Numerics

Against a CPU fp64 ruler on the same rounded inputs (bf16 activation, block-rounded
weight), tt's bf16-output errors match a CPU bf16 matmul's (median and mean abs error
within ~5%, 32x64x32 to 4096x8192x3072, bfp8 and bfp4). The rounding of the weight itself
is the larger effect: bfp4 keeps 3 mantissa bits per value.

fp32-output matmuls on tt are accurate to ~3e-4 relative (bf16 operands) and ~1e-3 (fp32
operands), not to CPU fp32 accuracy; work that needs fp32 accuracy uses fp32 vector ops.
