---
title: ktt_adam
summary: One Adam / AdamW step of one parameter as one program (torch.optim with fused=True) - what it computes, routing, coverage, the program, measured step times and limits.
tags:
  - reference
  - kernels
  - training
---

# ktt_adam

One optimizer step of one parameter as one program: from the parameter `p`, its gradient
`g` and the two moments `m`, `v` to the new `p`, `m`, `v`.

```
g' = (maximize ? -g : g) + l2 * p
m' = beta1 * m + (1 - beta1) * g'
v' = beta2 * v + (1 - beta2) * g'^2
p' = decay * p - step * m' / (sqrt(v') * scale + eps)
```

with `step = lr / (1 - beta1^t)` and `scale = 1 / sqrt(1 - beta2^t)`. AdamW:
`decay = 1 - lr * weight_decay`, `l2 = 0`. Adam: `decay = 1`, `l2 = weight_decay`.

Source: `src/kohakutt_native/csrc/aten/optim/` (`aten_optim.cpp` entry,
`device_optim.cpp`, `ktt_adam/`).

## Routing

`torch.optim.Adam(..., fused=True)` and `torch.optim.AdamW(..., fused=True)` call
`_fused_adam_` / `_fused_adamw_`, which run one `ktt_adam` program per parameter. Without
`fused=True` PyTorch runs its per-tensor composite (about ten device ops per parameter).

| Call | Route |
|---|---|
| bf16 / fp32 parameter, gradient and moments of one shape and dtype | `ktt_adam` |
| `amsgrad=True`, or a gradient of another dtype | PyTorch's single-tensor step on tt ops |
| `grad_scale` (a GradScaler) | the gradient is divided first, then as above |
| `found_inf` | not supported |
| `KTT_CUSTOM_KERNELS=0` | PyTorch's single-tensor step on tt ops |

## Coverage

- Any shape; Flat-stored weights step as stored (the moments are converted to the
  parameter's format once).
- On a mesh, operands of one placement without partial axes step per chip on their shards.
- The step count is a 0-d device tensor that PyTorch increments before the call. Its value
  is tracked on the host through that increment, so no step reads it back from the device
  after the first.

## The program

- Reader: `block` (4) tiles of each of p, g, m, v per read barrier into CB 0 - 3.
- Compute: one tile per DST acquire, the four tiles in fp32 DST, standard SFPU tile ops
  (scalar multiply / add, square, square root, divide); p', m', v' packed to CB 16 - 18.
- Writer: 4 tiles of each output per write barrier.
- The coefficients are common runtime arguments, outside the program key: every step of a
  parameter reuses one program and one cached descriptor.
- bf16 operands are computed in fp32 and rounded once when packed.

## Performance

One P150A, bf16, eager, each step synchronized, 2026-11-10. "Update" is `optimizer.step()`
alone; "step" is forward, backward and update.

| Model | parameters | update, composite | update, fused | step, composite | step, fused |
|---|---|---|---|---|---|
| DINOv3 ViT-B/16, 224 px | 210 | 66 ms | 20 ms | 169 - 177 ms | 123 - 132 ms |
| DINOv3 ConvNeXt-tiny, 224 px | 180 | 54 ms | 17 ms | 125 - 130 ms | 89 - 90 ms |

## Limits

- One program per parameter: the update costs about 0.09 ms per parameter, most of it the
  launch. Parameters are not batched into one program.
- Each step counter is still incremented by its own small device op.
- bf16 moments keep 8 bits: with `beta2 = 0.999` the second moment's per-step change is
  below one bf16 step, as in PyTorch's own bf16 arithmetic. Keep optimizer state in fp32
  when that matters.
