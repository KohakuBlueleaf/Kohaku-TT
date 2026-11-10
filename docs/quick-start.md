---
title: Quick start
summary: One chip, two chips, compile and capture, block-float weights and a whole model, each in a few lines.
tags:
  - overview
  - usage
---

# Quick start

**One chip.** `kohakutt.device(i)` opens chip *i* on first use and returns its
`torch.device`; everything after `.to(dev)` is ordinary PyTorch:

```python
dev = kohakutt.device(0)
x = torch.randn(4096, 4096, dtype=torch.bfloat16).to(dev)
y = (x @ x.T).softmax(-1)  # queued; returns at once
kohakutt.synchronize(dev)  # waits for the queued work
print(y[0, :4])  # a host read waits too
```

**Training.** Autograd and optimizers run on the device; a fused AdamW runs one program
per parameter:

```python
model = MyModel().bfloat16().to(dev)
opt = torch.optim.AdamW(model.parameters(), fused=True)  # one program per parameter

loss = loss_fn(model(x.to(dev)), y.to(dev))  # plain PyTorch from here on
loss.backward()
opt.step()
```

**Two chips as one device.** `kohakutt.mesh([0, 1])` is a 1 x 2 mesh: large tensors are
sharded over the chips, small ones replicated, and each op runs on the shards
([how it works](concepts/architecture.md#multiple-cards), [mesh](concepts/mesh.md)). To
use two chips for two jobs instead, run one process per chip with `TT_VISIBLE_DEVICES`:

```
TT_VISIBLE_DEVICES=0 python generate.py --shard 0/2 &
TT_VISIBLE_DEVICES=1 python generate.py --shard 1/2 &
```

**Compile or capture a repeated step.** Both replay the step as one device trace:

```python
compiled = torch.compile(model, backend="ktt_tt")  # inference; "ktt_tt_aot" trains
step = kohakutt.capture(train_step, x, y)  # capture once, replay on every call
```

**Block-float weights.**

```python
kohakutt.block_float_linears(model, "bfp8")  # every nn.Linear weight, in place
y = model(x)  # bf16 activations x bfp8 weights
```

**A whole model.** [examples/06_moss_image](../examples/06_moss_image/) runs MOSS-Image (a
Qwen3.5 text encoder, a DiT flow sampler with CFG, the FLUX.2 VAE) as plain PyTorch
modules on tt, at the size each prompt asks for:

```
python examples/06_moss_image/generate.py --model MOSS-Image-Base \
    --width 1344 --height 768
python examples/06_moss_image/generate.py --model MOSS-Image-Base --prompts prompts/ \
    --cfgs 2 3 4.5 --seeds 0 1 --out-dir images/
```

Command-line tools: [knobs, commands and tools](reference/knobs.md#commands).
