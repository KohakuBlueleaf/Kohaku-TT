# Kohaku-TT

A native PyTorch device for Tenstorrent Blackhole cards (developed on P150A). `tt` is a
`torch.device` like `cuda`: every aten op a model calls runs on the card as a TT-NN op,
a composite of TT-NN ops or one of Kohaku-TT's own Metalium programs. Models, training
loops and tools move to the card with `.to(dev)` and keep running op by op. It is for
people who want to run and train ordinary PyTorch code on a Blackhole card, and for
kernel authors who want their own Metalium programs as torch ops.

**Status: early development.** Eager execution, training and `torch.compile` run end to
end on real models (ViT, Qwen2.5 training, the MOSS-Image text-to-image pipeline).

```python
import torch
import kohakutt

dev = kohakutt.device(0)  # chip 0 as a torch.device ("tt:0")
x = torch.randn(4, 8, device=dev)
w = torch.randn(8, 3).to(dev)
y = torch.relu(x @ w + 1.0)  # queued on the card; returns at once
print(y.sum(dim=1).cpu())  # a host read waits for the result
```

## Highlights

- **Every Core ATen op runs on tt**: 193 of 193 Core ATen overloads have a tt kernel or
  a composite of tt kernels ([support table](docs/reference/aten_support.md)). An op
  without one raises `KohakuTTUnsupportedOperator`; the CPU runs it only when asked.
- **A real device**: storage-backed tensors with views and aliasing, an asynchronous
  default stream, autograd and optimizers on the card, `torch.tt`, `torch.save` / `load`.
- **Trace capture and `torch.compile`**: `kohakutt.capture` and the `ktt_tt` /
  `ktt_tt_aot` backends replay a step as one device trace.
- **Own kernels where TT-NN falls short**: flash attention (MHA / GQA / MQA, masks,
  causal, packed variable-length batches), layer / RMS norm with fp32 statistics,
  grouped and channels-last convolutions, fused transformer ops.
- **Two cards as one device** (experimental): `kohakutt.mesh([0, 1])` shards tensors
  over a 1 x 2 mesh.
- **Your kernels as torch ops**, from Python, the Python kernel DSL or C++.
- **Block-float weights** (BFLOAT8_B / BFLOAT4_B), **precision as a scope**
  (`kohakutt.numerics`) and tools that report error against a float64 ruler.
<!-- tinygrad:start -->
- **tinygrad**: an early add-on registers the card as the tinygrad device `"KTT"`.
<!-- tinygrad:end -->

One Blackhole P150A, bf16, eager, warm. The TT-NN column is the same model on TT-NN's
kernels (`KTT_CUSTOM_KERNELS=0`); error is the median relative error against CPU float64
(CPU bf16 in brackets). MOSS-Image: bfp8 DiT weights, HiFi2, one chip.

| Model | Kohaku-TT | TT-NN kernels | Error, Kohaku-TT / TT-NN |
|---|---|---|---|
| DINOv3 ConvNeXt-tiny, forward 224 px | 14.1 ms | 365.8 ms | 4.1e-2 / 8.3e-2 (3.4e-2) |
| DINOv3 ConvNeXt-tiny, training step | 199.8 ms | 1427.9 ms | gradients 0.067 / 0.127 (0.061) |
| DINOv3 ViT-B/16, forward 224 px | 35.6 ms | 45.3 ms | 2.7e-2 / 3.5e-2 (2.0e-2) |
| DINOv3 ViT-B/16, training step | 103.2 ms | 112.2 ms | loss 0.007 / 0.019 (0.011) |
| MOSS-Image CFG step, 1024 x 1024, 300 text tokens | 0.935 s | | |

More results and their conditions, including tt-xla: [measured](docs/reference/measured.md).
The full feature list: [features](docs/features.md).

## Install

A Blackhole card with its kernel driver and firmware. From PyPI (Linux x86-64, glibc
2.39 or newer, Python 3.13 or 3.14):

```
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0+cpu
pip install kohakutt          # --pre for the nightly builds
ktt sfpi                      # once: fetches the RISC-V kernel compiler into the cache
ktt doctor                    # read-only report of the host and every card
```

No tt-metal checkout is needed for the wheel. Building from a clone: [install](docs/install.md).

## Usage

### PyTorch: tensors, training, capture and compile

Model, loss and optimizer code is unchanged PyTorch; a repeated step can be captured
once and replayed as one device trace ([trace contract](docs/concepts/trace.md);
[examples/02_training.py](examples/02_training.py), [03_trace.py](examples/03_trace.py),
[04_compile.py](examples/04_compile.py)):

```python
import torch
import torch.nn.functional as F
from torch import nn

import kohakutt

dev = kohakutt.device(0)
model = nn.Sequential(nn.Linear(256, 512), nn.SiLU(), nn.Linear(512, 10)).to(dev)
opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)


def step(x, y):
    opt.zero_grad(set_to_none=False)  # a trace needs fixed grad buffers
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    return loss.detach()


x, y = torch.randn(128, 256).to(dev), torch.randint(0, 10, (128,)).to(dev)
print(step(x, y).item())  # eager: each op queued on the card
replay = kohakutt.capture(step, x, y, warmup=3)  # capture once, replay as one trace
print(replay(x, y).item())
compiled = torch.compile(model, backend="ktt_tt_aot")  # "ktt_tt" for inference
```

<!-- tinygrad:start -->
### tinygrad (early add-on)

`import kohakutt.tinygrad` registers the card (chip 0) as the tinygrad device `"KTT"`;
the code is plain tinygrad v0.14.0
([examples/tinygrad/01_tensors.py](examples/tinygrad/01_tensors.py),
[02_training.py](examples/tinygrad/02_training.py)):

```python
from tinygrad import Context, Tensor, nn

import kohakutt.tinygrad  # noqa: F401  registers the device "KTT" (chip 0)

model = nn.Linear(256, 4)
for p in nn.state.get_parameters(model):
    p.to_("KTT")
opt = nn.optim.AdamW(nn.state.get_parameters(model), lr=1e-3)
images = Tensor.randn(64, 256).to("KTT")  # random numbers: drawn on the CPU, moved
labels = Tensor([0, 1, 2, 3] * 16, device="KTT")
with Context(TRAINING=1):
    loss = model(images).sparse_categorical_crossentropy(labels)
    opt.zero_grad()
    loss.backward()
    opt.step()
```

It is an early add-on and runs on one card. It shares the native bridge with the torch
backend: both frontends run in one process on one open device (`kohakutt.open`).
Results match tinygrad's CPU backend up to the card's default matmul precision (`with kohakutt.tinygrad.numerics(fidelity=3):`
narrows it; fused attention and the conv backward stay bf16). To check a training run
against the CPU backend, compare losses or gradients: parameters after an Adam step
differ by about `lr` wherever a gradient is near zero, on any backend. Not supported
yet: random numbers drawn on the device; float16 / int64 kernels (they run on the
host); tinygrad's memory planner (off); `import kohakutt.tinygrad` needs torch
installed. See [docs/tinygrad.md](docs/tinygrad.md) and
[examples/tinygrad](examples/tinygrad/README.md).
<!-- tinygrad:end -->

### Two cards as one device (experimental)

`kohakutt.mesh([0, 1])` opens two chips as one torch device. The mesh is experimental:
it is built and tested as 1 x 2, and its API and defaults may change. Tensors of 1 MiB or more are sharded when created, smaller ones replicated,
and the TT-NN collective an op needs is inserted for it. Marks set a module's own
placement and return the same module ([examples/05_mesh.py](examples/05_mesh.py)):

```python
class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = kohakutt.replicate(nn.LayerNorm(d))  # a copy on each chip
        self.up = kohakutt.colwise(nn.Linear(d, 4 * d))  # output features split
        self.down = kohakutt.rowwise(nn.Linear(4 * d, d))  # input features split

    def forward(self, x):
        return x + self.down(nn.functional.gelu(self.up(self.norm(x))))


mesh = kohakutt.mesh([0, 1])
model = nn.Sequential(Block(1024), Block(1024)).bfloat16().to(mesh)
with kohakutt.record_placements():
    out = model(torch.randn(512, 1024, dtype=torch.bfloat16).to(mesh))
print(kohakutt.placement(model[0].down.weight))  # "Sharded(dim=1)"
print(kohakutt.placement_log()[-1])  # op, inputs, output, conversions, reason
print(torch.tt.mesh_stats())  # collectives so far
```

`kohakutt.shard(module, dim)` marks a module's tensors sharded on `dim`, and
`kohakutt.place(module, weight="S(1)", bias="R")` marks them one by one. Design and
measurements: [mesh](docs/concepts/mesh.md).

### Custom kernels

A Metalium program (reader, compute and writer kernels over circular buffers) becomes a
torch op on tt tensors, run in order with every other op. In C++, the program builder
and the op are built into an extension library
([examples/kernels/cpp_extension](examples/kernels/cpp_extension/)):

```cpp
// device side: swiglu_program.cpp builds the ProgramDescriptor for (a, b, out)
KTT_REGISTER_PROGRAM("kohakutt_example_cpp::swiglu", swiglu);

// torch side: swiglu_op.cpp
at::Tensor swiglu(const at::Tensor& a, const at::Tensor& b) {
    return ktt::extension::run(
        "kohakutt_example_cpp::swiglu", {a, b}, {{a.sizes().vec(), a.scalar_type()}}
    )[0];
}
```

`kohakutt.extension.load("build/swiglu_ext/swiglu_ext.so")` loads the library; the op
is then `torch.ops.kohakutt_example_cpp.swiglu(a_tt, b_tt)`.

With the kernel DSL (`kohakutt.dsl.metal`) the whole program is Python; the kernel
sources are generated and checked against the hardware's rules, and sizes are runtime
arguments, so a new shape or scale compiles nothing. The whole program and op for
`a * s + b`, one tile per step (`km` is `kohakutt.dsl.metal`, `kk` is `kohakutt.kernels`,
`bits` gives a float's fp32 bits;
[examples/kernels/dsl_metal/scaled_add_km.py](examples/kernels/dsl_metal/scaled_add_km.py)):

```python
@km.program("scaled_add")
def scaled_add_program(p, a: km.TensorRef, b: km.TensorRef, out: km.Out):
    start, count = p.split(p.runtime("tiles"))  # each core's range of tiles
    s = p.runtime("s")  # fp32 bits of the scale
    cb_a, cb_b, cb_out = p.cb(0, like=a), p.cb(1, like=b), p.cb(16, like=out)

    @p.dm0
    def reader(k):
        for i in k.range(count):
            a_slot, b_slot = cb_a.reserve(1), cb_b.reserve(1)
            k.read(a.page(start + i), a_slot)
            k.read(b.page(start + i), b_slot)
            k.reads_done()
            cb_a.push(1)
            cb_b.push(1)

    @p.compute(fp32_dest=True)
    def compute(c):
        c.startup(cb_a, cb_b, cb_out)
        for _ in c.range(count):
            cb_a.wait(1)
            cb_b.wait(1)
            with c.registers():
                c.init("copy", cb_a)
                c.copy(cb_a, 0, 0)  # a in register slot 0
                c.init("copy", cb_b)
                c.copy(cb_b, 0, 1)  # b in register slot 1
                c.init("mul_scalar")
                c.op("mul_scalar", 0, value=s)  # slot 0 = slot 0 * s
                c.init("add")
                c.op("add", 0, 1, 0)  # slot 0 = slot 0 + slot 1
                cb_out.reserve(1)
                c.pack(0, cb_out)
                cb_out.push(1)
            cb_a.pop(1)
            cb_b.pop(1)

    @p.dm1
    def writer(k):
        for i in k.range(count):
            cb_out.wait(1)
            k.write(cb_out, out.page(start + i))
            k.flushed()
            cb_out.pop(1)
        k.writes_done()


@km.op(
    "kohakutt_example::scaled_add_km(Tensor a, Tensor b, float s) -> Tensor",
    scaled_add_program,
)
def scaled_add(a, b, s):
    return km.Call(
        inputs={"a": a, "b": b},
        outputs={"out": (list(a.shape), a.dtype)},
        runtime={"tiles": kk.tile_count(a), "s": bits(s)},
    )
```

Hand-written Metalium kernels from Python (`kohakutt.kernels`) and the full API:
[custom kernels](docs/reference/custom-kernels.md), [kernel DSL](docs/concepts/dsl.md).

## Limitations

- One device or mesh per process; the mesh is built and tested as 1 x 2. An op without
  a placed form gathers its input to both chips.
- On the mesh, a bf16 partial sum is rounded on each chip before the chips' sums are
  added, so ops on it carry a larger rounding error than on one card
  ([mesh](docs/concepts/mesh.md)).
- A trace holds every intermediate of the captured graph; very large forwards do not fit
  as one trace.
- fp32 matmuls are accurate to about 1e-3 relative, not to CPU fp32.
- 1-D tensors are one padded tile row; bf16 NaN may come back as inf after device ops.
- Data movement (permutes, gathers of strided views) runs as TT-NN op chains, not near
  memory bandwidth; tensors whose last two dims are small pad to 32 x 32 tiles.
- Some attention calls (fp32, dropout) run PyTorch's math composite on tt ops.
- L1-resident activations are opt-in (`KTT_L1_BUDGET`); when they spill to DRAM, a view
  and its base that shared an L1 buffer no longer alias.
- The first call at a new shape of TT-NN's reshape, padded slice, permute and conv2d
  costs host setup (seconds at image sizes).

Details: [limitations](docs/reference/limitations.md).

## Documentation

| Page | Read it for |
|---|---|
| [docs index](docs/README.md) | every concept and reference page |
| [quick start](docs/quick-start.md) | one chip, two chips, compile, block floats, a whole model |
| [examples](examples/README.md) | tensors, training, trace, compile, the mesh, MOSS-Image, custom kernels |
| [how it works](docs/concepts/architecture.md) | the layers, tensors, the default stream, op routing, precision |
| [knobs, commands and tools](docs/reference/knobs.md) | every environment variable, `ktt` commands, audit and profiling tools |
| [measured](docs/reference/measured.md) | results against tt-xla and TT-NN, with conditions |
| [development](docs/development.md) | repository layout, checks, conventions |

## Contributing

Open an issue first for a feature. Code conventions and the checks to run before a pull
request are in [development](docs/development.md).

## License

Apache-2.0.
