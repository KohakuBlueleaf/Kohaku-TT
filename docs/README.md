---
title: Kohaku-TT Documentation
summary: Home for the concept model, the reference tables and kernel pages, and the map of the codebase.
tags:
  - overview
  - docs
---

# Kohaku-TT Documentation

Kohaku-TT makes a Tenstorrent Blackhole card a native PyTorch device: `tt` is a
`torch.device` like `cuda`. Every aten op a model calls runs as a TT-NN op, a composite
of TT-NN ops, or one of Kohaku-TT's own Metalium programs; eager ops queue on an
asynchronous stream, autograd trains on the device, `torch.compile` replays captured
device traces, and several cards act as one device through the mesh.

These docs are split by what you want from them: concepts (why things are the way they
are) and reference (everything that exists). Task-oriented walkthroughs live in the
[examples](../examples/README.md), the [quick start](quick-start.md) and the top-level
[README](../README.md).

## Pick a path

| You are... | Start here |
|---|---|
| **Running a model on the card** | [install](install.md) · [quick start](quick-start.md) · [examples](../examples/README.md) · [numerics](concepts/numerics.md) |
| **Using several cards** | [mesh](concepts/mesh.md) · [examples/05_mesh.py](../examples/05_mesh.py) |
| **Capturing traces or compiling** | [trace contract](concepts/trace.md) · [performance](concepts/performance.md) |
| **Tuning or diagnosing a run** | [knobs, commands and tools](reference/knobs.md) · [measured](reference/measured.md) · [limitations](reference/limitations.md) |
| **Writing your own kernel as a torch op** | [custom kernels](reference/custom-kernels.md) · [kernel DSL](concepts/dsl.md) · [examples/kernels](../examples/kernels/README.md) |
| **Checking whether an op is supported** | [aten support](reference/aten_support.md) · [census](reference/census.md) · [TT-NN known issues](reference/ttnn-known-issues.md) |
| **Writing an aten kernel** | [kernel contract](concepts/kernel-contract.md) · [tensor layout](concepts/layout.md) · [csrc/aten](../src/kohakutt_native/csrc/aten/README.md) |
| **Writing a Metalium kernel for Kohaku-TT** | [hardware](concepts/hardware.md) · [kernel design](concepts/kernel-design.md) · [kernel reference](reference/kernels/README.md) |

## Documentation structure

### Start

- [Install](install.md): from PyPI, or built from a clone against your own tt-metal build.
- [Quick start](quick-start.md): one chip, two chips, compile and capture, block floats, a whole model.
- [Features](features.md): what Kohaku-TT provides, with the results behind each claim.
- [tinygrad add-on](tinygrad.md): the card as the tinygrad device `"KTT"`, its numerics and limits.
- [Development](development.md): repository layout, checks, conventions.

### Concepts

Mental models: why things are the way they are. See the [concepts index](concepts/README.md).

- [How it works](concepts/architecture.md): the layers from the dispatcher to the card, tensors, the default stream, op routing, precision, trace, the mesh.
- [Hardware](concepts/hardware.md): the Blackhole p150a as a kernel author sees it: grid, DRAM, NoCs, multicast, the Tensix tile.
- [Kernel design](concepts/kernel-design.md): how Kohaku-TT's own kernels are planned, and the gate before one replaces a TT-NN op.
- [Kernel contract](concepts/kernel-contract.md): what an aten kernel receives and returns.
- [Tensor layout](concepts/layout.md): TILE and ROW_MAJOR storage, conversions and views on the device.
- [Numerics](concepts/numerics.md): fidelity, approximations, scopes, and how precision is checked.
- [Block floats](concepts/block-float.md): BFLOAT8_B / BFLOAT4_B storage and `F.scaled_mm`.
- [Trace contract](concepts/trace.md): what a captured step may do; the compile backends.
- [Mesh](concepts/mesh.md): several chips as one device.
- [Performance](concepts/performance.md): how performance problems are found and fixed, with the MOSS-Image measurements.
- [Kernel DSL](concepts/dsl.md): kernel programs written in Python (`kohakutt.dsl`): binding times, what never recompiles, the verifier.

### Reference

Exhaustive lookup. See the [reference index](reference/README.md).

- [Knobs, commands and tools](reference/knobs.md): every environment variable, the `ktt` commands, the audit and profiling tools.
- [Measured results](reference/measured.md): against tt-xla and TT-NN, with conditions.
- [Limitations](reference/limitations.md): what does not work, or works slowly, today.
- [Custom kernels](reference/custom-kernels.md): your own Metalium programs as torch ops, from Python or C++.
- [aten support](reference/aten_support.md): every aten overload and its status (generated).
- [Census](reference/census.md): the ops real models call and their status on tt (generated).
- [TT-NN known issues](reference/ttnn-known-issues.md): TT-NN defects and how Kohaku-TT avoids them.
- [Kernels](reference/kernels/README.md): Kohaku-TT's own Metalium kernels, one page each.

## Codebase map

```
src/kohakutt/                  the package you import
  device/                      device and mesh API
  torch_backend/               registration, torch.tt, fallback, trace capture, compile
  kernels.py, extension.py     user Metalium programs from Python; C++ extension libraries
  block_float.py               block-float weights
  core/  tools/  bench/        numerics and stats; support, coverage and sweep tools; benchmarks
src/kohakutt_native/           the native layer (README inside)
  csrc/aten/<family>/          every aten op: aten_*.cpp (torch side), device_*.cpp (device side),
                               ktt_<name>/ (Kohaku-TT's own Metalium programs)
  csrc/core/torch/, device/    the tensor model; the device, stream, handles, program budget
  csrc/runtime/                device guard, hooks, capture, ktt:: ops, user programs
  include/ktt/extension/       the C++ extension API
examples/                      plain-PyTorch examples, MOSS-Image, custom kernels
experiments/                   standalone proofs of concept (README inside)
tests/                         core/ (CPU) and hardware/ (needs a card)
docs/                          this tree
```

The full layout and the checks: [development](development.md).

## What the docs promise

- **Concepts** tell you why X works that way.
- **Reference** tells you every X that exists, with its current numbers and conditions.
- Measurements state their conditions (card, dtype, fidelity, shape). A number without
  them is a bug in the docs.
