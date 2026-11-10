---
title: Concepts
summary: Mental models for Kohaku-TT - the hardware, how kernels are designed and written, tensor layout, numerics, traces, the mesh and performance.
tags:
  - concepts
  - overview
---

# Concepts

Concept pages explain why Kohaku-TT works the way it does. They are not a reference:
op tables, kernel pages and issue lists live in [reference/](../reference/README.md).

## Reading paths

### Using Kohaku-TT

You run PyTorch models on the card and want to know what to expect.

1. [Numerics](numerics.md): what precision tt ops run at and how to change it.
2. [Trace contract](trace.md): capture, replay and `torch.compile`.
3. [Mesh](mesh.md): two cards as one device.
4. [Performance](performance.md): where time goes and how it is measured.
5. [Block floats](block-float.md): smaller weights.

### Writing aten kernels

You add or fix an aten op.

1. [Kernel contract](kernel-contract.md)
2. [Tensor layout](layout.md)
3. [Numerics](numerics.md#requirements)

### Writing Metalium kernels

You write one of Kohaku-TT's own device programs.

1. [Hardware](hardware.md): the grid, the NoCs and the Tensix tile.
2. [Kernel design](kernel-design.md): topology first, the roofline target, the design
   sheet and the gate.
3. The [kernel reference pages](../reference/kernels/README.md) as worked examples.
4. [Kernel DSL](dsl.md): where kernel authoring may go next.

## Pages

| Page | What it explains |
|---|---|
| [hardware.md](hardware.md) | the Blackhole p150a as a kernel author sees it |
| [kernel-design.md](kernel-design.md) | how own kernels are planned, the design-sheet sections, program rules, the gate |
| [kernel-contract.md](kernel-contract.md) | what an aten kernel receives and returns: views, dtypes, outputs |
| [layout.md](layout.md) | TILE vs ROW_MAJOR storage, conversion costs, views |
| [numerics.md](numerics.md) | fidelity and approximation settings, scopes, kernel requirements, checking |
| [block-float.md](block-float.md) | BFLOAT8_B / BFLOAT4_B storage, `F.scaled_mm`, numerics |
| [trace.md](trace.md) | the trace contract and the compile backends |
| [mesh.md](mesh.md) | placements, op rules, collectives, marks, traces, measured speed |
| [performance.md](performance.md) | principles, MOSS-Image measurements, first-call costs, own kernels against TT-NN |
| [dsl.md](dsl.md) | kernel programs written in Python (`kohakutt.dsl`): binding times, what never recompiles, the verifier |
