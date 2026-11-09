---
title: Kernel DSL (design note)
summary: "Status: proposal; nothing here is implemented yet."
tags:
  - concepts
  - dsl
---

# Kernel DSL (design note)

Status: proposal; nothing here is implemented yet.

Kohaku-TT runs most ops through TT-NN, and custom Metalium kernels where TT-NN has no
op or a slow one. This note scopes a kernel language for the second case and for
fusion inside `torch.compile`.

## What the kernels look like

Custom kernels written so far (the SwiGLU example, the resident MLP, flash attention and
ResNet-block kernels of [tt-x280-dispatch](https://github.com/KohakuBlueleaf/tt-x280-dispatch))
share one structure:

| Part | Content | Where the effort goes |
|---|---|---|
| reader | tile ranges per core from runtime args; DRAM -> L1 circular buffers; multicast or forwarding between cores | data-movement quality (K/V forwarding, halo reads) |
| compute | `*_init` + per-tile LLK calls on circular buffers; fidelity and fp32 accumulation as compile config | numerics choices |
| writer | circular buffer -> DRAM by tile range | little |
| host | program descriptor: circular buffers, kernel sources, compile / runtime args, core split | boilerplate, error-prone |

The compute bodies are short; the cost is data movement and host-side plumbing.

## Relation to tt-lang

tt-lang (`ttl.operation(grid=...)` with `@ttl.compute()` / `@ttl.datamovement()`
functions over dataflow buffers, `ttl.copy` transfers and block arithmetic) matches the
reader / compute / writer structure one to one and removes the descriptor boilerplate.
It brings its own compiler and runtime, which cannot share a process with TT-NN's, so it
is a reference for the language level, not a dependency.

## Proposal

1. **tt-lang's abstraction level**: tile blocks, dataflow buffers, separate compute and
   data-movement functions, a core grid with node coordinates.
2. **Lowered in process** to TT-NN `generic_op` program descriptors (Metalium kernel
   source, circular buffers, runtime args), cached by structure and run through the same
   bridge as every other op, so DSL kernels work in eager mode and inside traces.
3. **Numerics are arguments**: fidelity, fp32 accumulation and approximation modes are
   explicit, never hidden defaults.
4. **First target: fusion in `torch.compile`**. Chains of elementwise ops on equal
   shapes become one generated compute body with one reader and one writer, then
   matmul epilogues (bias, activation) and normalization; intermediates stay in L1
   instead of round-tripping through DRAM.
5. **View-aware readers**: data-movement functions take strided views directly (the NoC
   reads contiguous runs anywhere), so a consumer kernel can absorb a transpose, slice or
   permute instead of materializing it.
6. **Hand-written Metalium stays** for kernels whose value is in their data movement;
   the DSL can generate their host side.

## Open questions

- A user-facing language, or only generated fusion inside `torch.compile` at first?
- tt-lang syntax for portability of user kernels, or only its concepts?
