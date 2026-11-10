---
title: Kernel design
summary: How Kohaku-TT's own Metalium kernels are designed - topology first, a roofline target written before code, TT-NN as a baseline only, the design-sheet sections every kernel page covers, the program rules and the gate a kernel passes before it replaces a TT-NN op.
tags:
  - concepts
  - kernels
  - performance
---

# Kernel design

Kohaku-TT ships its own Metalium programs under `csrc/aten/<family>/ktt_<name>/` where
TT-NN is wrong, slow, or missing an op. The [kernel contract](kernel-contract.md) covers
the ATen side (what a kernel receives and returns); this page covers the device side:
how work, data movement, L1 and compute are planned, and what a kernel must show before
it replaces a TT-NN path. It assumes the machine model in [hardware](hardware.md).

## Topology first

A Blackhole chip is not a uniform grid. Its Tensix cores sit in two physical panels,
DRAM sits in two columns (one at the left edge, one between the panels), and each NoC
routes in one fixed direction. Every kernel design starts there:

- **Each core's role comes from its physical position.** Which cores read DRAM, which
  relay or multicast, and which only compute is decided by where they sit relative to
  the DRAM columns and to the panel boundary, using the SoC descriptor's coordinate
  translation, not logical indices.
- **Each stream's NoC comes from its path.** For every stream the design names its
  source and destination tiles and picks the NoC whose route between them is short and
  does not share links with another heavy stream. A DRAM read's NoC depends on the
  reader's panel and the page's DRAM column (the table in
  [hardware](hardware.md#which-noc-a-dram-transfer-should-use)); a relay between
  neighbours travels in its NoC's own direction (NoC0: +x / +y, NoC1: -y / -x). There is
  no fixed "reads on NoC0, writes on NoC1" assignment.
- **Bytes are counted per link.** Congestion happens only on shared links, so the plan
  states how many bytes cross the busiest link of each NoC, not just per core.

## The roofline is the target

Before code, the design sheet writes down the expected cycles from the hardware
documents: FPU tile products at the chosen fidelity (16 `MVMUL` per 32^3 tile product
per phase), DRAM bytes at 512 GB/s, NoC bytes on the busiest link at 64 B per cycle,
and per-reader request limits. The largest term is the bound; the sheet names which
one binds and what fraction of it the kernel must reach. Measurements are then read
against that bound, not against another implementation.

TT-NN's time for the same op and shape is a baseline only: a kernel must not be slower
than TT-NN where it replaces it (see the gate), but matching TT-NN is not the goal, and
TT-NN's design choices are not a reference for the plan.

## The design sheet

Every kernel has a reference page under
[reference/kernels/](../reference/kernels/README.md) that covers these sections. A
kernel without one does not replace a TT-NN op.

1. **Grid and roles.** What one unit of work is, how units map to cores, which cores
   read, relay, multicast or only compute and why (their physical position), how the
   last partial unit is handled (sizes are runtime arguments), and how load is balanced
   (triangular work such as causal attention is dealt zig-zag, pairing an early unit
   with a late one). Idle cores are listed with the reason.
2. **NoC plan per stream.** For every stream: source, destination, the RISC that moves
   it, the NoC and why (its path), unicast, relay or multicast.
   - Data that many cores need (an A block for a row of cores, a K / V chunk for every
     query head of a group) is read from DRAM once and relayed or multicast, not read
     once per core.
   - A multicast rectangle is sent from a corner so packets travel in the NoC's own
     direction (NoC0: the low-x / low-y corner; NoC1: the high corner), excludes the
     sender unless the loopback variant is used, and lies inside one physical panel: a
     logical row of cores is split by panel and each part is multicast separately.
   - Cores that read DRAM in lockstep start at different pages (the read order is
     rotated by the core's position), or they hit the same bank at the same step.
   - A kernel that issues on both NoCs from one RISC runs in
     `NOC_MODE::DM_DYNAMIC_NOC`.
   - No integer division or modulo in a data-movement inner loop (`div` / `rem` take
     6-33 cycles): indices advance and wrap by comparison.
3. **L1 table.** One row per circular buffer: contents, tile format, tiles per block,
   depth, bytes. Streamed operands are at least double-buffered so the reader runs one
   block ahead of compute; the resident operand gets what is left. The total is checked
   on the host before launch against `program::cb_budget()` (`core/device/program.hpp`):
   the free L1 span below the lowest live L1 buffer at the call, after the program's own
   outputs are allocated, less 32 KiB. Every plan field that depends on it is part of the
   program key, so a cached program always fits the L1 it runs in. A CB read by
   tile index holds each indexed set contiguously (the unpacker does not wrap at the
   end of the ring).
4. **Compute and DST plan.** Subblock shape (filling one DST half: 8 bf16 or 4 fp32
   tiles), accumulation across K blocks (packer L1 accumulation into an fp32 CB, not a
   reload through DST), math fidelity per input dtype (measured against float64, see
   [numerics](numerics.md)), SFPU work per tile and its approximation mode, and how
   unpack, math and pack stay balanced.
5. **Roofline and bound.** The expected cycles of each term (FPU, DRAM, NoC per link,
   reader issue rate), which binds, and the measured fraction next to it.
6. **Variant boundaries.** When the family's dispatcher picks this kernel over its
   siblings, decided from shapes and dtypes before launch, never by trying.

## Program rules

- **One program per variant and block configuration.** Every size (M, N, K, batch,
  sequence lengths, offsets) is a runtime argument; the program cache key holds only
  what changes the compiled code. Warm calls compile nothing.
- **Argument layout checks.** Each kernel checks a sentinel or layout version after its
  runtime arguments and returns at once on a mismatch, so a kernel source and a host
  built from different argument layouts cannot hang the card.
- **Ragged shapes** run on zeroed pad tiles; edges are per-core runtime arguments. A
  shape that is not a multiple of 32 never routes to a different, slower op.
- **No host reads and no host round-trips**, and nothing that differs between a trace's
  prepare pass and its capture (see [trace](trace.md)).
- **Mesh layouts are ordinary inputs.** A kernel runs on any per-chip shard (uneven
  splits, half the heads, odd K). A K-split variant writes fp32 partial sums in the
  layout the reduce-scatter reads (see [mesh](mesh.md)).
- **Forward and backward.** Each family ships the inference forward, the training
  forward (which saves what backward needs; it may be a different kernel) and the
  backward kernels.
- **dtypes.** Matmul and convolution take bf16, bfp8 and bfp4 operands; accumulation is
  always fp32.
- **Variants are expected.** One entry point per family picks among them; no kernel
  tries to cover every regime.

## The gate

A kernel replaces the TT-NN path for the shapes it claims when its sweep
(`python -m kohakutt.tools.sweep <family>`, or the family's sweep test under
`tests/hardware/ops/`) shows:

- the family's full shape grid runs, with no fallback to TT-NN, a composite or the host
  inside the claimed coverage;
- error statistics against CPU float64 no worse than CPU at the same dtype;
- a smooth curve: no drop of more than 15 % between neighbouring sizes (shapes below
  about 1 GFLOP are exempt);
- forward and backward tests pass;
- zero compiles in warm calls;
- every required dtype passes;
- it runs on one card and at the per-chip shapes of the 1x2 mesh;
- TT-NN's time for the same cell, measured in the same run, is no faster.

Until then the kernel stays behind its environment switch (for example `KTT_FLASH=1`), or
runs by default only on the sizes where it passes (`ktt_gemm` under `KTT_GEMM=auto`), and
the reference page states what it covers and how it compares.
