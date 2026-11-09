---
title: The Blackhole p150a
summary: The chip as a kernel author must see it - grid and harvesting, where DRAM and the other tiles sit, the two NoCs and which one a stream should use, multicast, and the Tensix tile's memories and units.
tags:
  - concepts
  - hardware
  - kernels
---

# The Blackhole p150a

Kohaku-TT runs on Tenstorrent p150a cards (one Blackhole chip each). This page describes
the chip from the point of view of someone writing a Metalium program for it: where
things sit, how data moves, and what each unit inside a core can do per cycle. The
[kernel design](kernel-design.md) page builds on it.

Sources, cited by name below:

- **ISA docs**: Tenstorrent's `tt-isa-documentation`, `BlackholeA0/` (NoC, Tensix tile,
  baby RISC-V). Where the Blackhole tree defers to the Wormhole tree (L1, SrcA / SrcB,
  Dst, unpackers, packers), the Wormhole figures are quoted and marked so.
- **SoC descriptor**: `tt_metal/soc_descriptors/blackhole_140_arch.yaml` in TT-Metal.
- **TT-Metal sources**: `tt_metal/hw/inc/internal/dataflow/dataflow_api_raw.h` and the
  dispatch configuration.
- **ttx findings**: the multicast and harvesting measurements of the `tt-x280-dispatch`
  project (`docs/findings.md`), which drives the same cards from the on-chip X280 cores.
- **Blackhole architecture guide**: "tenstorrent blackhole architecture guide",
  `anuraagw.me/blog/blackhole-architecture` (an independent description of the chip;
  consistent with the layout and NoC directions below).
- **p150 specification**: Tenstorrent's product specification for the p150 board
  (DRAM bandwidth).

Clock: 1.35 GHz for the RISC-V cores and the NoC (ISA docs).

## The grid

The chip is a 17 x 12 grid of tiles addressed by NoC0 coordinates (x 0-16, y 0-11)
(SoC descriptor):

```
         x  0   1  2  3  4  5  6   7   8   9   10  11 12 13 14 15  16
row 0       D   .  P  .  .  .  .   .   A   D   .   P  .  .  .  .   .
row 1       D   E  E  E  E  E  E   E   L   D   E   E  E  E  E  E   E
rows 2-11   D   T  T  T  T  T  T   h   L   D   h   T  T  T  T  T   t

D  DRAM tile            E  Ethernet tile         T  Tensix, program grid
P  PCIe                 L  L2CPU / security      t  Tensix kept by TT-Metal for dispatch
A  ARC                  .  router only           h  Tensix fused off (harvested)
```

- **Tensix tiles.** The full die has 140: columns 1-7 and 10-16, rows 2-11. Our p150a
  cards are harvested with mask `0b11000000`, which fuses off NoC0 columns 7 and 10, so
  120 Tensix tiles remain (12 columns x 10 rows). TT-Metal's fast dispatch keeps one
  more column (NoC0 x 16) for its dispatch kernels, so a program gets an 11 x 10 grid of
  110 cores (`compute_with_storage_grid_size`).
- **Logical, translated and physical coordinates.** Programs address cores by logical
  (x, y). Logical columns 0-5 are NoC0 x 1-6; logical columns 6-10 are NoC0 x 11-15;
  logical rows 0-9 are NoC0 rows 2-11. The translated coordinates TT-Metal hands to
  kernels are not physical either: translated x 7 is NoC0 x 11 on these cards (ttx
  findings). A rectangle that looks contiguous in logical or translated coordinates can
  span the middle of the chip.
- **Two panels.** The Tensix array is two physical panels: x 1-6 (left) and x 11-15
  (right, x 16 being the dispatch column). Between them sit four columns with no usable
  Tensix: 7 (fused), 8 (L2CPU, security, ARC), 9 (DRAM) and 10 (fused).
- **DRAM.** 8 GDDR6 channels of 4 GiB, each exposed on 3 DRAM tiles: channels 0-3 in
  column 0, channels 4-7 in column 9 (SoC descriptor). Interleaved buffers place page p
  on channel p mod 8, so every interleaved tensor has half of its pages on each DRAM
  column. Total bandwidth 512 GB/s (p150 specification), 64 GB/s per channel.
- **Ethernet** occupies row 1. A Tensix tile in row 2 reaches the Ethernet tile above it
  in one hop on NoC1 (which travels up first); the Ethernet tile reaches it in one hop on
  NoC0 (which travels down).
- **Row 0** holds PCIe (x 2 and 11) and ARC (x 8); the rest of the row is routers only.

## The two NoCs

Every tile sits on two physically separate 2-D torus networks (ISA docs, NoC):

| | NoC0 | NoC1 |
|---|---|---|
| route | +x, then +y (right, then down) | -y, then -x (up, then left) |
| turns | at most one | at most one |
| edges | wrap (torus) | wrap (torus) |
| link | one 512-bit flit per cycle (64 B / cycle, about 86 GB/s at 1.35 GHz) | same |
| router hop | 9 cycles, cut-through | same |
| NIU to router | about 5 cycles | same |

A packet is one header flit and up to 256 data flits (16 KiB). Congestion happens only
when two packets need the same link; paths that cross inside a router do not interfere
(ISA docs, RoutingPaths). Read responses and acknowledgements travel on the same NoC as
the request, from the responder back to the requester. So a read's data follows the
responder-to-reader path of the NoC it was issued on, and a write's data follows the
writer-to-target path.

### Which NoC a DRAM transfer should use

Since a route is fixed by its direction, the shorter path to a DRAM column depends on
which side of it the core sits. For reads (data from DRAM to the core):

| Reader | Pages on column 0 (channels 0-3) | Pages on column 9 (channels 4-7) |
|---|---|---|
| left panel (x 1-6) | NoC0: right 1-6 hops, then down the reader's column | NoC1: up column 9, then left 3-8 hops |
| right panel (x 11-15) | NoC1: up column 0, then left through the wrap at x 16, 1-6 hops | NoC0: right 2-6 hops, then down the reader's column |

The other choice in each cell sends the data most of the way around the torus and
across the other panel's links. Writes (data from the core to DRAM) mirror the table:
a left-panel core writes column 0 on NoC1 and column 9 on NoC0; a right-panel core
writes column 0 on NoC0 and column 9 on NoC1.

The table counts horizontal hops only. It holds when readers are spread over many
columns, where the vertical legs are spread too: with every core reading evenly, per-page
NoC choice reaches the DRAM limit (512 GB/s), while all reads on NoC0 cap near 447 GB/s
(channels 0 and 4 share the NoC0 exit link of row y=11) and all on NoC1 near 288 GB/s
(the four channels of a DRAM column share that column's vertical NoC1 links). For a single
column of readers the vertical legs dominate instead (NoC0 funnels down the readers'
column, NoC1 up the DRAM columns), so the per-stream choice has to be computed from the
full routes, not read off this table.

Two consequences:

- Because interleaving puts half of every tensor on each column, one core reading a
  whole tensor uses both NoCs. The NoC is chosen per stream (or per page) by the
  direction its data travels, not by role: "reads on one NoC, writes on the other" is
  wrong on this chip.
- A whole row of cores reading the same DRAM column on NoC0 shares the row links next
  to the DRAM tile; a whole column of cores on NoC1 shares the column links. The plan
  for a kernel counts bytes per link, not per core.

## Multicast

A multicast (a NoC broadcast) targets a rectangle of tiles, and only Tensix tiles
receive it (ISA docs, NoC). Every multicast TT-Metal's dataflow API issues sets path
reservation (`multicast_path_reserve = true` in `dataflow_api_raw.h`): all routers of
the broadcast tree reserve a virtual channel before data leaves the sender, and a
refused reservation is retried by hardware with randomized exponential backoff (ISA
docs). The ISA docs also record a Blackhole defect: a path-reserved broadcast refused at
a router two or more hops from the sender, because another broadcast holds the virtual
channel it needs, can lose a flit and hang the NoC; software should keep other
broadcasts off its routers while one is in flight.

A rectangle must stay inside one physical panel. On these cards logical (or translated)
columns that look contiguous cross the fused, L2CPU and DRAM columns: ttx measured that
multicasts from the X280 over such a rectangle lost or zeroed a command every 1e5 to 1e6
commands, and splitting them into per-panel rectangles gave no failure in more than
16 million (ttx findings). Tensix-originated multicasts over the same span happened to
work in those runs, which is not a guarantee. Unicast is not affected: it routes around
harvested columns.

## Inside a Tensix tile

| Unit | Size and rate | What it means for a kernel |
|---|---|---|
| L1 | 1536 KiB of SRAM in 128-bit banks (`worker_l1_size` 1572864); one 128-bit read or write per bank per cycle (ISA docs) | firmware, kernel binaries and the allocator's base take part of it; circular buffers (CBs) are sized before launch |
| narrow writes | any write under 128 bits is an atomic read-modify-write that holds its bank and port for 5 cycles; RISC-V stores to L1 are always narrow: one 32-bit store per 5 cycles (ISA docs, Wormhole L1) | tile data is never moved by RISC-V loops; it moves by NoC, unpacker or packer |
| RISC-V | 5 in-order cores, RV32IM, 1.35 GHz: two data-movement cores (BRISC, NCRISC) and three that drive the coprocessor (TRISC0 unpack, TRISC1 math, TRISC2 pack); `div` / `rem` take 6-33 cycles (ISA docs, baby RISC-V) | a kernel is three programs (reader, compute, writer) that meet only in CBs; inner data-movement loops advance indices by comparison, not division |
| unpackers | 2, L1 to SrcA / SrcB / Dst: one alone up to four 128-bit reads per cycle, both together five (Wormhole figures, ISA docs) | operand bandwidth into the FPU |
| SrcA / SrcB | 2 banks x 64 rows x 16 columns of 19-bit values each (Wormhole figures); the unpacker fills one bank while the FPU uses the other | FPU operands carry at most 19 bits: fp32 operands are never multiplied exactly on the FPU |
| FPU (matrix unit) | `MVMUL`: 8x16 @ 16x16 accumulated into Dst per cycle per fidelity phase (4096 FLOP / cycle); a 32 x 32 x 32 tile product is 16 `MVMUL` per phase, 32 cycles at HiFi2 | 5.5 TFLOPS per core at LoFi, 2.76 at HiFi2; 304 TFLOPS at HiFi2 on the 110-core program grid |
| fidelity | each phase multiplies 5 bits of the SrcA mantissa by 7 bits of SrcB (ISA docs, MatrixUnitArithmetic); LoFi one phase, HiFi2 two, HiFi3 three, HiFi4 four | a bf16 operand needs phases 0 and 1 in SrcA but 0 and 2 in SrcB: HiFi2 drops the SrcB operand's last mantissa bit, HiFi3 misses only the product of the two low parts (below 2^-11 relative), HiFi4 is exact; see [numerics](numerics.md) |
| Dst | 1024 rows x 16 columns of 16 bit (16 tiles) or 512 x 16 of 32 bit (8 tiles); in half-sync mode (the default) math owns one half while pack drains the other: 8 bf16 or 4 fp32 tiles per half | a matmul subblock with fp32 accumulation holds at most 4 output tiles |
| SFPU | 32 lanes x 32-bit SIMD on Dst; about 1/64 of the FPU's multiply-add rate | exponentials and other transcendentals are overlapped with FPU work, never run as a separate serial pass |
| packers | 4, Dst to L1, one 128-bit write per cycle each; L1 accumulation costs one 128-bit accumulate per 5 cycles (atomic) or 2 (non-atomic) (Wormhole figures) | fp32 partial sums accumulated in L1 are slower than plain packs and are budgeted |
| tile formats | 32 x 32: bf16 2048 B, bfp8 1088 B, bfp4 576 B, fp32 4096 B | block-float tiles fit a larger K block in the same L1 |

## Limits of this page

- Rates marked "Wormhole figures" come from the Wormhole tree of the ISA docs, which
  the Blackhole tree defers to; Blackhole may differ in detail.
- The per-panel NoC table assumes interleaved DRAM buffers and the harvesting of our
  cards (mask `0b11000000`). A card harvested differently has different panels; code
  derives panels from the SoC descriptor's coordinate translation, never from constants.
- The 512 GB/s DRAM figure is the board specification. A single core's DRAM read rate
  is far lower than one NoC link: 14-28 GB/s per reading core in Kohaku-TT's matmul
  (device profiler), so one reader is bound by its outstanding requests, not by the
  link, and wide kernels spread their DRAM reads over many cores.
