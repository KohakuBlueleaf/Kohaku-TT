// ktt_flash constants shared by the host program and the reader, compute and writer kernels.
#pragma once

#include <stdint.h>

namespace ktt_flash {

    // Circular buffers. Reader: q, k, v and the user mask (row: cb_um, broadcast: cb_mb).
    // Compute: its constant tiles (ones, identity, causal, column-0 ones, tail), P, the row
    // state, O, the step tokens and the output rows the writer drains.
    constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2;
    constexpr uint32_t cb_ones = 3;   // all 1.0: the row-sum column of P V
    constexpr uint32_t cb_p = 4;      // P of a step (bf16, SC slots)
    constexpr uint32_t cb_st = 5;     // row state tiles (bf16, two per row)
    constexpr uint32_t cb_o = 6;      // O and l per row (fp32, packer L1 accumulation)
    constexpr uint32_t cb_stok = 7;   // step token: P and state packed
    constexpr uint32_t cb_otok = 8;   // row token: the row's O complete
    constexpr uint32_t cb_ident = 9;  // identity: left operand of every mask term
    constexpr uint32_t cb_causal = 10;  // -1e30 above the diagonal
    constexpr uint32_t cb_um = 11;    // user mask tiles of a step (row mask)
    constexpr uint32_t cb_mb = 12;    // user mask row of a key block (broadcast mask)
    constexpr uint32_t cb_col0 = 13;  // column 0 = 1: spreads a broadcast mask row
    constexpr uint32_t cb_tail = 14;  // -1e30 in the key columns past the last key
    constexpr uint32_t cb_out = 16;   // normalized output rows (bf16)

    // Score tiles per step (the Dst half holds them and the row state tile).
    constexpr uint32_t kSubChunk = 3;

}  // namespace ktt_flash
