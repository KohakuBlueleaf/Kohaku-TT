// Circular buffers of the backward programs, shared by the host program and the
// kernels.
#pragma once

#include <stdint.h>

namespace ktt_flash_bw {

    // Statistics of a query tile row (fp32), with w = 639 - lse log2 e and D =
    // rowsum(dO O). Row tile: w at the even columns, D at the odd columns of each row.
    // Column tiles (FLASH_BW_KQ): w, then D, of query row c in every row of column c.
    //
    // dQ program. Reader: q (R rows), k / v (key blocks, as the forward), dO and O
    // rows, the rows' log-sum-exp tiles (fp32). Compute: ones and causal tiles, dO * O
    // (bf16), dS (bf16), dQ accumulators (fp32), the rows' row statistics tiles (fp32),
    // dQ rows out (bf16), statistics tiles out (fp32: the row tile, or with FLASH_BW_KQ
    // the two column tiles).
    namespace dq {
        constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_do = 3, cb_oin = 4,
                           cb_lse = 5, cb_ones = 6;
        constexpr uint32_t cb_prod = 7, cb_tok = 8, cb_ds = 9, cb_acc = 10,
                           cb_causal = 12;
        constexpr uint32_t cb_w = 13, cb_out = 16, cb_dout = 17;
        // Mask tiles of a ragged length: key columns past the last key (cb_tail; on the
        // causal diagonal tile cb_ctail), query rows past the last query (cb_rtail).
        constexpr uint32_t cb_tail = 11, cb_ctail = 14, cb_rtail = 15;
        // FLASH_VARLEN: a row's mask tiles of a chunk, and the plan pages of a job's
        // rows.
        constexpr uint32_t cb_um = 18, cb_plan = 19;
        // FLASH_MASK_BCAST (with FLASH_VARLEN): the key mask's row tiles of a key
        // block.
        constexpr uint32_t cb_mb = 20;
        // FLASH_LENGTHS: per job (key tiles streamed, keys in the last real key tile,
        // real key tiles) for the compute; the reader's copy of a key lengths tile.
        constexpr uint32_t cb_len = 21, cb_lens = 22;
    }  // namespace dq

    // dK / dV program. Reader: k / v rows (R rows), query blocks of Q and dO and their
    // statistics tiles (fp32). Compute: the causal tile, P and dS of a chunk (bf16), dK
    // / dV accumulators (fp32), dK and dV rows out (bf16); without FLASH_BW_KQ also Q^T
    // and dO^T of a chunk (cb_qt, cb_dot) and a finished row before it is turned
    // (cb_fin).
    namespace dkv {
        constexpr uint32_t cb_k = 0, cb_v = 1, cb_q = 2, cb_do = 3, cb_tok = 4,
                           cb_wd = 5;
        constexpr uint32_t cb_qt = 6, cb_dot = 7, cb_fin = 8;
        constexpr uint32_t cb_p = 9, cb_ds = 10, cb_dk = 11, cb_dv = 12;
        constexpr uint32_t cb_causal = 13, cb_out = 16;
        // Mask tiles of the last key tile of a ragged length (default orientation).
        constexpr uint32_t cb_tail = 14, cb_ctail = 15;
        // FLASH_VARLEN: a key row's mask tiles of a chunk, and the plan pages of a
        // job's rows.
        constexpr uint32_t cb_um = 17, cb_plan = 18;
        // FLASH_MASK_BCAST (with FLASH_VARLEN): the key mask's row tile of each key
        // row.
        constexpr uint32_t cb_mb = 19;
        // FLASH_LENGTHS: per job (keys in the last real key tile, real key tiles) for
        // the compute; the reader's copy of a key lengths tile.
        constexpr uint32_t cb_len = 20, cb_lens = 21;
    }  // namespace dkv

    // cb_tok (both programs): a token the pack thread pushes once the tiles it packed
    // into slots of a reserved buffer (statistics, accumulators) are in L1; the
    // unpacker waits for it.

    // Key tiles per dQ accumulation and query tiles per dK / dV accumulation.
    constexpr uint32_t kChunk = 6;

    // Tiles of one Dst section (fp32 Dst, half-sync).
    constexpr uint32_t kDstTiles = 4;

}  // namespace ktt_flash_bw
