// ktt_flash constants shared by the host program and the reader, compute and writer
// kernels.
#pragma once

#include <stdint.h>

namespace ktt_flash {

    // Circular buffers. Reader: q, k, v and the user mask (row: cb_um, broadcast:
    // cb_mb). Compute: its constant tiles (ones, causal, tail, both), P, the row state,
    // O, the step tokens and the output rows the writer drains.
    constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2;
    constexpr uint32_t cb_ones = 3;  // all 1.0: the row-sum column of P V
    constexpr uint32_t cb_p = 4;     // P of a step (bf16, SC slots)
    constexpr uint32_t cb_st = 5;    // row state tiles (fp32, two per row)
    constexpr uint32_t cb_o = 6;     // O and l per row (fp32, packer L1 accumulation)
    constexpr uint32_t cb_stok = 7;  // step token: P and state packed
    constexpr uint32_t cb_otok = 8;  // row token: the row's O complete
    constexpr uint32_t
        cb_ctail = 9;  // causal diagonal and last key tile masks combined
    constexpr uint32_t cb_causal = 10;  // -1e30 above the diagonal
    constexpr uint32_t cb_um = 11;      // user mask tiles of a step (row mask)
    constexpr uint32_t cb_mb = 12;      // user mask row of a key block (broadcast mask)
    constexpr uint32_t cb_dump = 13;  // FLASH_DUMP: one row's intermediate tiles (fp32)
    constexpr uint32_t cb_tail = 14;  // -1e30 in the key columns past the last key
    constexpr uint32_t
        cb_lane = 15;  // fp32 lane sums of a step's P (statistics variant)
    constexpr uint32_t cb_out = 16;  // normalized output rows (bf16)
    constexpr uint32_t
        cb_stats = 17;  // a query tile row's log-sum-exp tile (fp32, every column)

    constexpr uint32_t
        cb_plan = 18;  // FLASH_VARLEN: the plan pages holding a job's row records
    // FLASH_LENGTHS: per job (key tiles streamed, keys in the last real key tile, real
    // key tiles) for the compute; the reader's copy of a key lengths tile.
    constexpr uint32_t cb_len = 19, cb_lens = 20;

    // Score tiles per step (the Dst half holds them and the row state tile).
    constexpr uint32_t kSubChunk = 3;

    // FLASH_VARLEN (packed variable-length attention). The plan tensor holds one record
    // of kRecordWords uint32 per query tile row, kRecordsPerPage per 4 KiB page: the
    // row's first and last key tile, the leading (a) and trailing (b) tiles of that
    // window taking a mask tile, and the first page of its a + b mask tiles. A job
    // reads the kPlanPages pages from its first row's on.
    constexpr uint32_t kRecordWords = 8, kRecordsPerPage = 128, kPlanPages = 2;

    // Zeroes rows [first, 32) of a bf16 tile (four 16 x 16 faces) at `half`.
    inline void zero_tile_rows(volatile uint16_t* half, uint32_t first) {
        for (uint32_t r = first; r < 32; ++r) {
            const uint32_t base = (r / 16) * 512 + (r % 16) * 16;
            for (uint32_t c = 0; c < 16; ++c) {
                half[base + c] = 0;
                half[base + 256 + c] = 0;
            }
        }
    }

    // Zeroes columns [first, 32) of row 0 of a bf16 tile at `half`.
    inline void zero_row0_cols(volatile uint16_t* half, uint32_t first) {
        for (uint32_t c = first; c < 32; ++c) {
            half[c < 16 ? c : 256 + c - 16] = 0;
        }
    }

    // Where the bf16 tile at `tile` holds zero: the value of row 0 of the bf16 tile at
    // `row` in that column.
    inline void fill_zeros_from_row0(
        volatile uint16_t* tile, const volatile uint16_t* row
    ) {
        for (uint32_t face = 0; face < 4; ++face) {
            volatile uint16_t* dst = tile + face * 256;
            const volatile uint16_t* src = row + (face % 2) * 256;
            for (uint32_t i = 0; i < 256; ++i) {
                if ((dst[i] & 0x7fff) == 0) {
                    dst[i] = src[i % 16];
                }
            }
        }
    }

    // A row's record with its tiles relative to the first tile of the job's stream (key
    // tiles; in the backward's key-side records query tiles, where last + 1 == first:
    // none).
    struct RowWindow {
        uint32_t first, last, a, b, u0;  // first / last relative to the stream

        bool empty() const {
            return last + 1 == first;
        }

        uint32_t tiles() const {
            return last - first + 1;
        }

        // True when the row's window tile `rel` (0 = its first key tile) takes a mask
        // tile.
        bool masked(uint32_t rel) const {
            return rel < a || rel >= tiles() - b;
        }

        // The mask page of window tile `rel` (masked(rel)).
        uint32_t mask_page(uint32_t rel) const {
            return rel < a ? u0 + rel : u0 + a + rel - (tiles() - b);
        }
    };

}  // namespace ktt_flash
