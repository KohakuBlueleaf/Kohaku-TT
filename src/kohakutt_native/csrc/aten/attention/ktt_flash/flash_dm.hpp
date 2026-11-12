// ktt_flash data movement helpers shared by the reader kernels (device side only).
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace ktt_flash {

    // Issues NoC reads of the core's zero region over rows [first, 32) of the bf16
    // tile at L1 `addr` (four 16 x 16 faces); complete after noc_async_read_barrier.
    inline void zero_rows_async(uint32_t addr, uint32_t first) {
        static_assert(MEM_ZEROS_SIZE >= 512);
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t face = 0; face < 4; ++face) {
            const uint32_t r0 = face < 2 ? first : (first > 16 ? first - 16 : 0);
            if (r0 < 16) {
                noc_async_read(zeros, addr + face * 512 + r0 * 32, (16 - r0) * 32);
            }
        }
    }

    // Zeroes columns [first, 32) of the bf16 tile at L1 `addr`: zero-region NoC reads
    // (16 B aligned; done after noc_async_read_barrier), stores when first % 8 != 0.
    inline void zero_cols(uint32_t addr, uint32_t first) {
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t face = 0; face < 4; ++face) {
            const uint32_t c0 = face % 2 == 0 ? first : (first > 16 ? first - 16 : 0);
            if (c0 >= 16) {
                continue;
            }
            const uint32_t base = addr + face * 512;
            if (c0 == 0) {
                noc_async_read(zeros, base, 512);
                continue;
            }
            for (uint32_t r = 0; r < 16; ++r) {
                if (c0 % 8 == 0) {
                    noc_async_read(zeros, base + r * 32 + c0 * 2, (16 - c0) * 2);
                    continue;
                }
                auto* half = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                    base + r * 32
                );
                for (uint32_t c = c0; c < 16; ++c) {
                    half[c] = 0;
                }
            }
        }
    }

    // bf16 `x` times the fp32 constant `c` (positive, normal) rounded to nearest even,
    // in integers (the RISCs have no FPU); results below the normal range are zero.
    inline uint32_t scale_bf16(uint32_t x, uint32_t c) {
        const uint32_t sign = x & 0x8000u, ex = (x >> 7) & 0xFFu;
        if (ex == 0 || ex == 0xFFu) {
            return ex == 0 ? sign : x;  // zero / subnormal; inf / NaN
        }
        // (1.m_x 2^7)(1.m_c 2^23): bit 30 or 31 leads; keep 8 bits.
        const uint32_t p = ((x & 0x7Fu) | 0x80u) * ((c & 0x7FFFFFu) | 0x800000u);
        const uint32_t top = p >> 31;
        int32_t e = static_cast<int32_t>(ex + ((c >> 23) & 0xFFu)) - 127 +
            static_cast<int32_t>(top);
        const uint32_t shift = 23 + top, half = 1u << (shift - 1);
        uint32_t m = p >> shift;
        const uint32_t rem = p & ((1u << shift) - 1);
        if (rem > half || (rem == half && (m & 1u))) {
            if (++m == 0x100u) {
                m >>= 1;
                ++e;
            }
        }
        if (e <= 0) {
            return sign;
        }
        return e >= 0xFF ? (sign | 0x7F80u)
                         : (sign | (static_cast<uint32_t>(e) << 7) | (m & 0x7Fu));
    }

    // Row 0 of a raw additive bf16 mask tile at `addr`, in place: <= -floor (and the
    // columns from `tail` on, 0: none) to `masked`, the rest times `inv` (1 / scale).
    inline void raw_mask_row(
        uint32_t addr, uint32_t tail, uint32_t floor, uint32_t masked, uint32_t inv
    ) {
        auto* half = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr);
        for (uint32_t c = 0; c < 32; ++c) {
            const uint32_t at = c < 16 ? c : 256 + c - 16;  // faces 0 and 1, row 0
            const uint32_t x = half[at], mag = x & 0x7FFFu;
            uint32_t y;
            if ((tail != 0 && c >= tail) ||
                ((x & 0x8000u) && mag >= floor && mag <= 0x7F80u)) {
                y = masked;
            } else if (mag == 0 || mag > 0x7F80u) {
                y = mag == 0 ? 0u : x;  // zero; NaN stays
            } else {
                y = scale_bf16(x, inv);
            }
            half[at] = static_cast<uint16_t>(y);
        }
    }

    // Writes the bf16 mask tile at L1 `addr` (four 16 x 16 faces): -1e30 in the
    // columns from `tail` on and, with `causal`, above the diagonal; 0 elsewhere.
    inline void mask_tile(uint32_t addr, uint32_t tail, bool causal) {
        constexpr uint32_t kMasked = 0xF14A;  // bf16 -1e30
        auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
        for (uint32_t r = 0; r < 32; ++r) {
            for (uint32_t c = 0; c < 32; c += 2) {
                const bool m0 = c >= tail || (causal && c > r);
                const bool m1 = c + 1 >= tail || (causal && c + 1 > r);
                words
                    [((r / 16) * 2 + c / 16) * 128 + (r % 16) * 8 +
                     (c % 16) / 2] = (m0 ? kMasked : 0u) | ((m1 ? kMasked : 0u) << 16);
            }
        }
    }

    // Pushes mask_tile(tail) to `cb_t` and, with `causal`, its causal form to `cb_c`.
    inline void push_mask_tiles(
        uint32_t cb_t, uint32_t cb_c, uint32_t tail, bool causal
    ) {
        cb_reserve_back(cb_t, 1);
        mask_tile(get_write_ptr(cb_t), tail, false);
        cb_push_back(cb_t, 1);
        if (causal) {
            cb_reserve_back(cb_c, 1);
            mask_tile(get_write_ptr(cb_c), tail, true);
            cb_push_back(cb_c, 1);
        }
    }

}  // namespace ktt_flash
