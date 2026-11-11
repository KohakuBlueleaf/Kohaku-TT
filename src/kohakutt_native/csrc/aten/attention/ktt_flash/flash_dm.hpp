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
