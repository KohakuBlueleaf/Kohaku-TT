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

}  // namespace ktt_flash
