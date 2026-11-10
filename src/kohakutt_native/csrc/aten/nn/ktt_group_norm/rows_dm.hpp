// Data-movement helper of the row-form group norm kernels: tile rows zeroed in L1.
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace ktt_group_norm {

    // Zeroes rows [r0, r1) of the tile at L1 address `addr` (`elem` bytes per element)
    // with NoC reads of the zero region.
    inline void zero_rows(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t elem) {
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        const uint32_t row = 16 * elem, face = 256 * elem;
        for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t lo = r0 > 16 * half ? r0 - 16 * half : 0;
            const uint32_t hi = r1 > 16 * half
                ? (r1 - 16 * half < 16 ? r1 - 16 * half : 16)
                : 0;
            if (lo >= hi) {
                continue;
            }
            for (uint32_t f = 2 * half; f < 2 * half + 2; ++f) {
                const uint32_t base = addr + f * face + lo * row;
                const uint32_t bytes = (hi - lo) * row;
                for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
                    const uint32_t n = bytes - off < MEM_ZEROS_SIZE ? bytes - off
                                                                    : MEM_ZEROS_SIZE;
                    noc_async_read(zeros, base + off, n);
                }
            }
        }
        noc_async_read_barrier();
    }

}  // namespace ktt_group_norm
