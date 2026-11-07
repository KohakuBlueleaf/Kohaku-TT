// Blocked tile reader: tiles [start, start + count) of one interleaved tensor into CB
// 0, `block` tiles per CB transaction with one read barrier per block.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Runtime args: source address, first tile, tile count, block size.
void kernel_main() {
    constexpr auto src_args = TensorAccessorArgs<0>();
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const uint32_t block = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb = 0;
    const uint32_t page = get_tile_size(cb);
    const auto src = TensorAccessor(src_args, src_addr, page);

    const uint32_t end = start + count;
    for (uint32_t t = start; t < end; t += block) {
        const uint32_t n = end - t < block ? end - t : block;
        cb_reserve_back(cb, n);
        uint32_t addr = get_write_ptr(cb);
        for (uint32_t k = 0; k < n; ++k) {
            noc_async_read_page(t + k, src, addr);
            addr += page;
        }
        noc_async_read_barrier();
        cb_push_back(cb, n);
    }
}
