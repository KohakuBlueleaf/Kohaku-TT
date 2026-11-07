// Blocked tile writer: tiles [start, start + count) of one interleaved tensor from
// CB 16, `block` tiles per CB transaction.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Runtime args: destination address, first tile, tile count, block size.
void kernel_main() {
    constexpr auto dst_args = TensorAccessorArgs<0>();
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const uint32_t block = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb = 16;
    const uint32_t page = get_tile_size(cb);
    const auto dst = TensorAccessor(dst_args, dst_addr, page);

    const uint32_t end = start + count;
    for (uint32_t t = start; t < end; t += block) {
        const uint32_t n = end - t < block ? end - t : block;
        cb_wait_front(cb, n);
        uint32_t addr = get_read_ptr(cb);
        for (uint32_t k = 0; k < n; ++k) {
            noc_async_write_page(t + k, dst, addr);
            addr += page;
        }
        noc_async_writes_flushed();
        cb_pop_front(cb, n);
    }
    noc_async_write_barrier();
}
