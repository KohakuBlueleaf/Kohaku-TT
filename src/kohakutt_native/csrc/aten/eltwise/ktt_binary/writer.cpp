// Broadcast binary writer: output tiles of the (j, r) pair range from CB 16 to tile
// r * sr + j * sj, in the reader's blocks.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Runtime args: destination address, first pair, pair count, repeats per j (nr), stride
// per r (sr), stride per j (sj), block size.
void kernel_main() {
    constexpr auto dst_args = TensorAccessorArgs<0>();
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const uint32_t nr = get_arg_val<uint32_t>(3);
    const uint32_t sr = get_arg_val<uint32_t>(4);
    const uint32_t sj = get_arg_val<uint32_t>(5);
    const uint32_t block = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb = 16;
    const uint32_t page = get_tile_size(cb);
    const auto dst = TensorAccessor(dst_args, dst_addr, page);

    const uint32_t end = start + count;
    uint32_t idx = start;
    while (idx < end) {
        const uint32_t j = idx / nr, r = idx % nr;
        uint32_t n = nr - r;
        n = n < block ? n : block;
        n = n < end - idx ? n : end - idx;
        cb_wait_front(cb, n);
        uint32_t addr = get_read_ptr(cb);
        for (uint32_t k = 0; k < n; ++k) {
            noc_async_write_page((r + k) * sr + j * sj, dst, addr);
            addr += page;
        }
        noc_async_writes_flushed();
        cb_pop_front(cb, n);
        idx += n;
    }
    noc_async_write_barrier();
}
