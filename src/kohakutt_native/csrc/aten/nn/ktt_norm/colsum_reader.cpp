// Column-sum reader: for each of the core's column tiles t, the partial-sum tiles (g,
// t) of every group g of the [groups * 32, W] fp32 tensor, read in blocks of up to 4
// per barrier.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "args.hpp"

// Compile-time args: TensorAccessorArgs of the partials, then the common arg count.
// Common runtime args: partials address, Wt, groups, cores. Per core: first column
// tile.
void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();
    if (!ktt_norm::args_ok<args.next_compile_time_args_offset()>(5)) {
        return;
    }
    const uint32_t addr = get_common_arg_val<uint32_t>(0);
    const uint32_t wt = get_common_arg_val<uint32_t>(1);
    const uint32_t groups = get_common_arg_val<uint32_t>(2);
    const uint32_t cores = get_common_arg_val<uint32_t>(3);
    const uint32_t start = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = 0, kTile = 4096, kBlock = 4;
    const auto p = TensorAccessor(args, addr, kTile);
    const auto& fifo = get_local_cb_interface(cb_in);
    for (uint32_t t = start; t < wt; t += cores) {
        uint32_t page = t;
        for (uint32_t g = 0; g < groups;) {
            // A block never crosses the end of the ring (pushes wrap only exactly at
            // it).
            const uint32_t to_end = (fifo.fifo_limit - get_write_ptr(cb_in)) / kTile;
            uint32_t n = groups - g < kBlock ? groups - g : kBlock;
            n = n < to_end ? n : to_end;
            cb_reserve_back(cb_in, n);
            const uint32_t l1 = get_write_ptr(cb_in);
            for (uint32_t b = 0; b < n; ++b, page += wt) {
                noc_async_read_page(page, p, l1 + b * kTile);
            }
            noc_async_read_barrier();
            cb_push_back(cb_in, n);
            g += n;
        }
    }
}
