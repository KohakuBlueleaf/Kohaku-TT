// Column-sum writer: the core's column tiles t of the [1, W] output.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "args.hpp"

// Compile-time args: TensorAccessorArgs of the output. Common runtime args: output
// address, Wt, cores. Per core: first column tile.
void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();
    if (!ktt_norm::args_ok<args.next_compile_time_args_offset()>(4)) {
        return;
    }
    const uint32_t addr = get_common_arg_val<uint32_t>(0);
    const uint32_t wt = get_common_arg_val<uint32_t>(1);
    const uint32_t cores = get_common_arg_val<uint32_t>(2);
    const uint32_t start = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_out = 16;
    const auto out = TensorAccessor(args, addr, get_tile_size(cb_out));
    for (uint32_t t = start; t < wt; t += cores) {
        cb_wait_front(cb_out, 1);
        noc_async_write_page(t, out, get_read_ptr(cb_out));
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
}
