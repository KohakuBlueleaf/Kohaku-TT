// Chunk-sum writer: per block b, its 1024 summed values (CB 16: fp32, or bf16 for a
// split gather's output tiles) to page b of the sums.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "../ktt_conv/args.hpp"

// Compile-time args: TensorAccessorArgs of the sums, then the argument-layout version.
// Common runtime args: version, sums address, block step, blocks. Per core: first
// block.
void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();
    if (get_compile_time_arg_val(args.next_compile_time_args_offset()) !=
            ktt_conv::kArgsVersion ||
        get_common_arg_val<uint32_t>(0) != ktt_conv::kArgsVersion) {
        return;
    }
    const uint32_t addr = get_common_arg_val<uint32_t>(1);
    const uint32_t step = get_common_arg_val<uint32_t>(2);
    const uint32_t blocks = get_common_arg_val<uint32_t>(3);
    const uint32_t start = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_out = 16;
    const auto sums = TensorAccessor(args, addr, get_tile_size(cb_out));
    for (uint32_t b = start; b < blocks; b += step) {
        cb_wait_front(cb_out, 1);
        noc_async_write_page(b, sums, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
