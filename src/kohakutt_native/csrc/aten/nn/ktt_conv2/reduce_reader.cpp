// Chunk-sum reader: per block b of 1024 fp32 values, the block of every chunk s (page
// s B + b of the partials [S B, 1024] fp32) into CB 0, one page each.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "../ktt_conv/args.hpp"

// Compile-time args: TensorAccessorArgs of the partials, then the argument-layout
// version. Common runtime args: version, partials address, block step, blocks B, chunks
// S. Per core: first block.
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
    const uint32_t chunks = get_common_arg_val<uint32_t>(4);
    const uint32_t start = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = 0;
    const auto part = TensorAccessor(args, addr, 4096);
    for (uint32_t b = start; b < blocks; b += step) {
        for (uint32_t s = 0; s < chunks; ++s) {
            cb_reserve_back(cb_in, 1);
            noc_async_read_page(s * blocks + b, part, get_write_ptr(cb_in));
            noc_async_read_barrier();
            cb_push_back(cb_in, 1);
        }
    }
}
