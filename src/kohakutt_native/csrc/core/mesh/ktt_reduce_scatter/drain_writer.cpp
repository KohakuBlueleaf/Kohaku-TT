// Drain writer of the 1x2 fp32 reduce-scatter: this drain's chunks of CB 16 to output
// tiles [first, first + count) (chunks d, d + drains, ... of `chunk` tiles).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/tensor_accessor_args.h"

// Compile-time args: output accessor, then the runtime-arg count. Runtime args: output
// address, first tile, tile count, tile bytes, chunk tiles, drains, this drain's index.
void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr uint32_t kArgs = get_compile_time_arg_val(out_args.next_compile_time_args_offset());
    constexpr uint32_t cb_out = 16;

    uint32_t i = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(i++);
    const uint32_t first = get_arg_val<uint32_t>(i++);
    const uint32_t count = get_arg_val<uint32_t>(i++);
    const uint32_t tile_bytes = get_arg_val<uint32_t>(i++);
    const uint32_t chunk = get_arg_val<uint32_t>(i++);
    const uint32_t drains = get_arg_val<uint32_t>(i++);
    const uint32_t d = get_arg_val<uint32_t>(i++);
    if (i != kArgs) {
        return;  // runtime-arg layout does not match this source
    }

    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);
    for (uint32_t t = d * chunk; t < count; t += drains * chunk) {
        const uint32_t n = count - t < chunk ? count - t : chunk;
        cb_wait_front(cb_out, chunk);
        uint32_t l1 = get_read_ptr(cb_out);
        for (uint32_t k = 0; k < n; ++k) {
            noc_async_write_page(first + t + k, out, l1);
            l1 += tile_bytes;
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, chunk);
    }
    noc_async_write_barrier();
}
