// Sender reader of the 1x2 fp32 reduce-scatter: this chip's tiles of the peer's half,
// tiles [first, first + count) after `offset` in the tile order, into CB 0 in batches
// of `chunk` tiles (one read barrier per batch).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/tensor_accessor_args.h"
#include "order.hpp"

// Compile-time args: source accessor, then the runtime-arg count. Runtime args: source
// address, tile offset, first tile, tile count, tile bytes, chunk tiles, row and column
// tiles of the column-major order (0, 0: tile order).
void kernel_main() {
    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr uint32_t kArgs = get_compile_time_arg_val(
        src_args.next_compile_time_args_offset()
    );
    constexpr uint32_t cb = 0;

    uint32_t i = 0;
    const uint32_t src_addr = get_arg_val<uint32_t>(i++);
    const uint32_t offset = get_arg_val<uint32_t>(i++);
    const uint32_t first = get_arg_val<uint32_t>(i++);
    const uint32_t count = get_arg_val<uint32_t>(i++);
    const uint32_t tile_bytes = get_arg_val<uint32_t>(i++);
    const uint32_t chunk = get_arg_val<uint32_t>(i++);
    const uint32_t rows = get_arg_val<uint32_t>(i++);
    const uint32_t cols = get_arg_val<uint32_t>(i++);
    if (i != kArgs) {
        return;  // runtime-arg layout does not match this source
    }

    const auto src = TensorAccessor(src_args, src_addr, tile_bytes);
    for (uint32_t t = 0; t < count; t += chunk) {
        const uint32_t n = count - t < chunk ? count - t : chunk;
        cb_reserve_back(cb, chunk);
        uint32_t l1 = get_write_ptr(cb);
        for (uint32_t k = 0; k < n; ++k) {
            noc_async_read_page(
                ktt_rs::page_of(offset + first + t + k, rows, cols), src, l1
            );
            l1 += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb, chunk);
    }
}
