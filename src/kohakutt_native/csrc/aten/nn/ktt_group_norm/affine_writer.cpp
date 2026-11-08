// Group norm affine writer: per plane its output tiles (CB 16) in order.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Compile-time args: TensorAccessorArgs of y. Runtime args (per core: start; the rest
// common): y address, first plane, plane step, planes, tiles per plane.
void kernel_main() {
    const uint32_t y_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t step = get_common_arg_val<uint32_t>(1);
    const uint32_t planes = get_common_arg_val<uint32_t>(2);
    const uint32_t tiles = get_common_arg_val<uint32_t>(3);
    constexpr uint32_t cb_out = 16;
    constexpr auto y_args = TensorAccessorArgs<0>();
    const auto y = TensorAccessor(y_args, y_addr, get_tile_size(cb_out));

    for (uint32_t p = start; p < planes; p += step) {
        for (uint32_t t = 0; t < tiles; ++t) {
            cb_wait_front(cb_out, 1);
            noc_async_write_page(p * tiles + t, y, get_read_ptr(cb_out));
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
