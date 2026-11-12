// Columns writer: per unit (m, tap) the Ct tiles of CB 16 to tiles (m, tap Ct .. tap
// Ct + Ct - 1) of the columns [rows, KH KW Cp].
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Compile-time: the columns' accessor. Per core: first, end unit; common: columns
// address, Ct, taps (KH KW).
void kernel_main() {
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t u1 = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t ct = get_common_arg_val<uint32_t>(1);
    const uint32_t taps = get_common_arg_val<uint32_t>(2);
    constexpr uint32_t cb_out = 16;
    constexpr auto out_args = TensorAccessorArgs<0>();
    const uint32_t tile = get_tile_size(cb_out);
    const auto out = TensorAccessor(out_args, out_addr, tile);
    const uint32_t kt = taps * ct;

    for (uint32_t u = u0; u < u1; ++u) {
        const uint32_t m = u / taps, tap = u % taps;
        cb_wait_front(cb_out, ct);
        uint32_t src = get_read_ptr(cb_out);
        const uint32_t page = m * kt + tap * ct;
        for (uint32_t c = 0; c < ct; ++c) {
            noc_async_write_page(page + c, out, src);
            src += tile;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, ct);
    }
}
