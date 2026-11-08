// Conv writer: per output row tile m, this core's nt column tiles (CB 16) to the output
// [N OH Wp, O] at tiles (m, n0 .. n0 + nt - 1).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Compile-time args: TensorAccessorArgs of the output. Runtime args (per core: m0, m1,
// n0, nt; the rest common): output address, first / end output row tile, first column
// tile, column tiles, column tiles in all.
void kernel_main() {
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t m0 = get_arg_val<uint32_t>(0);
    const uint32_t m1 = get_arg_val<uint32_t>(1);
    const uint32_t n0 = get_arg_val<uint32_t>(2);
    const uint32_t nt = get_arg_val<uint32_t>(3);
    const uint32_t ot = get_common_arg_val<uint32_t>(1);
    constexpr uint32_t cb_out = 16;
    constexpr auto out_args = TensorAccessorArgs<0>();
    const uint32_t tile = get_tile_size(cb_out);
    const auto out = TensorAccessor(out_args, out_addr, tile);

    for (uint32_t m = m0; m < m1; ++m) {
        cb_wait_front(cb_out, nt);
        uint32_t src = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < nt; ++j) {
            noc_async_write_page(m * ot + n0 + j, out, src);
            src += tile;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, nt);
    }
}
