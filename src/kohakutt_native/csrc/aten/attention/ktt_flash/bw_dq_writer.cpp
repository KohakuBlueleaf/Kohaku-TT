// Flash attention backward dQ writer: per job, the NST statistics tiles of each query
// row (fp32, flash_bw.hpp), then the DT dQ tiles of each query row.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "flash_bw.hpp"

// Compile-time args: DT, NST, then TensorAccessorArgs of dQ and the statistics. Runtime
// args: dQ and statistics addresses, iterations, jobs, then per job: first query tile
// row (global), query tile rows, row stride.
void kernel_main() {
    using namespace ktt_flash_bw::dq;
    constexpr uint32_t DT = get_compile_time_arg_val(0);
    constexpr uint32_t NST = get_compile_time_arg_val(1);
    constexpr auto adq = TensorAccessorArgs<2>();
    constexpr auto ad = TensorAccessorArgs<adq.next_compile_time_args_offset()>();
    const uint32_t page = get_tile_size(cb_out);
    const uint32_t page32 = get_tile_size(cb_dout);
    const auto dq = TensorAccessor(adq, get_arg_val<uint32_t>(0), page);
    const auto dd = TensorAccessor(ad, get_arg_val<uint32_t>(1), page32);
    const uint32_t iters = get_arg_val<uint32_t>(2);
    const uint32_t jobs = get_arg_val<uint32_t>(3);
    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t job = 0; job < jobs; ++job) {
            const uint32_t row0 = get_arg_val<uint32_t>(4 + 3 * job);
            const uint32_t rows = get_arg_val<uint32_t>(5 + 3 * job);
            const uint32_t stride = get_arg_val<uint32_t>(6 + 3 * job);
            for (uint32_t r = 0; r < rows; ++r) {
                cb_wait_front(cb_dout, NST);
                const uint32_t row = row0 + r * stride;
                for (uint32_t t = 0; t < NST; ++t) {
                    noc_async_write_page(
                        NST * row + t, dd, get_read_ptr(cb_dout) + t * page32
                    );
                }
                noc_async_write_barrier();
                cb_pop_front(cb_dout, NST);
            }
            for (uint32_t r = 0; r < rows; ++r) {
                cb_wait_front(cb_out, DT);
                const uint32_t base = get_read_ptr(cb_out);
                for (uint32_t d = 0; d < DT; ++d) {
                    noc_async_write_page(
                        (row0 + r * stride) * DT + d, dq, base + d * page
                    );
                }
                noc_async_write_barrier();
                cb_pop_front(cb_out, DT);
            }
        }
    }
}
