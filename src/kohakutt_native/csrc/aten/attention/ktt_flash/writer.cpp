// Flash attention writer: the output rows (cb_out, DV tiles per row) of each job of each
// iteration.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "flash.hpp"

// Compile-time args: DV, then TensorAccessorArgs of O. Runtime args: O address, iterations,
// jobs, then per job: first O page, query tile rows, row stride.
void kernel_main() {
    using ktt_flash::cb_out;
    constexpr uint32_t DV = get_compile_time_arg_val(0);
    constexpr auto ao = TensorAccessorArgs<1>();

    const uint32_t page = get_tile_size(cb_out);
    const auto o = TensorAccessor(ao, get_arg_val<uint32_t>(0), page);
    const uint32_t iters = get_arg_val<uint32_t>(1);
    const uint32_t jobs = get_arg_val<uint32_t>(2);

    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t job = 0; job < jobs; ++job) {
            const uint32_t o0 = get_arg_val<uint32_t>(3 + 3 * job);
            const uint32_t rows = get_arg_val<uint32_t>(4 + 3 * job);
            const uint32_t stride = get_arg_val<uint32_t>(5 + 3 * job);
            for (uint32_t r = 0; r < rows; ++r) {
                cb_wait_front(cb_out, DV);
                const uint32_t base = get_read_ptr(cb_out);
                for (uint32_t c = 0; c < DV; ++c) {
                    noc_async_write_page(o0 + r * stride * DV + c, o, base + c * page);
                }
                noc_async_write_barrier();
                cb_pop_front(cb_out, DV);
            }
        }
    }
}
