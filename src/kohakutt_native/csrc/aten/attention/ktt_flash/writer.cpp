// Flash attention writer: the output rows (cb_out, DV tiles per row) of each job of
// each iteration; with FLASH_STATS first each row's log-sum-exp tile (cb_stats, fp32);
// with FLASH_DUMP first the dumped job's intermediate tiles (cb_dump, fp32).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "flash.hpp"

// Compile-time args: DV, then TensorAccessorArgs of O, the statistics and (FLASH_DUMP)
// the dump. Runtime args: O address, statistics address, iterations, jobs, per job:
// first O page, query tile rows, row stride; then (FLASH_DUMP) dump address, dumped
// job, dump pages.
void kernel_main() {
    using ktt_flash::cb_out;
    using ktt_flash::cb_stats;
    constexpr uint32_t DV = get_compile_time_arg_val(0);
    constexpr auto ao = TensorAccessorArgs<1>();
    [[maybe_unused]] constexpr auto
        as = TensorAccessorArgs<ao.next_compile_time_args_offset()>();

    const uint32_t page = get_tile_size(cb_out);
    const auto o = TensorAccessor(ao, get_arg_val<uint32_t>(0), page);
#ifdef FLASH_STATS
    const auto st = TensorAccessor(
        as, get_arg_val<uint32_t>(1), get_tile_size(cb_stats)
    );
#endif
    const uint32_t iters = get_arg_val<uint32_t>(2);
    const uint32_t jobs = get_arg_val<uint32_t>(3);
#ifdef FLASH_DUMP
    using ktt_flash::cb_dump;
    constexpr auto ad = TensorAccessorArgs<as.next_compile_time_args_offset()>();
    const auto dump = TensorAccessor(
        ad, get_arg_val<uint32_t>(4 + 3 * jobs), get_tile_size(cb_dump)
    );
    const uint32_t dump_job = get_arg_val<uint32_t>(5 + 3 * jobs);
    const uint32_t dump_pages = get_arg_val<uint32_t>(6 + 3 * jobs);
#endif

    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t job = 0; job < jobs; ++job) {
            const uint32_t o0 = get_arg_val<uint32_t>(4 + 3 * job);
            const uint32_t rows = get_arg_val<uint32_t>(5 + 3 * job);
            const uint32_t stride = get_arg_val<uint32_t>(6 + 3 * job);
#ifdef FLASH_DUMP
            if (it == 0 && job == dump_job) {
                for (uint32_t p = 0; p < dump_pages; ++p) {
                    cb_wait_front(cb_dump, 1);
                    noc_async_write_page(p, dump, get_read_ptr(cb_dump));
                    noc_async_write_barrier();
                    cb_pop_front(cb_dump, 1);
                }
            }
#endif
            for (uint32_t r = 0; r < rows; ++r) {
#ifdef FLASH_STATS
                cb_wait_front(cb_stats, 1);
                noc_async_write_page(o0 / DV + r * stride, st, get_read_ptr(cb_stats));
#endif
                cb_wait_front(cb_out, DV);
                const uint32_t base = get_read_ptr(cb_out);
                for (uint32_t c = 0; c < DV; ++c) {
                    noc_async_write_page(o0 + r * stride * DV + c, o, base + c * page);
                }
                noc_async_write_barrier();
#ifdef FLASH_STATS
                cb_pop_front(cb_stats, 1);
#endif
                cb_pop_front(cb_out, DV);
            }
        }
    }
}
