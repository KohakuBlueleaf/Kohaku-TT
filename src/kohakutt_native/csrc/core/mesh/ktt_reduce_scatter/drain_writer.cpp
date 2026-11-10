// Drain writer of the 1x2 collectives: this drain's chunks of CB 16 to output tiles
// [first, first + count) (chunks d, d + drains, ... of `chunk` tiles). With a progress
// block, after the drain's last chunk of each block of `block` tiles is written it
// increments a semaphore on each listed core.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/tensor_accessor_args.h"
#include "order.hpp"

constexpr uint32_t kMaxWatchers = 16;

// Compile-time args: output accessor, then the runtime-arg count before the watcher
// list. Runtime args: output address, first tile, tile count, tile bytes, chunk tiles,
// drains, this drain's index, row and column tiles of the column-major order (0, 0:
// tile order), tiles per progress block (0: none), progress semaphore id, watchers,
// then x / y of each watcher.
void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr uint32_t kArgs = get_compile_time_arg_val(
        out_args.next_compile_time_args_offset()
    );
    constexpr uint32_t cb_out = 16;

    uint32_t i = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(i++);
    const uint32_t first = get_arg_val<uint32_t>(i++);
    const uint32_t count = get_arg_val<uint32_t>(i++);
    const uint32_t tile_bytes = get_arg_val<uint32_t>(i++);
    const uint32_t chunk = get_arg_val<uint32_t>(i++);
    const uint32_t drains = get_arg_val<uint32_t>(i++);
    const uint32_t d = get_arg_val<uint32_t>(i++);
    const uint32_t rows = get_arg_val<uint32_t>(i++);
    const uint32_t cols = get_arg_val<uint32_t>(i++);
    const uint32_t block = get_arg_val<uint32_t>(i++);
    const uint32_t progress = get_arg_val<uint32_t>(i++);
    const uint32_t watchers = get_arg_val<uint32_t>(i++);
    if (i != kArgs || watchers > kMaxWatchers) {
        return;  // runtime-arg layout does not match this source
    }
    uint64_t watcher_noc[kMaxWatchers];
    for (uint32_t w = 0; w < watchers; ++w) {
        const uint32_t x = get_arg_val<uint32_t>(i++);
        const uint32_t y = get_arg_val<uint32_t>(i++);
        watcher_noc[w] = get_noc_addr(x, y, get_semaphore(progress));
    }

    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);
    const uint32_t step = drains * chunk;
    for (uint32_t t = d * chunk; t < count; t += step) {
        const uint32_t n = count - t < chunk ? count - t : chunk;
        cb_wait_front(cb_out, chunk);
        uint32_t l1 = get_read_ptr(cb_out);
        for (uint32_t k = 0; k < n; ++k) {
            noc_async_write_page(ktt_rs::page_of(first + t + k, rows, cols), out, l1);
            l1 += tile_bytes;
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, chunk);
        if (block > 0 && (t + step >= count || (t + step) / block != t / block)) {
            noc_async_write_barrier();
            for (uint32_t w = 0; w < watchers; ++w) {
                noc_semaphore_inc(watcher_noc[w], 1);
            }
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
