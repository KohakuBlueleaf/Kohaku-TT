// Drain reader of the 1x2 collectives: for each chunk this drain owns (chunks d,
// d + drains, ... of tiles [first, first + count), `chunk` tiles each), reads the kept
// tiles (after `kept_offset`) into CB 1 and, with an addend, its tiles into CB 2, waits
// until the peer has written the chunk into scratch (the drain semaphore counts tiles),
// then reads the scratch tiles into CB 0. Resets the semaphore at the end.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/tensor_accessor_args.h"
#include "order.hpp"

// Compile-time args: scratch accessor, source accessor, addend accessor, then the
// runtime-arg count. Runtime args: scratch address, source address, kept tile offset,
// first tile, tile count, tile bytes, chunk tiles, drains, this drain's index, drain
// semaphore address, addend address, 1 with an addend, tiles per scratch page, row and
// column tiles of the column-major order of the source and addend (0, 0: tile order).
void kernel_main() {
    constexpr auto scratch_args = TensorAccessorArgs<0>();
    constexpr auto
        src_args = TensorAccessorArgs<scratch_args.next_compile_time_args_offset()>();
    constexpr auto
        add_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr uint32_t kArgs = get_compile_time_arg_val(
        add_args.next_compile_time_args_offset()
    );
    constexpr uint32_t cb_in = 0, cb_kept = 1, cb_add = 2;

    uint32_t i = 0;
    const uint32_t scratch_addr = get_arg_val<uint32_t>(i++);
    const uint32_t src_addr = get_arg_val<uint32_t>(i++);
    const uint32_t kept_offset = get_arg_val<uint32_t>(i++);
    const uint32_t first = get_arg_val<uint32_t>(i++);
    const uint32_t count = get_arg_val<uint32_t>(i++);
    const uint32_t tile_bytes = get_arg_val<uint32_t>(i++);
    const uint32_t chunk = get_arg_val<uint32_t>(i++);
    const uint32_t drains = get_arg_val<uint32_t>(i++);
    const uint32_t d = get_arg_val<uint32_t>(i++);
    const uint32_t drain_sem = get_arg_val<uint32_t>(i++);
    const uint32_t add_addr = get_arg_val<uint32_t>(i++);
    const uint32_t has_add = get_arg_val<uint32_t>(i++);
    const uint32_t pack = get_arg_val<uint32_t>(i++);
    const uint32_t rows = get_arg_val<uint32_t>(i++);
    const uint32_t cols = get_arg_val<uint32_t>(i++);
    if (i != kArgs || pack == 0 || chunk % pack != 0 || first % pack != 0) {
        return;  // runtime-arg layout does not match this source
    }

    const auto scratch = TensorAccessor(scratch_args, scratch_addr, pack * tile_bytes);
    const auto src = TensorAccessor(src_args, src_addr, tile_bytes);
    const auto add = TensorAccessor(add_args, add_addr, tile_bytes);
    auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(drain_sem);
    uint32_t received = 0;
    for (uint32_t t = d * chunk; t < count; t += drains * chunk) {
        const uint32_t n = count - t < chunk ? count - t : chunk;
        cb_reserve_back(cb_kept, chunk);
        uint32_t l1 = get_write_ptr(cb_kept);
        for (uint32_t k = 0; k < n; ++k) {
            noc_async_read_page(
                kept_offset + ktt_rs::page_of(first + t + k, rows, cols), src, l1
            );
            l1 += tile_bytes;
        }
        if (has_add) {
            cb_reserve_back(cb_add, chunk);
            l1 = get_write_ptr(cb_add);
            for (uint32_t k = 0; k < n; ++k) {
                noc_async_read_page(
                    ktt_rs::page_of(first + t + k, rows, cols), add, l1
                );
                l1 += tile_bytes;
            }
        }
        received += n;
        noc_semaphore_wait_min(sem, received);
        cb_reserve_back(cb_in, chunk);
        l1 = get_write_ptr(cb_in);
        for (uint32_t k = 0; k < n; k += pack) {
            noc_async_read_page((first + t + k) / pack, scratch, l1);
            l1 += pack * tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_kept, chunk);
        if (has_add) {
            cb_push_back(cb_add, chunk);
        }
        cb_push_back(cb_in, chunk);
    }
    noc_semaphore_set(sem, 0);
}
