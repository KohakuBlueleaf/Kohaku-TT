// Drain compute of the 1x2 fp32 reduce-scatter: per tile of this drain's chunks, the
// received tile (CB 0) plus the kept tile (CB 1) on the SFPU in fp32 Dst, into CB 16.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"

// Runtime args: tile count, chunk tiles, drains, this drain's index (as the reader).
void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0);
    const uint32_t chunk = get_arg_val<uint32_t>(1);
    const uint32_t drains = get_arg_val<uint32_t>(2);
    const uint32_t d = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_in = 0, cb_kept = 1, cb_out = 16;

    compute_kernel_hw_startup(cb_in, cb_kept, cb_out);
    for (uint32_t t = d * chunk; t < count; t += drains * chunk) {
        const uint32_t n = count - t < chunk ? count - t : chunk;
        cb_wait_front(cb_in, chunk);
        cb_wait_front(cb_kept, chunk);
        cb_reserve_back(cb_out, chunk);
        for (uint32_t k = 0; k < n; ++k) {
            tile_regs_acquire();
            copy_init(cb_in);
            copy_tile(cb_in, k, 0);
            copy_init(cb_kept);
            copy_tile(cb_kept, k, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
        }
        cb_pop_front(cb_in, chunk);
        cb_pop_front(cb_kept, chunk);
        cb_push_back(cb_out, chunk);
    }
}
