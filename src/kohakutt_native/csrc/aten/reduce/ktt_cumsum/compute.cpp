// Cumsum compute (fp32 SFPU): per row tile, its transposed W tiles in order through
// cumsum_tile (column-wise, totals carried across the tiles), packed to CB 16.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cumsum.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

// Runtime args: first row tile, row tile step, row tiles, W tiles.
void kernel_main() {
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t step = get_arg_val<uint32_t>(1);
    const uint32_t rows = get_arg_val<uint32_t>(2);
    const uint32_t wt = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_x = 0, cb_out = 16;

    compute_kernel_hw_startup(cb_x, cb_out);
    copy_init(cb_x);
    for (uint32_t u = start; u < rows; u += step) {
        cumsum_tile_init();
        for (uint32_t t = 0; t < wt; ++t) {
            cb_wait_front(cb_x, 1);
            cb_reserve_back(cb_out, 1);
            tile_regs_acquire();
            copy_tile(cb_x, 0, 0);
            cumsum_tile(0, t == 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
            cb_pop_front(cb_x, 1);
        }
    }
}
