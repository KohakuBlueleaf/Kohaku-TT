// Argmax compute (fp32 SFPU): per leading row p and row tile, over its W tiles t
// elementwise B = max(B, x), I = I + (x > B) (32 t + column - I) (the first t at the
// maximum), in DST across the row; B and I packed to CB 16 / 17.
#include <cstdint>

#include "api/compute/binary_max_min.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

// Runtime args: first unit, unit step, P, row tiles, W tiles.
void kernel_main() {
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t step = get_arg_val<uint32_t>(1);
    const uint32_t p_count = get_arg_val<uint32_t>(2);
    const uint32_t rt = get_arg_val<uint32_t>(3);
    const uint32_t wt = get_arg_val<uint32_t>(4);
    constexpr uint32_t cb_x = 0, cb_col = 3, cb_b = 16, cb_i = 17;

    compute_kernel_hw_startup(cb_x, cb_b);
    cb_wait_front(cb_col, 1);
    const uint32_t units = (p_count + 31) / 32 * rt;
    for (uint32_t u = start; u < units; u += step) {
        const uint32_t p0 = (u / rt) * 32;
        const uint32_t p1 = p0 + 32 < p_count ? p0 + 32 : p_count;
        for (uint32_t p = p0; p < p1; ++p) {
            tile_regs_acquire();
            for (uint32_t t = 0; t < wt; ++t) {
                cb_wait_front(cb_x, 1);
                reconfig_data_format_srca(cb_x);
                copy_init(cb_x);
                if (t == 0) {
                    copy_tile(cb_x, 0, 0);
                    reconfig_data_format_srca(cb_col);
                    copy_init(cb_col);
                    copy_tile(cb_col, 0, 1);
                } else {
                    copy_tile(cb_x, 0, 2);
                    // DST 3 = (x > B).
                    sub_binary_tile_init();
                    sub_binary_tile(2, 0, 3);
                    gtz_tile_init();
                    gtz_tile(3);
                    binary_max_tile_init();
                    binary_max_tile(0, 2, 0);
                    // DST 2 = 32 t + column; I += mask (DST 2 - I).
                    reconfig_data_format_srca(cb_col);
                    copy_init(cb_col);
                    copy_tile(cb_col, 0, 2);
                    binop_with_scalar_tile_init();
                    add_unary_tile(
                        2, __builtin_bit_cast(uint32_t, static_cast<float>(32 * t))
                    );
                    sub_binary_tile_init();
                    sub_binary_tile(2, 1, 2);
                    mul_binary_tile_init();
                    mul_binary_tile(2, 3, 2);
                    add_binary_tile_init();
                    add_binary_tile(1, 2, 1);
                }
                cb_pop_front(cb_x, 1);
            }
            tile_regs_commit();
            cb_reserve_back(cb_b, 1);
            cb_reserve_back(cb_i, 1);
            tile_regs_wait();
            pack_tile(0, cb_b);
            pack_tile(1, cb_i);
            tile_regs_release();
            cb_push_back(cb_b, 1);
            cb_push_back(cb_i, 1);
        }
    }
}
