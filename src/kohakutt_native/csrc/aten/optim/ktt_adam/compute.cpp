// Adam / AdamW step on the SFPU in fp32 DST, one tile per DST acquire: from p, g, m, v
// (CB 0 - 3) to p', m', v' (CB 16 - 18):
//   g' = sign g + l2 p;  m' = b1 m + (1 - b1) g';  v' = b2 v + (1 - b2) g'^2
//   p' = decay p - step m' / (sqrt(v') scale + eps)
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/tile_move_copy.h"

// Common runtime args (fp32 bits unless noted): block size (tiles, integer), b1,
// 1 - b1, b2, 1 - b2, scale, eps, step, decay, l2, sign (integer: 1 negates g); per
// core: tile count.
void kernel_main() {
    const uint32_t block = get_common_arg_val<uint32_t>(0);
    const uint32_t b1 = get_common_arg_val<uint32_t>(1);
    const uint32_t c1 = get_common_arg_val<uint32_t>(2);
    const uint32_t b2 = get_common_arg_val<uint32_t>(3);
    const uint32_t c2 = get_common_arg_val<uint32_t>(4);
    const uint32_t scale = get_common_arg_val<uint32_t>(5);
    const uint32_t eps = get_common_arg_val<uint32_t>(6);
    const uint32_t step = get_common_arg_val<uint32_t>(7);
    const uint32_t decay = get_common_arg_val<uint32_t>(8);
    const uint32_t l2 = get_common_arg_val<uint32_t>(9);
    const uint32_t negate = get_common_arg_val<uint32_t>(10);
    const uint32_t count = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_p = 0, cb_g = 1, cb_m = 2, cb_v = 3;
    constexpr uint32_t out_p = 16, out_m = 17, out_v = 18;
    constexpr uint32_t kMinusOne = 0xbf800000;

    compute_kernel_hw_startup(cb_p, cb_g, out_p);
    for (uint32_t t = 0; t < count; t += block) {
        const uint32_t n = count - t < block ? count - t : block;
        for (uint32_t cb = cb_p; cb <= cb_v; ++cb) {
            cb_wait_front(cb, n);
        }
        for (uint32_t cb = out_p; cb <= out_v; ++cb) {
            cb_reserve_back(cb, n);
        }
        for (uint32_t j = 0; j < n; ++j) {
            tile_regs_acquire();
            // DST 1: g'.
            copy_init(cb_g);
            copy_tile(cb_g, j, 1);
            binop_with_scalar_tile_init();
            if (negate) {
                mul_unary_tile(1, kMinusOne);
            }
            if (l2 != 0) {
                copy_init(cb_p);
                copy_tile(cb_p, j, 0);
                binop_with_scalar_tile_init();
                mul_unary_tile(0, l2);
                add_binary_tile_init();
                add_binary_tile(1, 0, 1);
            }
            // DST 2: m' = b1 m + (1 - b1) g'.
            copy_init(cb_m);
            copy_tile(cb_m, j, 2);
            copy_init(cb_v);
            copy_tile(cb_v, j, 3);
            fill_tile_init();
            fill_tile(0, 0.0f);
            add_binary_tile_init();
            add_binary_tile(1, 0, 0);
            binop_with_scalar_tile_init();
            mul_unary_tile(2, b1);
            mul_unary_tile(0, c1);
            mul_unary_tile(3, b2);
            add_binary_tile_init();
            add_binary_tile(2, 0, 2);
            // DST 3: v' = b2 v + (1 - b2) g'^2.
            mul_binary_tile_init();
            mul_binary_tile(1, 1, 1);
            binop_with_scalar_tile_init();
            mul_unary_tile(1, c2);
            add_binary_tile_init();
            add_binary_tile(3, 1, 3);
            // DST 1: step m' / (sqrt(v') scale + eps).
            fill_tile_init();
            fill_tile(1, 0.0f);
            add_binary_tile_init();
            add_binary_tile(3, 1, 1);
            sqrt_tile_init();
            sqrt_tile(1);
            binop_with_scalar_tile_init();
            mul_unary_tile(1, scale);
            add_unary_tile(1, eps);
            div_binary_tile_init();
            div_binary_tile(2, 1, 1);
            binop_with_scalar_tile_init();
            mul_unary_tile(1, step);
            // DST 0: p' = decay p - DST 1.
            copy_init(cb_p);
            copy_tile(cb_p, j, 0);
            binop_with_scalar_tile_init();
            mul_unary_tile(0, decay);
            sub_binary_tile_init();
            sub_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out_p);
            pack_tile(2, out_m);
            pack_tile(3, out_v);
            tile_regs_release();
        }
        for (uint32_t cb = cb_p; cb <= cb_v; ++cb) {
            cb_pop_front(cb, n);
        }
        for (uint32_t cb = out_p; cb <= out_v; ++cb) {
            cb_push_back(cb, n);
        }
    }
}
