// Argmax compute (fp32 SFPU): per item and leading row, over the chunk's W tiles mask =
// x > B or (x NaN, B not), B = mask ? x : B, I = mask ? 32 t + column : I; ARGMIN: -x.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/shift.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {

    // DST (full sync, 8 fp32 tiles): B, I, then scratch; kAt: 32 t + column.
    constexpr uint32_t kB = 0, kI = 1, kX = 2, kMask = 3, kNanX = 4, kKeyB = 5, kAt = 6;
    constexpr uint32_t cb_x = 0, cb_col = 3;

    // DST `dst` += `value` (fp32).
    inline void add_scalar(uint32_t dst, float value) {
        binop_with_scalar_tile_init();
        add_unary_tile(dst, __builtin_bit_cast(uint32_t, value));
    }

    // DST `dst` = column index + `offset` (fp32, exact below 2^24).
    inline void column_index(uint32_t dst, uint32_t offset) {
        reconfig_data_format_srca(cb_col);
        copy_init(cb_col);
        copy_tile(cb_col, 0, dst);
        if (offset != 0) {
            add_scalar(dst, static_cast<float>(offset));
        }
    }

    // DST `dst` = the front x tile (negated for argmin).
    inline void load_x(uint32_t dst) {
        reconfig_data_format_srca(cb_x);
        copy_init(cb_x);
        copy_tile(cb_x, 0, dst);
#ifdef X_BF16_BITS
        // bf16 read as its 16 raw bits: the fp32 value (NaN kept) is them << 16.
        left_shift_tile_init();
        left_shift_tile<DataFormat::UInt32>(dst, 16);
#endif
#ifdef ARGMIN
        negative_tile_init();
        negative_tile(dst);
#endif
    }

}  // namespace

// Common runtime args: item step, P, row tiles, W tiles, W chunks per unit; per core:
// first item.
void kernel_main() {
    const uint32_t step = get_common_arg_val<uint32_t>(0);
    const uint32_t p_count = get_common_arg_val<uint32_t>(1);
    const uint32_t rt = get_common_arg_val<uint32_t>(2);
    const uint32_t wt = get_common_arg_val<uint32_t>(3);
    const uint32_t split = get_common_arg_val<uint32_t>(4);
    const uint32_t start = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_b = 16, cb_i = 17;

    compute_kernel_hw_startup(cb_x, cb_b);
    cb_wait_front(cb_col, 1);
    const uint32_t items = (p_count + 31) / 32 * rt * split;
    for (uint32_t item = start; item < items; item += step) {
        const uint32_t u = item / split, j = item % split;
        const uint32_t t0 = j * wt / split, t1 = (j + 1) * wt / split;
        const uint32_t p0 = (u / rt) * 32;
        const uint32_t p1 = p0 + 32 < p_count ? p0 + 32 : p_count;
        for (uint32_t p = p0; p < p1; ++p) {
            tile_regs_acquire();
            for (uint32_t t = t0; t < t1; ++t) {
                cb_wait_front(cb_x, 1);
                if (t == t0) {
                    load_x(kB);
                    cb_pop_front(cb_x, 1);
                    column_index(kI, 32 * t);
                    column_index(kAt, 32 * t);
                    continue;
                }
                load_x(kX);
                cb_pop_front(cb_x, 1);
                // mask = (x > B) + (x != x) (B == B).
                gt_binary_tile_init();
                gt_binary_tile(kX, kB, kMask);
                ne_binary_tile_init();
                ne_binary_tile(kX, kX, kNanX);
                eq_binary_tile_init();
                eq_binary_tile(kB, kB, kKeyB);
                mul_binary_tile_init();
                mul_binary_tile(kNanX, kKeyB, kNanX);
                add_binary_tile_init();
                add_binary_tile(kMask, kNanX, kMask);
                // B = mask ? x : B; I = mask ? 32 t + column : I.
                add_scalar(kAt, 32.0f);
                where_tile_init();
                where_tile<DataFormat::Float32>(kMask, kX, kB, kB);
                where_tile<DataFormat::Float32>(kMask, kAt, kI, kI);
            }
            tile_regs_commit();
            cb_reserve_back(cb_b, 1);
            cb_reserve_back(cb_i, 1);
            tile_regs_wait();
            pack_tile(kB, cb_b);
            pack_tile(kI, cb_i);
            tile_regs_release();
            cb_push_back(cb_b, 1);
            cb_push_back(cb_i, 1);
        }
    }
}
