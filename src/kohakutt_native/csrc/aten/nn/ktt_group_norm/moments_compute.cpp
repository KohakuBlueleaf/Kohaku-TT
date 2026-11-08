// Group norm moments compute (fp32 SFPU): per plane, DST 0 / 1 accumulate x and x^2
// elementwise over its tiles in one acquire; each then reduced (columns, then rows) to
// a scalar at [0, 0] and packed to the plane's s1 / s2 tiles.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {

    // One face: dst[s1] += dst[x], dst[s2] += dst[x]^2 (fp32).
    inline void ktt_moments(uint32_t x, uint32_t s1, uint32_t s2) {
        constexpr uint32_t kTileRows = 32;
        for (int d = 0; d < 8; ++d) {
            const sfpi::vFloat v = sfpi::dst_reg[x * kTileRows];
            sfpi::dst_reg[s1 * kTileRows] = sfpi::dst_reg[s1 * kTileRows] + v;
            sfpi::dst_reg[s2 * kTileRows] = sfpi::dst_reg[s2 * kTileRows] + v * v;
            sfpi::dst_reg++;
        }
    }

}  // namespace ckernel::sfpu
#endif

// Runtime args: first plane, plane step, planes, tiles per plane.
void kernel_main() {
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t step = get_arg_val<uint32_t>(1);
    const uint32_t planes = get_arg_val<uint32_t>(2);
    const uint32_t tiles = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_x = 0, cb_s1 = 16, cb_s2 = 17;

    compute_kernel_hw_startup(cb_x, cb_s1);
    for (uint32_t p = start; p < planes; p += step) {
        tile_regs_acquire();
        for (uint32_t t = 0; t < tiles; ++t) {
            cb_wait_front(cb_x, 1);
            copy_init(cb_x);
            if (t == 0) {
                copy_tile(cb_x, 0, 0);
                copy_tile(cb_x, 0, 1);
                mul_binary_tile_init();
                mul_binary_tile(1, 1, 1);
            } else {
                copy_tile(cb_x, 0, 2);
                mul_binary_tile_init();
                MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, ktt_moments, 2, 0, 1, VectorMode::RC
                )));
            }
            cb_pop_front(cb_x, 1);
        }
        sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
        for (uint32_t d = 0; d < 2; ++d) {
            sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(d);
            sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(d);
        }
        tile_regs_commit();
        cb_reserve_back(cb_s1, 1);
        cb_reserve_back(cb_s2, 1);
        tile_regs_wait();
        pack_tile(0, cb_s1);
        pack_tile(1, cb_s2);
        tile_regs_release();
        cb_push_back(cb_s1, 1);
        cb_push_back(cb_s2, 1);
    }
}
