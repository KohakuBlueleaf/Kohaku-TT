// Row-form group norm backward, sums compute (fp32 SFPU): per piece, DST 0 / 1
// accumulate dy and dy x elementwise over its tiles in one acquire; each is then summed
// over its rows, and the two tiles (row 0: the per-channel sums) are packed to CB 16. A
// piece ending in a row tile shared with the next image then does the same for that one
// tile.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "rows.hpp"

#ifdef TRISC_MATH
namespace ckernel::sfpu {

    // One face: dst[0] += dst[3], dst[1] += dst[3] dst[2] (fp32; x in DST 2, dy in DST
    // 3).
    inline void ktt_rows_bwd_moments(uint32_t, uint32_t, uint32_t) {
        for (int d = 0; d < 8; ++d) {
            const sfpi::vFloat dy = sfpi::dst_reg[96];
            sfpi::dst_reg[0] = sfpi::dst_reg[0] + dy;
            sfpi::dst_reg[32] = sfpi::dst_reg[32] + dy * sfpi::dst_reg[64];
            sfpi::dst_reg++;
        }
    }

}  // namespace ckernel::sfpu
#endif

// Common runtime args: channel tiles Ct, rows per image, images, block. Per core: q0,
// q1.
void kernel_main() {
    const ktt_group_norm::Geometry g{
        get_common_arg_val<uint32_t>(0),
        get_common_arg_val<uint32_t>(1),
        get_common_arg_val<uint32_t>(2)
    };
    const uint32_t block = get_common_arg_val<uint32_t>(3);
    const uint32_t q1 = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_x = 0, cb_g = 1, cb_part = 16;

    compute_kernel_hw_startup(cb_x, cb_part);
    for (uint32_t q = get_arg_val<uint32_t>(0); q < q1;) {
        const ktt_group_norm::Piece p(g, q, q1);
        const uint32_t passes = g.shared(p.n) && p.last ? 2 : 1;
        for (uint32_t pass = 0; pass < passes; ++pass) {
            const uint32_t len = pass == 0 ? p.len : 1;
            tile_regs_acquire();
            fill_tile_init();
            fill_tile(0, 0.0f);
            fill_tile(1, 0.0f);
            for (uint32_t t = 0; t < len; t += block) {
                const uint32_t n = len - t < block ? len - t : block;
                cb_wait_front(cb_x, block);
                cb_wait_front(cb_g, block);
                for (uint32_t k = 0; k < n; ++k) {
                    reconfig_data_format_srca(cb_x);
                    copy_init(cb_x);
                    copy_tile(cb_x, k, 2);
                    copy_tile(cb_g, k, 3);
                    mul_binary_tile_init();
                    MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                        DST_SYNC_MODE,
                        DST_ACCUM_MODE,
                        ktt_rows_bwd_moments,
                        2,
                        0,
                        1,
                        VectorMode::RC
                    )));
                }
                cb_pop_front(cb_x, block);
                cb_pop_front(cb_g, block);
            }
            sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
            sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(0);
            sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(1);
            tile_regs_commit();
            cb_reserve_back(cb_part, 2);
            tile_regs_wait();
            pack_tile(0, cb_part);
            pack_tile(1, cb_part);
            tile_regs_release();
            cb_push_back(cb_part, 2);
        }
        q += p.len;
    }
}
