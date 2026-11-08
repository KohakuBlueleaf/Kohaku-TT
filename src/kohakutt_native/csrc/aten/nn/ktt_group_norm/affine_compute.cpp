// Group norm affine compute (fp32 SFPU): per plane and tile, y = x * A + B with A, B
// the plane's scalar tiles, two tiles per acquire; packed in the output format.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {

    // One face: dst[x] = dst[x] * dst[a] + dst[b] (fp32).
    inline void ktt_affine(uint32_t x, uint32_t a, uint32_t b) {
        constexpr uint32_t kTileRows = 32;
        for (int d = 0; d < 8; ++d) {
            sfpi::dst_reg[x * kTileRows] =
                (sfpi::dst_reg[x * kTileRows] * sfpi::dst_reg[a * kTileRows] +
                 sfpi::dst_reg[b * kTileRows]);
            sfpi::dst_reg++;
        }
    }

}  // namespace ckernel::sfpu
#endif

// Runtime args (per core: start; the rest common): first plane, plane step,
// planes, tiles per plane.
void kernel_main() {
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t step = get_common_arg_val<uint32_t>(0);
    const uint32_t planes = get_common_arg_val<uint32_t>(1);
    const uint32_t tiles = get_common_arg_val<uint32_t>(2);
    constexpr uint32_t cb_x = 0, cb_a = 1, cb_b = 2, cb_out = 16;

    compute_kernel_hw_startup(cb_x, cb_out);
    for (uint32_t p = start; p < planes; p += step) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);
        const uint32_t count = tiles % 2 == 0 ? 2 : 1;
        for (uint32_t t = 0; t < tiles; t += count) {
            cb_wait_front(cb_x, count);
            cb_reserve_back(cb_out, count);
            tile_regs_acquire();
            reconfig_data_format_srca(cb_a);
            copy_init(cb_a);
            copy_tile(cb_a, 0, 0);
            copy_tile(cb_b, 0, 1);
            reconfig_data_format_srca(cb_x);
            copy_init(cb_x);
            for (uint32_t i = 0; i < count; ++i) {
                copy_tile(cb_x, i, 2 + i);
            }
            mul_binary_tile_init();
            for (uint32_t i = 0; i < count; ++i) {
                MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE,
                    DST_ACCUM_MODE,
                    ktt_affine,
                    2 + i,
                    0,
                    1,
                    VectorMode::RC
                )));
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < count; ++i) {
                pack_tile(2 + i, cb_out);
            }
            tile_regs_release();
            cb_push_back(cb_out, count);
            cb_pop_front(cb_x, count);
        }
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
    }
}
