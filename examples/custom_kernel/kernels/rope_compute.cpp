// RoPE compute: rotates the (even, odd) column pairs of each bf16 tile of x by the
// (cos, sin) pairs of an fp32 table tile, with SFPU instructions on fp32 DST.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "ckernel.h"
#include "ckernel_ops.h"
#include "llk_math_eltwise_binary_sfpu_params.h"

namespace {

    // One face (16 DST rows) of tile `x` rotated by tile `table` into tile `out`. An SFPU
    // load or store moves the even or the odd columns of four DST rows.
    inline void rotate_face(uint32_t x, uint32_t table, uint32_t out) {
        using namespace ckernel;
        constexpr uint32_t x0 = p_sfpu::LREG0, x1 = p_sfpu::LREG1;
        constexpr uint32_t c = p_sfpu::LREG2, s = p_sfpu::LREG3;
        constexpr uint32_t t0 = p_sfpu::LREG4, t1 = p_sfpu::LREG5;
        constexpr uint32_t t2 = p_sfpu::LREG6, t3 = p_sfpu::LREG7;
        constexpr uint32_t zero = p_sfpu::LCONST_0, one = p_sfpu::LCONST_1;
        // A tile is 64 DST rows; +2 selects the odd columns.
        const uint32_t xa = x * 64, ta = table * 64, oa = out * 64;
        for (uint32_t rows = 0; rows < 4; ++rows) {
            TT_SFPLOAD(x0, 0, ADDR_MOD_7, xa);
            TT_SFPLOAD(x1, 0, ADDR_MOD_7, xa + 2);
            TT_SFPLOAD(c, 0, ADDR_MOD_7, ta);
            TT_SFPLOAD(s, 0, ADDR_MOD_7, ta + 2);
            // even = x0 c - x1 s, odd = x0 s + x1 c (SFPMUL's modifier 1 negates).
            TTI_SFPMUL(x0, c, zero, t0, 0);
            TTI_SFPMUL(x1, s, zero, t1, 1);
            TTI_SFPMUL(x0, s, zero, t2, 0);
            TTI_SFPMUL(x1, c, zero, t3, 0);
            TTI_SFPADD(one, t0, t1, t0, 0);
            TTI_SFPADD(one, t2, t3, t2, 0);
            TT_SFPSTORE(t0, 0, ADDR_MOD_7, oa);
            TT_SFPSTORE(t2, 0, ADDR_MOD_7, oa + 2);
            TTI_INCRWC(0, 4, 0, 0);
        }
    }

}  // namespace

// Runtime arg 0: this core's tile count. x arrives in CB 0 and the table in CB 1 in
// blocks of BLOCK tiles (the last block may hold fewer); the output goes to CB 16.
void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_x = 0, cb_table = 1, cb_out = 16;

    compute_kernel_hw_startup(cb_x, cb_table, cb_out);
    mul_binary_tile_init();
    for (uint32_t done = 0; done < count; done += BLOCK) {
        const uint32_t n = count - done < BLOCK ? count - done : BLOCK;
        cb_wait_front(cb_x, BLOCK);
        cb_wait_front(cb_table, BLOCK);
        for (uint32_t k = 0; k < n; ++k) {
            tile_regs_acquire();
            // x in DST[0] (bf16 held exactly in fp32), the table tile in DST[1].
            reconfig_data_format_srca(cb_x);
            copy_init(cb_x);
            copy_tile(cb_x, k, 0);
            reconfig_data_format_srca(cb_table);
            copy_init(cb_table);
            copy_tile(cb_table, k, 1);
            MATH((_llk_math_eltwise_binary_sfpu_params_(rotate_face, 0, 1, 0, VectorMode::RC)));
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }
        cb_pop_front(cb_x, BLOCK);
        cb_pop_front(cb_table, BLOCK);
    }
}
