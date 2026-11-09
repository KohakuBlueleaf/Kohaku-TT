// Norm backward compute (fp32 DST, SFPU only). Per tile row, with S = (mean, rstd) and per
// tile xh = (x - m) r, g = r dy gamma:
//   pass 1: xh and g to fp32 rows in L1; dy xh and dy accumulated over the core's rows by the
//           packer into the dgamma / dbeta partials;
//   sums:   A = sum g, B = sum g xh per row over the segment (DST accumulators, row
//           reduction), added over the group of cores sharing the row;
//   pass 2: dx = g - A / W - xh B / W (RMS norm: without A).
// A per-row statistics tile holds one value in faces 0 / 2 (rows 0-15 / 16-31) and another
// in faces 1 / 3, broadcast across the face's columns. Ragged columns (last tile) and the
// padding rows of a slice's last tile row are masked on the SFPU from the lane ids.
#include <cstdint>
#include <utility>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_dest.h"
#include "args.hpp"

#ifdef TRISC_MATH
namespace ckernel::sfpu {

    // SFPU units: a DST tile 32, a face 8 (face f: rows 16 (f / 2) +, columns 16 (f % 2) +);
    // unit d of a face: rows 4 (d / 2) +, the face's even (d even) or odd columns; lane l:
    // row + l / 8, column 2 (l % 8) + d % 2 (vConstTileId = 2 l).

    // Pass 1, unit O: DST 0 x -> xh, 1 dy -> dy (masked), 2 S -> g, 3 gamma -> dy xh. Padding
    // columns and padding rows are zero in all four.
    template <int O, bool kMask>
    sfpi_inline void p1_vec(sfpi::vFloat m, sfpi::vFloat r, uint32_t cols, uint32_t row16) {
        constexpr int H = O / 16, D = O % 8, C = (O / 8) % 2;
        sfpi::vFloat xh = (sfpi::dst_reg[O] - m) * r;
        sfpi::vFloat dy = sfpi::dst_reg[32 + O];
        sfpi::vFloat g = dy * sfpi::dst_reg[96 + O] * r;
        if constexpr (kMask) {
            v_if((sfpi::vConstTileId & 14) + (16 * C + D % 2) >= static_cast<int>(cols)) {
                xh = 0.0f;
                dy = 0.0f;
                g = 0.0f;
            }
            v_endif;
            v_if((sfpi::vConstTileId & 48) + 16 * (16 * H + 4 * (D / 2)) >= static_cast<int>(row16)) {
                xh = 0.0f;
                dy = 0.0f;
                g = 0.0f;
            }
            v_endif;
            sfpi::dst_reg[32 + O] = dy;
        }
        sfpi::dst_reg[O] = xh;
        sfpi::dst_reg[64 + O] = g;
        sfpi::dst_reg[96 + O] = dy * xh;
    }

    template <bool kMask, int H, int D>
    sfpi_inline void p1_band(uint32_t cols, uint32_t row16) {
        const sfpi::vFloat m = sfpi::dst_reg[64 + 16 * H + D];
        const sfpi::vFloat r = sfpi::dst_reg[64 + 16 * H + 8 + D];
        p1_vec<16 * H + D, kMask>(m, r, cols, row16);
        p1_vec<16 * H + 8 + D, kMask>(m, r, cols, row16);
    }

    template <bool kMask, int... B>
    sfpi_inline void p1_bands(std::integer_sequence<int, B...>, uint32_t cols, uint32_t row16) {
        (p1_band<kMask, B / 8, B % 8>(cols, row16), ...);
    }

    // Pass 1 on a whole tile: cols the valid columns, row16 = 16 (valid rows).
    inline void ktt_bwd_p1(uint32_t, uint32_t, uint32_t, uint32_t cols, uint32_t row16) {
        p1_bands<false>(std::make_integer_sequence<int, 16>{}, cols, row16);
    }

    inline void ktt_bwd_p1m(uint32_t, uint32_t, uint32_t, uint32_t cols, uint32_t row16) {
        p1_bands<true>(std::make_integer_sequence<int, 16>{}, cols, row16);
    }

    // One face: dst[2] += dst[0], dst[3] += dst[0] dst[1].
    inline void ktt_bwd_ab(uint32_t, uint32_t, uint32_t) {
        for (int d = 0; d < 8; ++d) {
            const sfpi::vFloat g = sfpi::dst_reg[d];
            const sfpi::vFloat xh = sfpi::dst_reg[32 + d];
            sfpi::dst_reg[64 + d] = sfpi::dst_reg[64 + d] + g;
            sfpi::dst_reg[96 + d] = sfpi::dst_reg[96 + d] + g * xh;
        }
    }

    // Pass 2 on band (H, D): DST 0 g -> dx, 1 xh, 2 S2 (A / W in faces 0 / 2, B / W in 1 / 3).
    template <int O>
    sfpi_inline void p2_vec(sfpi::vFloat c1, sfpi::vFloat c2) {
        const sfpi::vFloat t = sfpi::dst_reg[32 + O] * c2 + c1;
        sfpi::dst_reg[O] = sfpi::dst_reg[O] - t;
    }

    template <int H, int D>
    sfpi_inline void p2_band() {
        const sfpi::vFloat c1 = sfpi::dst_reg[64 + 16 * H + D];
        const sfpi::vFloat c2 = sfpi::dst_reg[64 + 16 * H + 8 + D];
        p2_vec<16 * H + D>(c1, c2);
        p2_vec<16 * H + 8 + D>(c1, c2);
    }

    template <int... B>
    sfpi_inline void p2_bands(std::integer_sequence<int, B...>) {
        (p2_band<B / 8, B % 8>(), ...);
    }

    inline void ktt_bwd_p2(uint32_t, uint32_t, uint32_t) {
        p2_bands(std::make_integer_sequence<int, 16>{});
    }

    // Faces 1 and 3 of dst[2] copied into dst[1].
    inline void ktt_norm_merge(uint32_t, uint32_t, uint32_t) {
        for (int d = 0; d < 8; ++d) {
            const sfpi::vFloat upper = sfpi::dst_reg[64 + 8 + d];
            const sfpi::vFloat lower = sfpi::dst_reg[64 + 24 + d];
            sfpi::dst_reg[32 + 8 + d] = upper;
            sfpi::dst_reg[32 + 24 + d] = lower;
        }
    }

}  // namespace ckernel::sfpu
#endif

namespace {

    constexpr uint32_t cb_x = 0, cb_g = 1, cb_gather = 4, cb_s = 5, cb_st = 8, cb_dy = 9;
    constexpr uint32_t cb_gg = 10, cb_xh = 11, cb_ab = 12, cb_s2 = 13, cb_dx = 16;
    constexpr uint32_t cb_part = 17, cb_dg = 21, cb_db = 22;

    inline void copy(uint32_t cb, uint32_t tile, uint32_t dst) {
        reconfig_data_format_srca(cb);
        copy_init(cb);
        copy_tile(cb, tile, dst);
    }

    inline void fill(uint32_t dst, float v) {
        fill_tile_init();
        fill_tile(dst, v);
    }

    // Packs to `cb` the tile with faces 0 / 2 of column 0 of `src` tile 0 broadcast (zero
    // when `first_zero`) and faces 1 / 3 of column 0 of `src` tile 1 broadcast.
    inline void stat_tile(uint32_t src, bool first_zero, uint32_t cb) {
        tile_regs_acquire();
        if (first_zero) {
            fill(1, 0.0f);
        }
        reconfig_data_format(src, src);
        unary_bcast_init<BroadcastType::COL>(src);
        if (!first_zero) {
            unary_bcast<BroadcastType::COL>(src, 0, 1);
        }
        unary_bcast<BroadcastType::COL>(src, 1, 2);
        mul_binary_tile_init();
        MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
            DST_SYNC_MODE, DST_ACCUM_MODE, ktt_norm_merge, 1, 2, 1, VectorMode::None
        )));
        tile_regs_commit();
        cb_reserve_back(cb, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb);
        pack_tile(1, cb);
        tile_regs_release();
        cb_push_back(cb, 1);
    }

    inline void reduce(uint32_t dst, bool rows) {
        sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
        if (rows) {
            sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(dst);
        } else {
            sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(dst);
        }
    }

}  // namespace

// Common runtime args: tile rows, groups, group size g, rms, has gamma, 1 / W (fp32 bits),
// tile rows per slice, valid rows of a slice's last tile row, want dx, want dgamma, want
// dbeta, x / dy ring pages per row. Per core: group, index in group, column tiles, write
// rotation, valid columns of the last tile (32: none padded). Compile-time arg 0: the
// common arg count.
void kernel_main() {
    if (!ktt_norm::args_ok<0>(13)) {
        return;
    }
    const uint32_t rows = get_common_arg_val<uint32_t>(0);
    const uint32_t groups = get_common_arg_val<uint32_t>(1);
    const uint32_t gsize = get_common_arg_val<uint32_t>(2);
    const bool rms = get_common_arg_val<uint32_t>(3) != 0;
    const bool has_gamma = get_common_arg_val<uint32_t>(4) != 0;
    const uint32_t inv_w = get_common_arg_val<uint32_t>(5);
    const uint32_t mt = get_common_arg_val<uint32_t>(6);
    const uint32_t hlast = get_common_arg_val<uint32_t>(7);
    const bool want_dx = get_common_arg_val<uint32_t>(8) != 0;
    const bool want_dg = get_common_arg_val<uint32_t>(9) != 0;
    const bool want_db = get_common_arg_val<uint32_t>(10) != 0;
    const uint32_t stride = get_common_arg_val<uint32_t>(11);
    const uint32_t gid = get_arg_val<uint32_t>(0);
    const uint32_t seg = get_arg_val<uint32_t>(2);
    const uint32_t rot = get_arg_val<uint32_t>(3);
    const uint32_t wlast = get_arg_val<uint32_t>(4);
    const uint32_t last = seg - 1;

    compute_kernel_hw_startup(cb_x, cb_dx);
    if (has_gamma) {
        cb_wait_front(cb_g, seg);
    }
    if (want_dg) {
        cb_reserve_back(cb_dg, seg);
    }
    if (want_db) {
        cb_reserve_back(cb_db, seg);
    }

    bool first_row = true;
    for (uint32_t r = gid; r < rows; r += groups) {
        const bool row_pad = hlast < 32 && r % mt == mt - 1;
        cb_wait_front(cb_st, 2);
        cb_wait_front(cb_x, stride);
        cb_wait_front(cb_dy, stride);
        stat_tile(cb_st, rms, cb_s);
        cb_wait_front(cb_s, 1);

        // Pass 1.
        cb_reserve_back(cb_gg, stride);
        cb_reserve_back(cb_xh, stride);
        for (uint32_t j = 0; j < seg; ++j) {
            tile_regs_acquire();
            copy(cb_x, j, 0);
            copy_tile(cb_dy, j, 1);
            copy(cb_s, 0, 2);
            if (has_gamma) {
                reconfig_data_format(cb_g, cb_g);
                unary_bcast_init<BroadcastType::ROW>(cb_g);
                unary_bcast<BroadcastType::ROW>(cb_g, j, 3);
            } else {
                fill(3, 1.0f);
            }
            mul_binary_tile_init();
            const bool col = j == last && wlast < 32;
            if (col || row_pad) {
                MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, ktt_bwd_p1m, 0, 1, 0, VectorMode::None,
                    col ? wlast : 32u, row_pad ? 16 * hlast : 512u
                )));
            } else {
                MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, ktt_bwd_p1, 0, 1, 0, VectorMode::None, 32u, 512u
                )));
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_xh);
            pack_tile<true>(0, cb_xh, j);
            pack_tile<true>(2, cb_gg, j);
            if (want_dg || want_db) {
                pack_reconfig_l1_acc(first_row ? 0 : 1);
                if (want_dg) {
                    pack_tile<true>(3, cb_dg, j);
                }
                if (want_db) {
                    pack_tile<true>(1, cb_db, j);
                }
                pack_reconfig_l1_acc(0);
            }
            tile_regs_release();
        }
        cb_push_back(cb_gg, stride);
        cb_push_back(cb_xh, stride);
        first_row = false;
        cb_wait_front(cb_gg, stride);
        cb_wait_front(cb_xh, stride);

        if (want_dx) {
            // Row sums of the segment: D2 A, D3 B (columns after the row reduction).
            tile_regs_acquire();
            fill(2, 0.0f);
            fill(3, 0.0f);
            reconfig_data_format_srca(cb_gg);
            copy_init(cb_gg);
            for (uint32_t j = 0; j < seg; ++j) {
                copy_tile(cb_gg, j, 0);
                copy_tile(cb_xh, j, 1);
                mul_binary_tile_init();
                MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, ktt_bwd_ab, 0, 1, 0, VectorMode::RC
                )));
                copy_init(cb_gg);
            }
            reduce(2, true);
            reduce(3, true);
            if (gsize > 1) {
                // Partial sums as rows to the exchange; the group's rows come back gathered.
                transpose_dest_init<DST_ACCUM_MODE>();
                transpose_dest<DST_ACCUM_MODE>(2);
                transpose_dest<DST_ACCUM_MODE>(3);
                tile_regs_commit();
                cb_reserve_back(cb_part, 2);
                tile_regs_wait();
                pack_reconfig_data_format(cb_part);
                pack_tile(2, cb_part);
                pack_tile(3, cb_part);
                tile_regs_release();
                cb_push_back(cb_part, 2);
                cb_wait_front(cb_gather, 2);
                tile_regs_acquire();
                copy(cb_gather, 0, 2);
                copy_tile(cb_gather, 1, 3);
                cb_pop_front(cb_gather, 2);
                reduce(2, false);
                reduce(3, false);
                transpose_dest_init<DST_ACCUM_MODE>();
                transpose_dest<DST_ACCUM_MODE>(2);
                transpose_dest<DST_ACCUM_MODE>(3);
            }
            binop_with_scalar_tile_init();
            mul_unary_tile(2, inv_w);
            mul_unary_tile(3, inv_w);
            if (rms) {
                fill(2, 0.0f);
            }
            tile_regs_commit();
            cb_reserve_back(cb_ab, 2);
            tile_regs_wait();
            pack_reconfig_data_format(cb_ab);
            pack_tile(2, cb_ab);
            pack_tile(3, cb_ab);
            tile_regs_release();
            cb_push_back(cb_ab, 2);
            cb_wait_front(cb_ab, 2);
            stat_tile(cb_ab, false, cb_s2);
            cb_pop_front(cb_ab, 2);

            // Pass 2.
            cb_wait_front(cb_s2, 1);
            pack_reconfig_data_format(cb_dx);
            uint32_t j = rot;
            for (uint32_t i = 0; i < seg; ++i) {
                tile_regs_acquire();
                copy(cb_gg, j, 0);
                copy_tile(cb_xh, j, 1);
                copy_tile(cb_s2, 0, 2);
                mul_binary_tile_init();
                MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, ktt_bwd_p2, 0, 1, 0, VectorMode::None
                )));
                tile_regs_commit();
                cb_reserve_back(cb_dx, 1);
                tile_regs_wait();
                pack_tile(0, cb_dx);
                tile_regs_release();
                cb_push_back(cb_dx, 1);
                j = j + 1 == seg ? 0 : j + 1;
            }
            cb_pop_front(cb_s2, 1);
        }
        cb_pop_front(cb_gg, stride);
        cb_pop_front(cb_xh, stride);
        cb_pop_front(cb_s, 1);
        cb_pop_front(cb_st, 2);
        cb_pop_front(cb_x, stride);
        cb_pop_front(cb_dy, stride);
    }
    if (want_dg) {
        cb_push_back(cb_dg, seg);
    }
    if (want_db) {
        cb_push_back(cb_db, seg);
    }
}
