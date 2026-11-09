// Norm forward compute (fp32 DST, SFPU only). Per tile row: statistics of the core's
// segment on transposed x tiles with the two-pass helpers (each lane one row; a ragged last
// tile is a row count), combined over the group of cores sharing the row; then
// y = (x - m) r g + b for two column tiles per DST acquire from the row's statistics tile S
// and the pair's parameter tile P (built once).
#include <cstdint>
#include <utility>

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/transpose_dest.h"
#include "api/compute/welford.h"
#include "args.hpp"
#include "api/debug/kernel_profiler.hpp"

#ifdef TRISC_MATH
namespace ckernel::sfpu {

    // SFPU units: a DST tile 32, a face 8 (face f: rows 16 (f / 2) +, columns 16 (f % 2) +);
    // unit d of a face: rows 4 (d / 2) +, the face's even (d even) or odd columns; lane l:
    // row + l / 8, column 2 (l % 8) + d % 2 (vConstTileId = 2 l). S (DST 2): m in faces 0 /
    // 2, r in faces 1 / 3. P (DST 3): face c, unit 4 t + e: gamma of pair tile t at the
    // columns of parity e; unit 4 t + 2 + e: beta.

    template <int T, int H, int C, int D, bool kMask>
    sfpi_inline void y_vec(sfpi::vFloat m, sfpi::vFloat r, uint32_t cols) {
        constexpr int O = 32 * T + 16 * H + 8 * C + D;
        constexpr int G = 96 + 8 * C + 4 * T + D % 2;
        sfpi::vFloat v = (sfpi::dst_reg[O] - m) * r;
        v = v * sfpi::dst_reg[G] + sfpi::dst_reg[G + 2];
        if constexpr (kMask) {
            v_if((sfpi::vConstTileId & 14) + (16 * C + D % 2) >= static_cast<int>(cols)) {
                v = 0.0f;
            }
            v_endif;
        }
        sfpi::dst_reg[O] = v;
    }

    // Band (H, D) of pair tiles 0 .. N - 1; the last one zeroed at columns >= cols when kMask.
    template <int N, bool kMask, int H, int D>
    sfpi_inline void y_band(uint32_t cols) {
        const sfpi::vFloat m = sfpi::dst_reg[64 + 16 * H + D];
        const sfpi::vFloat r = sfpi::dst_reg[64 + 16 * H + 8 + D];
        y_vec<0, H, 0, D, kMask && N == 1>(m, r, cols);
        y_vec<0, H, 1, D, kMask && N == 1>(m, r, cols);
        if constexpr (N == 2) {
            y_vec<1, H, 0, D, kMask>(m, r, cols);
            y_vec<1, H, 1, D, kMask>(m, r, cols);
        }
    }

    template <int N, bool kMask, int... B>
    sfpi_inline void y_bands(std::integer_sequence<int, B...>, uint32_t cols) {
        (y_band<N, kMask, B / 8, B % 8>(cols), ...);
    }

    template <int N, bool kMask>
    sfpi_inline void ktt_y(uint32_t cols) {
        y_bands<N, kMask>(std::make_integer_sequence<int, 16>{}, cols);
    }

    inline void ktt_norm_y2(uint32_t, uint32_t, uint32_t, uint32_t cols) { ktt_y<2, false>(cols); }
    inline void ktt_norm_y2m(uint32_t, uint32_t, uint32_t, uint32_t cols) { ktt_y<2, true>(cols); }
    inline void ktt_norm_y1(uint32_t, uint32_t, uint32_t, uint32_t cols) { ktt_y<1, false>(cols); }
    inline void ktt_norm_y1m(uint32_t, uint32_t, uint32_t, uint32_t cols) { ktt_y<1, true>(cols); }

    // P (DST 0) rows of pair tile T from the gamma (DST 1) and beta (DST 2) row broadcasts.
    template <int T>
    sfpi_inline void p_put() {
        for (int c = 0; c < 2; ++c) {
            for (int e = 0; e < 2; ++e) {
                const sfpi::vFloat g = sfpi::dst_reg[32 + 8 * c + e];
                const sfpi::vFloat b = sfpi::dst_reg[64 + 8 * c + e];
                sfpi::dst_reg[8 * c + 4 * T + e] = g;
                sfpi::dst_reg[8 * c + 4 * T + 2 + e] = b;
            }
        }
    }

    inline void ktt_norm_p0(uint32_t, uint32_t, uint32_t) { p_put<0>(); }
    inline void ktt_norm_p1(uint32_t, uint32_t, uint32_t) { p_put<1>(); }

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

    constexpr uint32_t cb_x = 0, cb_g = 1, cb_b = 2, cb_p = 3, cb_gather = 4, cb_s = 5;
    constexpr uint32_t cb_st = 6, cb_count = 7, cb_y = 16, cb_part = 17, cb_n = 18;
    constexpr uint32_t cb_so = 19, cb_row = 24;
    constexpr uint32_t kOne = 0x3f800000u;

    inline void copy(uint32_t cb, uint32_t tile, uint32_t dst) {
        reconfig_data_format_srca(cb);
        copy_init(cb);
        copy_tile(cb, tile, dst);
    }

    inline void row_bcast(uint32_t cb, uint32_t tile, uint32_t dst) {
        reconfig_data_format(cb, cb);
        unary_bcast_init<BroadcastType::ROW>(cb);
        unary_bcast<BroadcastType::ROW>(cb, tile, dst);
    }

    inline void fill(uint32_t dst, float v) {
        fill_tile_init();
        fill_tile(dst, v);
    }

    inline void reduce_cols(uint32_t dst) {
        sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(dst);
    }

}  // namespace

// Common runtime args: tile rows, groups, group size g, rms, has gamma, has beta, eps (fp32
// bits), stats wanted, x ring pages per row, 1 / W (fp32 bits). Per core: group, index in
// group, column tiles, valid columns of the last tile (32: none padded), 1 / n and n (fp32
// bits; n the core's valid columns). Compile-time arg 0: the common arg count. Rows: the
// indices in CB 24 (from the reader) until one >= tile rows.
void kernel_main() {
    if (!ktt_norm::args_ok<0>(11)) {
        return;
    }
    const uint32_t rows = get_common_arg_val<uint32_t>(0);
    const uint32_t gsize = get_common_arg_val<uint32_t>(2);
    const bool rms = get_common_arg_val<uint32_t>(3) != 0;
    const bool has_gamma = get_common_arg_val<uint32_t>(4) != 0;
    const bool has_beta = get_common_arg_val<uint32_t>(5) != 0;
    const uint32_t eps = get_common_arg_val<uint32_t>(6);
    const bool stats = get_common_arg_val<uint32_t>(7) != 0;
    const uint32_t stride = get_common_arg_val<uint32_t>(8);
    const uint32_t inv_w = get_common_arg_val<uint32_t>(9);
    const uint32_t seg = get_arg_val<uint32_t>(2);
    const uint32_t last_rows = get_arg_val<uint32_t>(3);
    const uint32_t inv_n = get_arg_val<uint32_t>(4);
    const uint32_t n_bits = get_arg_val<uint32_t>(5);
    const uint32_t last = seg - 1;
    const uint32_t pairs = (seg + 1) / 2;

    compute_kernel_hw_startup(cb_x, cb_y);

    // P tiles, once; the count row n for the group exchange.
    if (has_gamma) {
        cb_wait_front(cb_g, seg);
    }
    if (has_beta) {
        cb_wait_front(cb_b, seg);
    }
    cb_reserve_back(cb_p, pairs);
    for (uint32_t k = 0; k < pairs; ++k) {
        tile_regs_acquire();
        fill(0, 0.0f);
        for (uint32_t t = 0; t < 2 && 2 * k + t < seg; ++t) {
            const uint32_t j = 2 * k + t;
            if (has_gamma) {
                row_bcast(cb_g, j, 1);
            } else {
                fill(1, 1.0f);
            }
            if (has_beta) {
                row_bcast(cb_b, j, 2);
            } else {
                fill(2, 0.0f);
            }
            mul_binary_tile_init();
            if (t == 0) {
                MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, ktt_norm_p0, 0, 1, 0, VectorMode::None
                )));
            } else {
                MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, ktt_norm_p1, 0, 1, 0, VectorMode::None
                )));
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_p);
        pack_tile<true>(0, cb_p, k);
        tile_regs_release();
    }
    cb_push_back(cb_p, pairs);
    if (has_gamma) {
        cb_pop_front(cb_g, seg);
    }
    if (has_beta) {
        cb_pop_front(cb_b, seg);
    }
    if (gsize > 1) {
        tile_regs_acquire();
        fill(0, __builtin_bit_cast(float, n_bits));
        tile_regs_commit();
        cb_reserve_back(cb_n, 1);
        tile_regs_wait();
        pack_tile(0, cb_n);
        tile_regs_release();
        cb_push_back(cb_n, 1);
        cb_wait_front(cb_count, 1);
    }
    cb_wait_front(cb_p, pairs);

    while (true) {
        // The row index from the reader (the same on the three threads); >= rows: done.
        cb_wait_front(cb_row, 1);
        const uint32_t r = read_tile_value(cb_row, 0, 0);
        cb_pop_front(cb_row, 1);
        if (r >= rows) {
            break;
        }
        {
            DeviceZoneScopedN("x-wait");
            cb_wait_front(cb_x, stride);
        }
        {
            DeviceZoneScopedN("stats");

        // Segment statistics: D0 the transposed x tile, D1 mean row, D2 M2 (or var) row.
        tile_regs_acquire();
        fill(1, 0.0f);
        fill(2, 0.0f);
        reconfig_data_format_srca(cb_x);
        transpose_init(cb_x);
        two_pass_stats_init_shifted();
        if (rms) {
            two_pass_stats_clear();
        } else {
            for (uint32_t j = 0; j < seg; ++j) {
                transpose_tile(cb_x, j, 0);
                const uint32_t n = j == last ? last_rows : 32;
                if (j == 0) {
                    two_pass_stats_update_shifted_rows<TwoPassAccumulation::ShiftedSum, TwoPassAnchor::Initialise>(
                        0, 0, n
                    );
                } else {
                    two_pass_stats_update_shifted_rows<TwoPassAccumulation::ShiftedSum>(0, 0, n);
                }
            }
            two_pass_stats_finish_shifted_mean<true, false>(inv_n);
        }
        for (uint32_t j = 0; j < seg; ++j) {
            transpose_tile(cb_x, j, 0);
            two_pass_stats_update_rows(0, 0, j == last ? last_rows : 32);
        }
        two_pass_stats_finalize_to_row(1, gsize > 1 ? kOne : inv_n);
        if (gsize > 1) {
            // Partials (mean, M2) as rows to the exchange; the group's rows come back
            // gathered (row c: core c), with the counts n_c in the count tile.
            tile_regs_commit();
            cb_reserve_back(cb_part, 2);
            tile_regs_wait();
            pack_reconfig_data_format(cb_part);
            pack_tile(1, cb_part);
            pack_tile(2, cb_part);
            tile_regs_release();
            cb_push_back(cb_part, 2);

            // Merge about core 0's mean m0: d_c = m_c - m0, D = sum n_c d_c / W,
            // mean = m0 + D, M2 = sum M2_c + n_c (d_c - D)^2.
            cb_wait_front(cb_gather, 2);
            tile_regs_acquire();
            copy(cb_gather, 0, 0);
            copy_tile(cb_gather, 0, 3);
            copy_tile(cb_gather, 1, 2);
            copy_tile(cb_count, 0, 1);
            sfpu_bcast_row_init();
            sfpu_sub_bcast_row(0, 3);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 3);
            reduce_cols(3);
            binop_with_scalar_tile_init();
            mul_unary_tile(3, inv_w);
            sfpu_bcast_row_init();
            sfpu_sub_bcast_row(0, 3);
            mul_binary_tile_init();
            mul_binary_tile(0, 0, 0);
            mul_binary_tile(0, 1, 0);
            add_binary_tile_init();
            add_binary_tile(0, 2, 0);
            reduce_cols(0);
            binop_with_scalar_tile_init();
            mul_unary_tile(0, inv_w);
            copy(cb_gather, 0, 1);
            add_binary_tile_init();
            add_binary_tile(3, 1, 3);
            cb_pop_front(cb_gather, 2);
        }
        // Mean row in D1 (group: D3), variance row in D2 (group: D0); to columns, then rstd.
        const uint32_t dm = gsize > 1 ? 3 : 1, dv = gsize > 1 ? 0 : 2;
        transpose_dest_init<DST_ACCUM_MODE>();
        transpose_dest<DST_ACCUM_MODE>(dm);
        transpose_dest<DST_ACCUM_MODE>(dv);
        relu_tile_init();
        relu_tile(dv);
        binop_with_scalar_tile_init();
        add_unary_tile(dv, eps);
        rsqrt_tile_init();
        rsqrt_tile(dv);
        tile_regs_commit();
        cb_reserve_back(cb_st, 2);
        if (stats) {
            cb_reserve_back(cb_so, 2);
        }
        tile_regs_wait();
        pack_reconfig_data_format(cb_st);
        pack_tile(dm, cb_st);
        pack_tile(dv, cb_st);
        if (stats) {
            pack_tile(dm, cb_so);
            pack_tile(dv, cb_so);
        }
        tile_regs_release();
        cb_push_back(cb_st, 2);
        if (stats) {
            cb_push_back(cb_so, 2);
        }

        }
        // S: mean / rstd columns broadcast by the unpacker, rstd's faces 1 / 3 merged in.
        cb_wait_front(cb_st, 2);
        tile_regs_acquire();
        reconfig_data_format(cb_st, cb_st);
        unary_bcast_init<BroadcastType::COL>(cb_st);
        unary_bcast<BroadcastType::COL>(cb_st, 0, 1);
        unary_bcast<BroadcastType::COL>(cb_st, 1, 2);
        mul_binary_tile_init();
        MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
            DST_SYNC_MODE, DST_ACCUM_MODE, ktt_norm_merge, 1, 2, 1, VectorMode::None
        )));
        tile_regs_commit();
        cb_pop_front(cb_st, 2);
        cb_reserve_back(cb_s, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_s);
        pack_tile(1, cb_s);
        tile_regs_release();
        cb_push_back(cb_s, 1);

        {
        DeviceZoneScopedN("y");
        // y, two column tiles per acquire, pairs in write order from pair r mod pairs.
        cb_wait_front(cb_s, 1);
        pack_reconfig_data_format(cb_y);
        uint32_t k = r % pairs;
        for (uint32_t i = 0; i < pairs; ++i) {
            const uint32_t j = 2 * k;
            const bool two = j + 1 < seg;
            const bool mask = last_rows < 32 && j + (two ? 1 : 0) == last;
            tile_regs_acquire();
            copy(cb_s, 0, 2);
            copy_tile(cb_p, k, 3);
            copy(cb_x, j, 0);
            if (two) {
                copy_tile(cb_x, j + 1, 1);
            }
            mul_binary_tile_init();
            if (two && mask) {
                MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, ktt_norm_y2m, 0, 1, 0, VectorMode::None, last_rows
                )));
            } else if (two) {
                MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, ktt_norm_y2, 0, 1, 0, VectorMode::None, last_rows
                )));
            } else if (mask) {
                MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, ktt_norm_y1m, 0, 1, 0, VectorMode::None, last_rows
                )));
            } else {
                MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, ktt_norm_y1, 0, 1, 0, VectorMode::None, last_rows
                )));
            }
            tile_regs_commit();
            // One page per reservation: a pair never straddles the end of the ring.
            tile_regs_wait();
            for (uint32_t t = 0; t < (two ? 2u : 1u); ++t) {
                cb_reserve_back(cb_y, 1);
                pack_tile(t, cb_y);
                cb_push_back(cb_y, 1);
            }
            tile_regs_release();
            k = k + 1 == pairs ? 0 : k + 1;
        }
        }
        cb_pop_front(cb_s, 1);
        cb_pop_front(cb_x, stride);
    }
}
