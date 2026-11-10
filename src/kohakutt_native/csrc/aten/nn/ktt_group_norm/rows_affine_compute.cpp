// Row-form group norm affine compute (fp32 SFPU). Per piece: the segment's per-channel
// sums (the partial pages summed over their rows), each replaced by its group's sum
// through the mask tile, then mean, rstd, a = rstd w and b = bias - mean a as rows;
// then y = x a + b for the piece's tiles, a / b broadcast down the rows, two tiles per
// acquire. A row tile shared with the next image takes a / b tiles mixed by rows from
// both images' statistics.
#include <cstdint>

#include "api/compute/bcast.h"
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
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_dest.h"
#include "rows.hpp"

#ifdef TRISC_MATH
namespace ckernel::sfpu {

    // One face: dst[x] = dst[x] * dst[a] + dst[b] (fp32).
    inline void ktt_rows_affine(uint32_t x, uint32_t a, uint32_t b) {
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

namespace {

    constexpr uint32_t cb_x = 0, cb_w = 1, cb_b = 2, cb_gather = 3, cb_mask = 4,
                       cb_sums = 5;
    constexpr uint32_t cb_groups = 6, cb_ab = 7, cb_back = 8, cb_next = 9, cb_rows = 10;
    constexpr uint32_t cb_mix = 11, cb_stats = 12, cb_y = 16;

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

    inline void sum_rows(uint32_t dst) {
        sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(dst);
    }

    // DST tiles `first`, `second` packed to the two pages of fp32 CB `cb`, and of
    // `copy` too when it is a CB (not kNone).
    constexpr uint32_t kNone = 0xFFFFFFFFu;

    inline void pack_pair(
        uint32_t cb, uint32_t first, uint32_t second, uint32_t copy = kNone
    ) {
        tile_regs_commit();
        cb_reserve_back(cb, 2);
        if (copy != kNone) {
            cb_reserve_back(copy, 2);
        }
        tile_regs_wait();
        pack_reconfig_data_format(cb);
        pack_tile(first, cb);
        pack_tile(second, cb);
        if (copy != kNone) {
            pack_tile(first, copy);
            pack_tile(second, copy);
        }
        tile_regs_release();
        cb_push_back(cb, 2);
        if (copy != kNone) {
            cb_push_back(copy, 2);
        }
    }

}  // namespace

// Common runtime args: channel tiles Ct, rows per image, images, block, has weight, has
// bias, 1 / (elements per group) and eps (fp32 bits), channels per group, channels C,
// statistics wanted (each segment's per-channel mean and rstd rows to CB 12). Per core:
// q0, q1.
void kernel_main() {
    const ktt_group_norm::Geometry g{
        get_common_arg_val<uint32_t>(0),
        get_common_arg_val<uint32_t>(1),
        get_common_arg_val<uint32_t>(2)
    };
    const uint32_t block = get_common_arg_val<uint32_t>(3);
    const bool has_w = get_common_arg_val<uint32_t>(4) != 0;
    const bool has_b = get_common_arg_val<uint32_t>(5) != 0;
    const uint32_t inv_count = get_common_arg_val<uint32_t>(6);
    const uint32_t eps = get_common_arg_val<uint32_t>(7);
    const uint32_t cg = get_common_arg_val<uint32_t>(8);
    const uint32_t c = get_common_arg_val<uint32_t>(9);
    const bool stats_out = get_common_arg_val<uint32_t>(10) != 0;
    const uint32_t q1 = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_x, cb_y);
    for (uint32_t q = get_arg_val<uint32_t>(0); q < q1;) {
        const ktt_group_norm::Piece p(g, q, q1);
        // A piece ending in a row tile shared with the next image also takes that
        // image's a / b (to cb_next).
        const bool shared = g.shared(p.n) && p.last;
        const ktt_group_norm::Span span(p.cb, cg, c);
        for (uint32_t pass = 0; pass < (shared ? 2u : 1u); ++pass) {
            // Group sums (row 0 of the two cb_groups tiles, one column per group of the
            // tile), added up in L1 over the channel tiles the groups reach.
            cb_reserve_back(cb_groups, 2);
            for (uint32_t t = 0; t < span.tiles(); ++t) {
                // Per-channel sums: row 0 of DST 0 (sum x) and DST 1 (sum x^2).
                cb_wait_front(cb_gather, 2);
                tile_regs_acquire();
                copy(cb_gather, 0, 0);
                copy_tile(cb_gather, 1, 1);
                sum_rows(0);
                sum_rows(1);
                pack_pair(cb_sums, 0, 1);
                cb_pop_front(cb_gather, 2);

                // Channel c's sums to (c, group), summed over the channels.
                cb_wait_front(cb_sums, 2);
                cb_wait_front(cb_mask, 1);
                tile_regs_acquire();
                row_bcast(cb_sums, 0, 0);
                row_bcast(cb_sums, 1, 1);
                transpose_dest_init<DST_ACCUM_MODE>();
                transpose_dest<DST_ACCUM_MODE>(0);
                transpose_dest<DST_ACCUM_MODE>(1);
                copy(cb_mask, 0, 2);
                mul_binary_tile_init();
                mul_binary_tile(0, 2, 0);
                mul_binary_tile(1, 2, 1);
                sum_rows(0);
                sum_rows(1);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_groups);
                pack_reconfig_l1_acc(t > 0 ? 1 : 0);
                pack_tile<true>(0, cb_groups, 0);
                pack_tile<true>(1, cb_groups, 1);
                tile_regs_release();
                pack_reconfig_l1_acc(0);
                cb_pop_front(cb_sums, 2);
                cb_pop_front(cb_mask, 1);
            }
            cb_push_back(cb_groups, 2);

            // Back to the channels: group g's sum to (c, g) for its channels c,
            // transposed and summed over the groups: row 0, each channel its group's
            // sums. Then the statistics.
            cb_wait_front(cb_groups, 2);
            cb_wait_front(cb_back, 2);
            tile_regs_acquire();
            row_bcast(cb_groups, 0, 0);
            row_bcast(cb_groups, 1, 1);
            copy(cb_back, 0, 2);
            mul_binary_tile_init();
            mul_binary_tile(0, 2, 0);
            mul_binary_tile(1, 2, 1);
            transpose_dest_init<DST_ACCUM_MODE>();
            transpose_dest<DST_ACCUM_MODE>(0);
            transpose_dest<DST_ACCUM_MODE>(1);
            sum_rows(0);
            sum_rows(1);
            // DST 0: mean; DST 1: rstd = rsqrt(max(E x^2 - mean^2, 0) + eps).
            binop_with_scalar_tile_init();
            mul_unary_tile(0, inv_count);
            mul_unary_tile(1, inv_count);
            mul_binary_tile_init();
            mul_binary_tile(0, 0, 2);
            sub_binary_tile_init();
            sub_binary_tile(1, 2, 1);
            relu_tile_init();
            relu_tile(1);
            binop_with_scalar_tile_init();
            add_unary_tile(1, eps);
            rsqrt_tile_init();
            rsqrt_tile(1);
            if (stats_out && pass == 0 && p.i == 0) {
                // The piece starting a segment hands the mean and rstd rows to the
                // writer.
                pack_pair(cb_sums, 0, 1, cb_stats);
                cb_wait_front(cb_sums, 2);
                tile_regs_acquire();
                copy(cb_sums, 0, 0);
                copy_tile(cb_sums, 1, 1);
                cb_pop_front(cb_sums, 2);
            }
            // DST 1: a = rstd w; DST 0: b = bias - mean a; both zero at the padded
            // channels.
            if (has_w) {
                cb_wait_front(cb_w, 1);
                copy(cb_w, 0, 2);
                mul_binary_tile_init();
                mul_binary_tile(1, 2, 1);
            }
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            if (has_b) {
                cb_wait_front(cb_b, 1);
                copy(cb_b, 0, 2);
            } else {
                fill_tile_init();
                fill_tile(2, 0.0f);
            }
            sub_binary_tile_init();
            sub_binary_tile(2, 0, 0);
            copy(cb_back, 1, 2);
            mul_binary_tile_init();
            mul_binary_tile(0, 2, 0);
            mul_binary_tile(1, 2, 1);
            pack_pair(pass == 0 ? cb_ab : cb_next, 1, 0);
            cb_pop_front(cb_groups, 2);
            cb_pop_front(cb_back, 2);
            if (has_w) {
                cb_pop_front(cb_w, 1);
            }
            if (has_b) {
                cb_pop_front(cb_b, 1);
            }
        }

        cb_wait_front(cb_ab, 2);
        if (shared) {
            // The shared tile's a and b as whole tiles: the image's rows from cb_ab,
            // the next image's from cb_next.
            cb_wait_front(cb_next, 2);
            cb_wait_front(cb_rows, 2);
            cb_reserve_back(cb_mix, 2);
            for (uint32_t which = 0; which < 2; ++which) {
                tile_regs_acquire();
                row_bcast(cb_ab, which, 0);
                row_bcast(cb_next, which, 1);
                reconfig_data_format(cb_rows, cb_rows);
                unary_bcast_init<BroadcastType::COL>(cb_rows);
                unary_bcast<BroadcastType::COL>(cb_rows, 0, 2);
                unary_bcast<BroadcastType::COL>(cb_rows, 1, 3);
                mul_binary_tile_init();
                mul_binary_tile(0, 2, 0);
                mul_binary_tile(1, 3, 1);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_mix);
                pack_tile<true>(0, cb_mix, which);
                tile_regs_release();
            }
            cb_push_back(cb_mix, 2);
            cb_pop_front(cb_next, 2);
            cb_pop_front(cb_rows, 2);
            cb_wait_front(cb_mix, 2);
        }
        for (uint32_t t = 0; t < p.len; t += block) {
            const uint32_t n = p.len - t < block ? p.len - t : block;
            cb_wait_front(cb_x, block);
            cb_reserve_back(cb_y, block);
            uint32_t count = 0;
            for (uint32_t k = 0; k < n; k += count) {
                // The shared tile runs alone, with its own a / b tiles.
                const bool mixed = shared && p.ends_image(g, t + k);
                count = n - k < 2 || mixed || (shared && p.ends_image(g, t + k + 1))
                    ? 1
                    : 2;
                tile_regs_acquire();
                if (mixed) {
                    copy(cb_mix, 0, 0);
                    copy_tile(cb_mix, 1, 1);
                } else {
                    row_bcast(cb_ab, 0, 0);
                    row_bcast(cb_ab, 1, 1);
                }
                copy(cb_x, k, 2);
                if (count == 2) {
                    copy_tile(cb_x, k + 1, 3);
                }
                mul_binary_tile_init();
                for (uint32_t i = 0; i < count; ++i) {
                    MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                        DST_SYNC_MODE,
                        DST_ACCUM_MODE,
                        ktt_rows_affine,
                        2 + i,
                        0,
                        1,
                        VectorMode::RC
                    )));
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_y);
                for (uint32_t i = 0; i < count; ++i) {
                    pack_tile<true>(2 + i, cb_y, k + i);
                }
                tile_regs_release();
            }
            cb_push_back(cb_y, block);
            cb_pop_front(cb_x, block);
        }
        cb_pop_front(cb_ab, 2);
        if (shared) {
            cb_pop_front(cb_mix, 2);
        }
        q += p.len;
    }
}
