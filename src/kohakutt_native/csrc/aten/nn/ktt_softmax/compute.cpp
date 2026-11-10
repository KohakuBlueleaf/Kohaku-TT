// Softmax compute (fp32 DST, SFPU only), per tile row of the core's segment: row max, e
// = exp(x - m) to an fp32 ring, row sum and r = 1 / sum (group: merged), y = e r.
#include <cstdint>
#include <limits>
#include <utility>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_dest.h"
#include "args.hpp"

#ifdef TRISC_MATH
namespace ckernel::sfpu {

    // Units: tile 32, face 8 (f: rows 16 (f / 2), columns 16 (f % 2)); unit d: rows 4
    // (d / 2), parity d % 2; lane l: row + l / 8, column 2 (l % 8) + d % 2.

    constexpr int kNegInf = static_cast<int>(0xff800000u);
    constexpr int kExpMask = 0x7f800000;

    sfpi_inline sfpi::vFloat vmax(sfpi::vFloat a, sfpi::vFloat b) {
        return sfpi::min_max(a, b).second;
    }

    template <bool kMax>
    sfpi_inline sfpi::vFloat op(sfpi::vFloat a, sfpi::vFloat b) {
        if constexpr (kMax) {
            return vmax(a, b);
        } else {
            return a + b;
        }
    }

    template <int K>
    sfpi_inline sfpi::vFloat ror(sfpi::vFloat v) {
#pragma GCC unroll 4
        for (int k = 0; k < K; ++k) {
            v = sfpi::subvec_shflror1(v);
        }
        return v;
    }

    // The DST pointer 8 units on (one increment moves at most 3).
    sfpi_inline void skip_face() {
        sfpi::dst_reg += 3;
        sfpi::dst_reg += 3;
        sfpi::dst_reg += 2;
    }

    // exp(v): -inf and underflow to 0, NaN (either sign) to NaN.
    sfpi_inline sfpi::vFloat ex(sfpi::vFloat v) {
        return _sfpu_exp_fp32_accurate_<false>(v);
    }

    // 1 / x for a finite x > 0.
    sfpi_inline sfpi::vFloat recip(sfpi::vFloat x) {
        sfpi::vFloat y = sfpi::approx_recip(x);
        y = y * (2.0f - x * y);
        return y * (2.0f - x * y);
    }

    // r, or NaN where m is not finite (a row max of -inf: every element -inf).
    sfpi_inline sfpi::vFloat nan_unless_finite(sfpi::vFloat r, sfpi::vFloat m) {
        v_if((sfpi::as<sfpi::vInt>(m) & kExpMask) == kExpMask) {
            r = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;
        return r;
    }

    // Tiles 1 .. N maxed (kMax) or added into tile 0.
    template <bool kMax, int N>
    inline void ktt_sm_acc(uint32_t, uint32_t, uint32_t) {
        for (int u = 0; u < 32; ++u) {
            sfpi::vFloat a = sfpi::dst_reg[0];
            a = op<kMax>(a, sfpi::dst_reg[32]);
            if constexpr (N > 1) {
                a = op<kMax>(a, sfpi::dst_reg[64]);
            }
            if constexpr (N > 2) {
                a = op<kMax>(a, sfpi::dst_reg[96]);
            }
            sfpi::dst_reg[0] = a;
            sfpi::dst_reg++;
        }
    }

    // DST tile `tile`: columns >= cols and rows >= rows replaced by -inf (kNeg) or 0.
    template <bool kNeg>
    inline void ktt_sm_mask(
        uint32_t, uint32_t, uint32_t, uint32_t cols, uint32_t rows, uint32_t tile
    ) {
        for (uint32_t t = 0; t < tile; ++t) {
#pragma GCC unroll 11
            for (int k = 0; k < 10; ++k) {
                sfpi::dst_reg += 3;
            }
            sfpi::dst_reg += 2;
        }
        const sfpi::vInt col = sfpi::vConstTileId & 14;
        const sfpi::vInt row = sfpi::vConstTileId >> 4;
        const float fill = kNeg ? -std::numeric_limits<float>::infinity() : 0.0f;
        const int nc = static_cast<int>(cols), nr = static_cast<int>(rows);
#pragma GCC unroll 32
        for (int u = 0; u < 32; ++u) {
            // The unit's lanes cover columns c0 + 0, 2, .. 14 and rows r0 + 0 .. 3.
            const int c0 = 16 * ((u >> 3) & 1) + (u & 1);
            const int r0 = 16 * (u >> 4) + 4 * ((u >> 1) & 3);
            if (c0 >= nc || r0 >= nr) {
                sfpi::dst_reg[u] = fill;
            } else if (nr >= 32 && c0 + 14 >= nc) {
                sfpi::vFloat v = sfpi::dst_reg[u];
                v_if(col >= nc - c0) {
                    v = fill;
                }
                v_endif;
                sfpi::dst_reg[u] = v;
            } else if (c0 + 14 >= nc || r0 + 3 >= nr) {
                sfpi::vFloat v = sfpi::dst_reg[u];
                v_if(col >= nc - c0) {
                    v = fill;
                }
                v_elseif(row >= nr - r0) {
                    v = fill;
                }
                v_endif;
                sfpi::dst_reg[u] = v;
            }
        }
    }

    // Tile T folded over its 32 columns (max or sum) into a column broadcast; with
    // kMask the columns >= cols take no part.
    template <bool kMax, bool kMask, int T>
    inline void ktt_sm_fold(uint32_t, uint32_t, uint32_t, uint32_t cols) {
        const sfpi::vInt col = sfpi::vConstTileId & 14;
        const float id = kMax ? -std::numeric_limits<float>::infinity() : 0.0f;
        for (int h = 0; h < 2; ++h) {
            for (int b = 0; b < 4; ++b) {
                sfpi::vFloat v0 = sfpi::dst_reg[32 * T];
                sfpi::vFloat v1 = sfpi::dst_reg[32 * T + 1];
                sfpi::vFloat v2 = sfpi::dst_reg[32 * T + 8];
                sfpi::vFloat v3 = sfpi::dst_reg[32 * T + 9];
                if constexpr (kMask) {
                    const int n = static_cast<int>(cols);
                    v_if(col >= n) {
                        v0 = id;
                    }
                    v_endif;
                    v_if(col >= n - 1) {
                        v1 = id;
                    }
                    v_endif;
                    v_if(col >= n - 16) {
                        v2 = id;
                    }
                    v_endif;
                    v_if(col >= n - 17) {
                        v3 = id;
                    }
                    v_endif;
                }
                sfpi::vFloat a = op<kMax>(op<kMax>(v0, v1), op<kMax>(v2, v3));
                a = op<kMax>(a, ror<4>(a));
                a = op<kMax>(a, ror<2>(a));
                a = op<kMax>(a, ror<1>(a));
                sfpi::dst_reg[32 * T] = a;
                sfpi::dst_reg[32 * T + 1] = a;
                sfpi::dst_reg[32 * T + 8] = a;
                sfpi::dst_reg[32 * T + 9] = a;
                sfpi::dst_reg += 2;
            }
            skip_face();
        }
    }

    // Tile 0 (the row max m, broadcast): faces 0 / 2 to the exp shift (-inf -> 0),
    // faces 1 / 3 keep m; tile 1 = m when kCopy.
    template <bool kCopy>
    inline void ktt_sm_shift(uint32_t, uint32_t, uint32_t) {
        for (int h = 0; h < 2; ++h) {
            for (int d = 0; d < 8; ++d) {
                sfpi::vFloat m = sfpi::dst_reg[0];
                if constexpr (kCopy) {
                    sfpi::dst_reg[32] = m;
                    sfpi::dst_reg[40] = m;
                }
                v_if(sfpi::as<sfpi::vInt>(m) == kNegInf) {
                    m = 0.0f;
                }
                v_endif;
                sfpi::dst_reg[0] = m;
                sfpi::dst_reg++;
            }
            skip_face();
        }
    }

    // Tiles 1 .. N: x to exp(x - m), m the shift in tile 0's faces 0 / 2.
    template <int N>
    inline void ktt_sm_exp(uint32_t, uint32_t, uint32_t) {
        for (int h = 0; h < 2; ++h) {
            for (int d = 0; d < 8; ++d) {
                const sfpi::vFloat m = sfpi::dst_reg[0];
                sfpi::dst_reg[32] = ex(sfpi::dst_reg[32] - m);
                sfpi::dst_reg[40] = ex(sfpi::dst_reg[40] - m);
                if constexpr (N > 1) {
                    sfpi::dst_reg[64] = ex(sfpi::dst_reg[64] - m);
                    sfpi::dst_reg[72] = ex(sfpi::dst_reg[72] - m);
                }
                if constexpr (N > 2) {
                    sfpi::dst_reg[96] = ex(sfpi::dst_reg[96] - m);
                    sfpi::dst_reg[104] = ex(sfpi::dst_reg[104] - m);
                }
                sfpi::dst_reg++;
            }
            skip_face();
        }
    }

    // Tiles 1 .. N times tile 0 (r, broadcast).
    template <int N>
    inline void ktt_sm_mul(uint32_t, uint32_t, uint32_t) {
        for (int u = 0; u < 32; ++u) {
            const sfpi::vFloat r = sfpi::dst_reg[0];
            sfpi::dst_reg[32] = sfpi::dst_reg[32] * r;
            if constexpr (N > 1) {
                sfpi::dst_reg[64] = sfpi::dst_reg[64] * r;
            }
            if constexpr (N > 2) {
                sfpi::dst_reg[96] = sfpi::dst_reg[96] * r;
            }
            sfpi::dst_reg++;
        }
    }

    // Tile 0 (the row sum s) to 1 / s; NaN where s or the row max (tile 1's faces 1 /
    // 3) is not finite.
    template <int = 0>
    inline void ktt_sm_recip(uint32_t, uint32_t, uint32_t) {
        for (int h = 0; h < 2; ++h) {
            for (int d = 0; d < 8; ++d) {
                const sfpi::vFloat s = sfpi::dst_reg[0];
                const sfpi::vFloat r = nan_unless_finite(
                    nan_unless_finite(recip(s), s), sfpi::dst_reg[40]
                );
                sfpi::dst_reg[0] = r;
                sfpi::dst_reg[8] = r;
                sfpi::dst_reg++;
            }
            skip_face();
        }
    }

    // Transposed gather (column c: core c), tiles 0 m_c, 1 s_c, 3 the row max M: tile 1
    // = exp(m_c - M) s_c for c < g, 0 elsewhere.
    template <int = 0>
    inline void ktt_sm_merge(uint32_t, uint32_t, uint32_t, uint32_t g) {
        const sfpi::vInt col = sfpi::vConstTileId & 14;
        for (int u = 0; u < 32; ++u) {
            const int c0 = 16 * ((u >> 3) & 1) + (u & 1);
            const sfpi::vFloat m = sfpi::dst_reg[0];
            sfpi::vFloat w = ex(m - sfpi::dst_reg[96]) * sfpi::dst_reg[32];
            v_if(col >= static_cast<int>(g) - c0) {
                w = 0.0f;
            }
            v_endif;
            sfpi::dst_reg[32] = w;
            sfpi::dst_reg++;
        }
    }

    // Tile 2 = exp(m - M) / S: m the core's own row max (tile 2's faces 1 / 3), M (tile
    // 3) and S (tile 1) the group's.
    template <int = 0>
    inline void ktt_sm_rate(uint32_t, uint32_t, uint32_t) {
        for (int h = 0; h < 2; ++h) {
            for (int d = 0; d < 8; ++d) {
                const sfpi::vFloat m = sfpi::dst_reg[72], top = sfpi::dst_reg[96];
                const sfpi::vFloat sum = sfpi::dst_reg[32];
                const sfpi::vFloat r = nan_unless_finite(
                    nan_unless_finite(ex(m - top) * recip(sum), top), sum
                );
                sfpi::dst_reg[64] = r;
                sfpi::dst_reg[72] = r;
                sfpi::dst_reg++;
            }
            skip_face();
        }
    }

}  // namespace ckernel::sfpu
#endif

// One custom SFPU function over the DST half (it indexes tiles itself).
#define KTT_SM(FN, T, ...)                                             \
    MATH((SFPU_BINARY_CALL(                                            \
        DST_SYNC_MODE, DST_ACCUM_MODE, FN, T, 0, 0, 0, VectorMode::None, \
        ##__VA_ARGS__                                                  \
    )))

namespace {

    constexpr uint32_t cb_x = 0, cb_gather = 4, cb_m = 5, cb_r = 6, cb_e = 7;
    constexpr uint32_t cb_y = 16, cb_part = 17, cb_row = 24;

    inline void copy_from(uint32_t cb) {
        reconfig_data_format_srca(cb);
        copy_init(cb);
    }

    inline void sfpu_prep() {
        mul_binary_tile_init();
    }

    inline void transpose(uint32_t dst) {
        transpose_dest_init<DST_ACCUM_MODE>();
        transpose_dest<DST_ACCUM_MODE>(dst);
    }

    // Tiles 1 .. n of DST combined into tile 0 (max or add).
    template <bool kMax>
    inline void accumulate(uint32_t n) {
        sfpu_prep();
        if (n == 1) {
            KTT_SM(ktt_sm_acc, (kMax, 1));
        } else if (n == 2) {
            KTT_SM(ktt_sm_acc, (kMax, 2));
        } else {
            KTT_SM(ktt_sm_acc, (kMax, 3));
        }
    }

    // DST tile t's padding (columns >= cols, rows >= rows) to -inf (kNeg) or 0.
    template <bool kNeg>
    inline void mask(uint32_t t, uint32_t cols, uint32_t rows) {
        sfpu_prep();
        KTT_SM(ktt_sm_mask, (kNeg), cols, rows, t);
    }

}  // namespace

// Common: tile rows, g, x pages per row, tile rows per slice, a slice's valid last
// rows. Per core: group, index, column tiles, valid columns of the last (32: all).
void kernel_main() {
    if (!ktt_softmax::args_ok<0>(6)) {
        return;
    }
    const uint32_t rows = get_common_arg_val<uint32_t>(0);
    const uint32_t gsize = get_common_arg_val<uint32_t>(1);
    const uint32_t stride = get_common_arg_val<uint32_t>(2);
    const uint32_t mt = get_common_arg_val<uint32_t>(3);
    const uint32_t hlast = get_common_arg_val<uint32_t>(4);
    const uint32_t seg = get_arg_val<uint32_t>(2);
    const uint32_t wlast = get_arg_val<uint32_t>(3);
    const uint32_t last = seg - 1;
    const bool ragged = wlast < 32;
    const bool group = gsize > 1;

    compute_kernel_hw_startup(cb_x, cb_y);

    while (true) {
        cb_wait_front(cb_row, 1);
        const uint32_t r = read_tile_value(cb_row, 0, 0);
        cb_pop_front(cb_row, 1);
        if (r >= rows) {
            break;
        }
        // Valid rows of this tile row: the slice's last tile row may be ragged.
        const uint32_t valid = hlast < 32 && r % mt == mt - 1 ? hlast : 32;
        cb_wait_front(cb_x, stride);

        // Row max: tile 0 accumulates, tiles 1 .. 3 take the next x tiles.
        tile_regs_acquire();
        copy_from(cb_x);
        copy_tile(cb_x, 0, 0);
        if (ragged && last == 0) {
            mask<true>(0, wlast, 32);
        }
        for (uint32_t j = 1; j < seg; j += 3) {
            const uint32_t n = seg - j < 3 ? seg - j : 3;
            for (uint32_t i = 0; i < n; ++i) {
                copy_tile(cb_x, j + i, 1 + i);
            }
            if (ragged && j + n - 1 == last) {
                mask<true>(n, wlast, 32);
            }
            accumulate<true>(n);
        }
        sfpu_prep();
        KTT_SM(ktt_sm_fold, (true, false, 0), 32);
        if (group) {
            // The raw max as a row (tile 1) for the group exchange.
            KTT_SM(ktt_sm_shift, (true));
            transpose(1);
        } else {
            KTT_SM(ktt_sm_shift, (false));
        }
        tile_regs_commit();
        cb_reserve_back(cb_m, 1);
        if (group) {
            cb_reserve_back(cb_part, 2);
        }
        tile_regs_wait();
        pack_reconfig_data_format(cb_m);
        pack_tile(0, cb_m);
        if (group) {
            pack_tile<true>(1, cb_part, 0);
        }
        tile_regs_release();
        cb_push_back(cb_m, 1);

        // e = exp(x - m), up to three tiles per acquire, to the fp32 ring.
        cb_wait_front(cb_m, 1);
        cb_reserve_back(cb_e, stride);
        for (uint32_t j = 0; j < seg; j += 3) {
            const uint32_t n = seg - j < 3 ? seg - j : 3;
            tile_regs_acquire();
            copy_from(cb_m);
            copy_tile(cb_m, 0, 0);
            copy_from(cb_x);
            for (uint32_t i = 0; i < n; ++i) {
                copy_tile(cb_x, j + i, 1 + i);
            }
            if (ragged && j + n - 1 == last) {
                mask<true>(n, wlast, 32);
            }
            sfpu_prep();
            if (n == 1) {
                KTT_SM(ktt_sm_exp, (1));
            } else if (n == 2) {
                KTT_SM(ktt_sm_exp, (2));
            } else {
                KTT_SM(ktt_sm_exp, (3));
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_e);
            for (uint32_t i = 0; i < n; ++i) {
                pack_tile<true>(1 + i, cb_e, j + i);
            }
            tile_regs_release();
        }
        cb_push_back(cb_e, stride);
        cb_pop_front(cb_x, stride);

        // Row sum of e; r = 1 / sum, or the group merge.
        cb_wait_front(cb_e, stride);
        tile_regs_acquire();
        copy_from(cb_e);
        copy_tile(cb_e, 0, 0);
        for (uint32_t j = 1; j < seg; j += 3) {
            const uint32_t n = seg - j < 3 ? seg - j : 3;
            for (uint32_t i = 0; i < n; ++i) {
                copy_tile(cb_e, j + i, 1 + i);
            }
            accumulate<false>(n);
        }
        sfpu_prep();
        KTT_SM(ktt_sm_fold, (false, false, 0), 32);
        if (group) {
            transpose(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_part);
            pack_tile<true>(0, cb_part, 1);
            tile_regs_release();
            cb_push_back(cb_part, 2);

            // Rows c of the gather: core c's (max, sum) rows; transposed to columns.
            cb_wait_front(cb_gather, 2);
            tile_regs_acquire();
            copy_from(cb_gather);
            copy_tile(cb_gather, 0, 0);
            copy_tile(cb_gather, 1, 1);
            copy_tile(cb_gather, 0, 3);
            copy_from(cb_m);
            copy_tile(cb_m, 0, 2);
            transpose(0);
            transpose_dest<DST_ACCUM_MODE>(1);
            transpose_dest<DST_ACCUM_MODE>(3);
            sfpu_prep();
            KTT_SM(ktt_sm_fold, (true, true, 3), gsize);
            KTT_SM(ktt_sm_merge, (), gsize);
            KTT_SM(ktt_sm_fold, (false, true, 1), gsize);
            KTT_SM(ktt_sm_rate, ());
            tile_regs_commit();
            cb_pop_front(cb_gather, 2);
            cb_pop_front(cb_m, 1);
            cb_reserve_back(cb_r, 1);
            tile_regs_wait();
            pack_reconfig_data_format(cb_r);
            pack_tile(2, cb_r);
            tile_regs_release();
            cb_push_back(cb_r, 1);
        } else {
            copy_from(cb_m);
            copy_tile(cb_m, 0, 1);
            sfpu_prep();
            KTT_SM(ktt_sm_recip, ());
            tile_regs_commit();
            cb_pop_front(cb_m, 1);
            cb_reserve_back(cb_r, 1);
            tile_regs_wait();
            pack_reconfig_data_format(cb_r);
            pack_tile(0, cb_r);
            tile_regs_release();
            cb_push_back(cb_r, 1);
        }

        // y = e r in write order from tile r mod seg, up to three tiles per acquire.
        cb_wait_front(cb_r, 1);
        pack_reconfig_data_format(cb_y);
        uint32_t k = r % seg;
        for (uint32_t done = 0; done < seg;) {
            const uint32_t n = seg - done < 3 ? seg - done : 3;
            tile_regs_acquire();
            copy_from(cb_r);
            copy_tile(cb_r, 0, 0);
            uint32_t at[3];
            for (uint32_t i = 0; i < n; ++i) {
                at[i] = k;
                copy_tile(cb_e, k, 1 + i);
                k = k + 1 == seg ? 0 : k + 1;
            }
            sfpu_prep();
            if (n == 1) {
                KTT_SM(ktt_sm_mul, (1));
            } else if (n == 2) {
                KTT_SM(ktt_sm_mul, (2));
            } else {
                KTT_SM(ktt_sm_mul, (3));
            }
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t cols = ragged && at[i] == last ? wlast : 32;
                if (cols < 32 || valid < 32) {
                    mask<false>(1 + i, cols, valid);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < n; ++i) {
                cb_reserve_back(cb_y, 1);
                pack_tile(1 + i, cb_y);
                cb_push_back(cb_y, 1);
            }
            tile_regs_release();
            done += n;
        }
        cb_pop_front(cb_r, 1);
        cb_pop_front(cb_e, stride);
    }
}
