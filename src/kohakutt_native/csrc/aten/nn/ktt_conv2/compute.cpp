// Gather conv compute: per work item and output row tile m with taps, its phase's taps'
// 32 row-major input rows (CB 0, the item's Kc channel tiles in K pages) tilized into
// CB 2 in K blocks of up to B taps; for each of the item's nt output column tiles j,
// out[j] = sum over taps t and j's channel tiles k of act[t, k] @ w[j, t, k],
// accumulated in DST (fp32). One block: packed into CB 16. More blocks: each block's
// sums packed into CB 24 with packer L1 accumulation (fp32), then copied through DST
// into CB 16.
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/debug/kernel_profiler.hpp"
#include "taps.hpp"

namespace {

    // Tilizes the kct channel tiles of a tap from CB `icb`, its first row `first`
    // 64-byte units from the CB's read pointer (rows as far apart as the width
    // tilize_init was given), into CB `ocb` tiles 0 .. kct - 1.
    ALWI void tilize_tap(uint32_t icb, uint32_t first, uint32_t kct, uint32_t ocb) {
        for (uint32_t k = 0; k < kct; ++k) {
            UNPACK((llk_unpack_tilize(icb, first + k)));
        }
        for (uint32_t t = 0; t < kct; ++t) {
            MATH((llk_math_wait_for_dest_available()));
            PACK((llk_packer_wait_for_math_done()));
            MATH((llk_math_eltwise_unary_datacopy<
                  DataCopyType::A2D,
                  DST_ACCUM_MODE,
                  BroadcastType::NONE,
                  UnpackToDestEn>(0, icb)));
            PACK((llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(0, ocb, t)));
            MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
            PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
        }
    }

}  // namespace

// Compile-time args: the argument-layout version. Common runtime args: version, taps
// per K block (B), input rows per output position (per: Sw in window mode, else 1),
// window mode (0 / 1), window rows (span), then the shape (taps.hpp). Per core: items,
// then an Item (taps.hpp) per item. Window mode: taps tilized from the reader's CB 3
// pages (one per tap row, span rows of the item's K tiles), else from CB 0 (K pages per
// tap).
void kernel_main() {
    using namespace ktt_conv2;
    if (!args_match<0>()) {
        return;
    }
    const uint32_t block = get_common_arg_val<uint32_t>(1);
    const uint32_t per = get_common_arg_val<uint32_t>(2);
    const uint32_t window = get_common_arg_val<uint32_t>(3);
    const uint32_t span = get_common_arg_val<uint32_t>(4);
    const Shape s = read_shape(5);
    constexpr uint32_t cb_rows = 0, cb_w = 1, cb_act = 2, cb_window = 3, cb_out = 16,
                       cb_part = 24;
    const uint32_t taps = s.kh * s.kw;
    const uint32_t w_tiles = s.nt_max * taps * s.kmax;
    const uint32_t cb_in = window ? cb_window : cb_rows;
    const uint32_t page_units = span * s.kpages;  // 64-byte units per window page

    // Matmul order: act (in0) in srcB, weights (in1) in srcA.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act, cb_w, cb_out);
    Phase ph;
    // The CB srcA last unpacked (CB 24 unpacks to DST as fp32: transitions name both).
    uint32_t last_a = cb_w;
    const uint32_t items = get_arg_val<uint32_t>(0);
    for (uint32_t it = 0; it < items; ++it) {
        const Item item = read_item(it);
        const uint32_t m0 = item.m0, m1 = item.m1, n0 = item.n0, nt = item.nt;
        uint32_t kc0, kc1, unused;
        channel_tiles(n0, s, kc0, unused);
        channel_tiles(n0 + nt - 1, s, unused, kc1);
        const uint32_t kct = kc1 - kc0;
        // The weights are awaited at the item's first matmul (tilizing runs before).
        bool weights = false;

        for (uint32_t m = m0; m < m1; ++m) {
            load_phase(s, m, ph, item.a0, item.a1);
            const uint32_t count = ph.nh * ph.nw;
            if (count == 0) {
                continue;
            }
            const uint32_t blocks = (count + block - 1) / block;
            if (window) {
                // KH pages per row tile (tap rows past nh unused), so pages never wrap.
                DeviceZoneScopedN("c-win");
                cb_wait_front(cb_window, s.rows_h);
            }
            for (uint32_t b = 0; b < blocks; ++b) {
                const uint32_t t0 = b * block;
                const uint32_t t1 = t0 + block < count ? t0 + block : count;
                // Tilize unpacks CB 0 / CB 3 through srcA; the matmul then takes act
                // and weights.
                reconfig_data_format_srca(last_a, cb_in);
                pack_reconfig_data_format(cb_act);
                tilize_init(cb_in, per * kct, cb_act);
                uint32_t ta = t0 / ph.nw, tb = t0 - ta * ph.nw;
                for (uint32_t t = t0; t < t1; ++t) {
                    DeviceZoneScopedN("c-tilize");
                    cb_reserve_back(cb_act, s.kpages);
                    if (window) {
                        // Tap (ta, tb): window page ta from row coff - clo on.
                        tilize_tap(
                            cb_window,
                            ta * page_units + (ph.coff[tb] - ph.clo) * kct,
                            kct,
                            cb_act
                        );
                    } else {
                        cb_wait_front(cb_rows, s.kpages);
                        tilize_tap(cb_rows, 0, kct, cb_act);
                        cb_pop_front(cb_rows, s.kpages);
                    }
                    cb_push_back(cb_act, s.kpages);
                    if (++tb == ph.nw) {
                        tb = 0;
                        ++ta;
                    }
                }
                // A short block fills CB 2 to its whole size (it holds one block).
                if (t1 - t0 < block) {
                    cb_reserve_back(cb_act, (block - (t1 - t0)) * s.kpages);
                    cb_push_back(cb_act, (block - (t1 - t0)) * s.kpages);
                }
                tilize_uninit(cb_in, cb_act);
                reconfig_data_format<SrcOrder::Reverse>(cb_act, cb_w);
                last_a = cb_w;

                matmul_init(cb_act, cb_w);
                if (!weights) {
                    DeviceZoneScopedN("c-wts");
                    cb_wait_front(cb_w, w_tiles);
                    weights = true;
                }
                DeviceZoneScopedN("c-mm");
                cb_wait_front(cb_act, block * s.kpages);
                tile_regs_acquire();
                for (uint32_t jl = 0; jl < nt; ++jl) {
                    uint32_t k0, k1;
                    channel_tiles(n0 + jl, s, k0, k1);
                    for (uint32_t t = t0; t < t1; ++t) {
                        const uint32_t
                            tap = (ph.ridx[t / ph.nw] * s.kw + ph.cidx[t % ph.nw]);
                        for (uint32_t k = k0; k < k1; ++k) {
                            matmul_tiles(
                                cb_act,
                                cb_w,
                                (t - t0) * s.kpages + (k - kc0),
                                (jl * taps + tap) * s.kmax + (k - k0),
                                jl
                            );
                        }
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                if (blocks == 1) {
                    pack_reconfig_data_format(cb_out);
                    cb_reserve_back(cb_out, s.nt_max);
                    for (uint32_t jl = 0; jl < nt; ++jl) {
                        pack_tile(jl, cb_out);
                    }
                    tile_regs_release();
                    cb_push_back(cb_out, s.nt_max);
                } else {
                    pack_reconfig_data_format(cb_part);
                    if (b == 0) {
                        cb_reserve_back(cb_part, s.nt_max);
                    }
                    pack_reconfig_l1_acc(b > 0 ? 1 : 0);
                    for (uint32_t jl = 0; jl < nt; ++jl) {
                        pack_tile<true>(jl, cb_part, jl);
                    }
                    tile_regs_release();
                    pack_reconfig_l1_acc(0);
                }
                cb_pop_front(cb_act, block * s.kpages);
            }
            if (blocks > 1) {
                // The fp32 sums through DST into the bf16 output.
                cb_push_back(cb_part, s.nt_max);
                cb_wait_front(cb_part, s.nt_max);
                reconfig_data_format_srca(cb_w, cb_part);
                pack_reconfig_data_format(cb_out);
                copy_init(cb_part);
                last_a = cb_part;
                cb_reserve_back(cb_out, s.nt_max);
                tile_regs_acquire();
                for (uint32_t jl = 0; jl < nt; ++jl) {
                    copy_tile(cb_part, jl, jl);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t jl = 0; jl < nt; ++jl) {
                    pack_tile(jl, cb_out);
                }
                tile_regs_release();
                cb_push_back(cb_out, s.nt_max);
                cb_pop_front(cb_part, s.nt_max);
            }
            if (window) {
                cb_pop_front(cb_window, s.rows_h);
            }
        }
        if (!weights) {
            cb_wait_front(cb_w, w_tiles);
        }
        cb_pop_front(cb_w, w_tiles);
    }
}
