// Weight-gradient compute. Per unit (see the reader): the chunk's grad rows tilized
// (CB 4) and transposed into G^T tiles (CB 5, resident: [row tile][column tile]); then
// per tap the chunk's input rows tilized into CB 2 ([row tile][channel tile]) and, per
// weight tile (output column tile j, channel tile k of j's groups) in DST groups of 4,
// the sum over the chunk's row tiles of G^T[m, j] @ A[m, k] (32 output x 32 input
// channels) accumulated in DST (fp32) and packed as fp32 into CB 16.
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tilize.h"
#include "api/compute/transpose.h"
#include "taps.hpp"

// Compile-time args: the argument-layout version. Common runtime args: version, unit
// step, units, chunks, row tiles per chunk, row tiles, Ot, Kmax, Cg, Og, O, KH, KW, K,
// nt. Per core: first unit.
void kernel_main() {
    using namespace ktt_conv2;
    if (!args_match<0>()) {
        return;
    }
    uint32_t a = 1;
    const uint32_t step = get_common_arg_val<uint32_t>(a++);
    const uint32_t units = get_common_arg_val<uint32_t>(a++);
    const uint32_t chunks = get_common_arg_val<uint32_t>(a++);
    const uint32_t per_chunk = get_common_arg_val<uint32_t>(a++);
    const uint32_t mt = get_common_arg_val<uint32_t>(a++);
    Shape s{};
    s.ot = get_common_arg_val<uint32_t>(a++);
    s.kmax = get_common_arg_val<uint32_t>(a++);
    s.cg = get_common_arg_val<uint32_t>(a++);
    s.og = get_common_arg_val<uint32_t>(a++);
    s.o = get_common_arg_val<uint32_t>(a++);
    s.kh = get_common_arg_val<uint32_t>(a++);
    s.kw = get_common_arg_val<uint32_t>(a++);
    s.kpages = get_common_arg_val<uint32_t>(a++);
    s.nt_max = get_common_arg_val<uint32_t>(a++);
    const uint32_t start = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_rows = 0, cb_g = 1, cb_act = 2, cb_g_tiles = 4, cb_gt = 5;
    constexpr uint32_t cb_out = 16;
    const uint32_t taps = s.kh * s.kw;

    // Matmul order: G^T (in0) in srcB, A (in1) in srcA.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_gt, cb_act, cb_out);
    for (uint32_t u = start; u < units; u += step) {
        const uint32_t n0 = (u / chunks) * s.nt_max, sc = u % chunks;
        const uint32_t nt = s.ot - n0 < s.nt_max ? s.ot - n0 : s.nt_max;
        const uint32_t m0 = sc * per_chunk;
        const uint32_t count = (m0 + per_chunk < mt ? m0 + per_chunk : mt) - m0;
        uint32_t kc0, kc1, unused;
        channel_tiles(n0, s, kc0, unused);
        channel_tiles(n0 + nt - 1, s, unused, kc1);
        const uint32_t kct = kc1 - kc0;

        // G^T tiles of the chunk (CB 5 and CB 2 move by whole chunks, short ones
        // padded).
        cb_reserve_back(cb_gt, per_chunk * s.nt_max);
        for (uint32_t m = 0; m < count; ++m) {
            reconfig_data_format_srca(cb_g);
            pack_reconfig_data_format(cb_g_tiles);
            tilize_init(cb_g, nt, cb_g_tiles);
            cb_wait_front(cb_g, s.nt_max);
            cb_reserve_back(cb_g_tiles, s.nt_max);
            tilize_block(cb_g, nt, cb_g_tiles);
            cb_push_back(cb_g_tiles, s.nt_max);
            cb_pop_front(cb_g, s.nt_max);
            tilize_uninit(cb_g, cb_g_tiles);

            reconfig_data_format_srca(cb_g_tiles);
            pack_reconfig_data_format(cb_gt);
            transpose_init(cb_g_tiles);
            cb_wait_front(cb_g_tiles, s.nt_max);
            tile_regs_acquire();
            for (uint32_t jl = 0; jl < nt; ++jl) {
                transpose_tile(cb_g_tiles, jl, jl);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t jl = 0; jl < nt; ++jl) {
                pack_tile<true>(jl, cb_gt, m * s.nt_max + jl);
            }
            tile_regs_release();
            cb_pop_front(cb_g_tiles, s.nt_max);
        }
        cb_push_back(cb_gt, per_chunk * s.nt_max);
        cb_wait_front(cb_gt, per_chunk * s.nt_max);

        for (uint32_t t = 0; t < taps; ++t) {
            reconfig_data_format_srca(cb_rows);
            pack_reconfig_data_format(cb_act);
            tilize_init(cb_rows, kct, cb_act);
            for (uint32_t m = 0; m < count; ++m) {
                cb_wait_front(cb_rows, s.kpages);
                cb_reserve_back(cb_act, s.kpages);
                tilize_block(cb_rows, kct, cb_act);
                cb_push_back(cb_act, s.kpages);
                cb_pop_front(cb_rows, s.kpages);
            }
            if (count < per_chunk) {
                cb_reserve_back(cb_act, (per_chunk - count) * s.kpages);
                cb_push_back(cb_act, (per_chunk - count) * s.kpages);
            }
            tilize_uninit(cb_rows, cb_act);
            reconfig_data_format<SrcOrder::Reverse>(cb_gt, cb_act);
            pack_reconfig_data_format(cb_out);
            matmul_init(cb_gt, cb_act);
            cb_wait_front(cb_act, per_chunk * s.kpages);

            // Weight tiles (jl, k) in order, four per DST acquire.
            uint32_t jl = 0, k = 0, k0 = 0, k1 = 0;
            channel_tiles(n0, s, k0, k1);
            k = k0;
            while (jl < nt) {
                uint32_t slot_j[4], slot_k[4], slots = 0;
                while (jl < nt && slots < 4) {
                    slot_j[slots] = jl;
                    slot_k[slots] = k;
                    ++slots;
                    if (++k == k1) {
                        if (++jl < nt) {
                            channel_tiles(n0 + jl, s, k0, k1);
                            k = k0;
                        }
                    }
                }
                tile_regs_acquire();
                for (uint32_t sl = 0; sl < slots; ++sl) {
                    for (uint32_t m = 0; m < count; ++m) {
                        matmul_tiles(
                            cb_gt,
                            cb_act,
                            m * s.nt_max + slot_j[sl],
                            m * s.kpages + (slot_k[sl] - kc0),
                            sl
                        );
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t sl = 0; sl < slots; ++sl) {
                    cb_reserve_back(cb_out, 1);
                    pack_tile(sl, cb_out);
                    cb_push_back(cb_out, 1);
                }
                tile_regs_release();
            }
            cb_pop_front(cb_act, per_chunk * s.kpages);
        }
        cb_pop_front(cb_gt, per_chunk * s.nt_max);
    }
}
