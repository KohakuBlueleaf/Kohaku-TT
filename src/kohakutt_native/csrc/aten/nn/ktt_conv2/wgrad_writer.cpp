// Weight-gradient writer: per unit (column group from n0, chunk sc), tap t and weight
// tile (jl, k) in the compute kernel's order, its fp32 partial tile (CB 16) to page
// sc B + (j taps + t) Kmax + (k - k0) of the partials [S B, 1024] fp32 (B = tiles).
// Packed (Cg < 32): per (jl, t) the 32 Cg entries (cc, q) of output channel 32 j + cc
// and input channel group Cg + q gathered from its tiles (CB 6) and written at flat
// element (j taps + t) 32 Cg of chunk sc's B pages.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "taps.hpp"

// Compile-time args: TensorAccessorArgs of the partials, then the argument-layout
// version. Common runtime args: version, partials address, unit step, units, chunks,
// row tiles per chunk, row tiles, Ot, Kmax, Cg, Og, O, KH, KW, K, nt, packed (0 / 1),
// pages per chunk B. Per core: first unit.
void kernel_main() {
    using namespace ktt_conv2;
    constexpr auto out_args = TensorAccessorArgs<0>();
    if (!args_match<out_args.next_compile_time_args_offset()>()) {
        return;
    }
    uint32_t a = 1;
    const uint32_t out_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t step = get_common_arg_val<uint32_t>(a++);
    const uint32_t units = get_common_arg_val<uint32_t>(a++);
    const uint32_t chunks = get_common_arg_val<uint32_t>(a++);
    a += 2;  // row tiles per chunk, row tiles
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
    const uint32_t packed = get_common_arg_val<uint32_t>(a++);
    const uint32_t pages = get_common_arg_val<uint32_t>(a++);
    const uint32_t start = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_out = 16, cb_pack = 6;
    const uint32_t entries = 32 * s.cg;
    const auto out = TensorAccessor(out_args, out_addr, 4096);
    const uint32_t taps = s.kh * s.kw;
    uint32_t pack = 0;
    if (packed) {
        cb_reserve_back(cb_pack, 1);
        pack = get_write_ptr(cb_pack);
    }

    for (uint32_t u = start; u < units; u += step) {
        const uint32_t n0 = (u / chunks) * s.nt_max, sc = u % chunks;
        const uint32_t nt = s.ot - n0 < s.nt_max ? s.ot - n0 : s.nt_max;
        for (uint32_t t = 0; t < taps; ++t) {
            for (uint32_t jl = 0; jl < nt; ++jl) {
                const uint32_t j = n0 + jl;
                uint32_t k0, k1;
                channel_tiles(j, s, k0, k1);
                for (uint32_t k = k0; k < k1; ++k) {
                    cb_wait_front(cb_out, 1);
                    const uint32_t src = get_read_ptr(cb_out);
                    if (!packed) {
                        noc_async_write_page(
                            sc * pages + (j * taps + t) * s.kmax + (k - k0), out, src
                        );
                        noc_async_write_barrier();
                        cb_pop_front(cb_out, 1);
                        continue;
                    }
                    // Tile k holds dW^T[cc][ch mod 32] for the input channels of tile
                    // k.
                    const auto* part = reinterpret_cast<const float*>(src);
                    auto* dst = reinterpret_cast<float*>(pack);
                    // Output channel 32 j + cc is in group grp, rem channels in.
                    uint32_t grp = (32 * j) / s.og, rem = (32 * j) % s.og;
                    for (uint32_t cc = 0; cc < 32; ++cc) {
                        const uint32_t first = grp * s.cg;
                        if (++rem == s.og) {
                            rem = 0;
                            ++grp;
                        }
                        for (uint32_t q = 0; q < s.cg; ++q) {
                            const uint32_t ch = first + q;
                            if (ch / 32 == k) {
                                dst[cc * s.cg + q] = part[face_offset(cc, ch % 32)];
                            }
                        }
                    }
                    cb_pop_front(cb_out, 1);
                }
                if (packed) {
                    // The row in 4 KB pages: split where it crosses a page.
                    uint32_t e = (j * taps + t) * entries, done = 0;
                    while (done < entries) {
                        const uint32_t room = 1024 - e % 1024;
                        const uint32_t n = entries - done < room ? entries - done
                                                                 : room;
                        noc_async_write(
                            pack + done * 4,
                            out.get_noc_addr(sc * pages + e / 1024, (e % 1024) * 4),
                            n * 4
                        );
                        e += n;
                        done += n;
                    }
                    noc_async_write_barrier();
                }
            }
        }
    }
}
