// Gather conv writer: per work item and output row tile m, the item's nt column tiles
// (CB 16, bf16, or fp32 partial sums of a split gather) to the output TILE at tiles
// (mb + m, n0 .. n0 + nt - 1); row tiles whose phase has no taps in the item's tap rows
// are written as zeros. Packed weights (Cg < 32): per item first the
// item's weight tiles (CB 1): zeroed from a zero tile (CB 5), then each column tile's
// packed row (CB 6) read and its entries written into the tiles (the reader streams
// input rows meanwhile).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "taps.hpp"

// Compile-time args: TensorAccessorArgs of the output, then of the weights, then the
// argument-layout version. Common runtime args: version, output address, weights
// address, packed (0 / 1), then the shape (taps.hpp). Per core: items, then an Item
// (taps.hpp) per item.
void kernel_main() {
    using namespace ktt_conv2;
    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr auto
        w_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    if (!args_match<w_args.next_compile_time_args_offset()>()) {
        return;
    }
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t packed = get_common_arg_val<uint32_t>(3);
    const Shape s = read_shape(4);
    constexpr uint32_t cb_w = 1, cb_zero = 4, cb_wzero = 5, cb_pack = 6, cb_out = 16;
    const uint32_t tile = get_tile_size(cb_out);
    const uint32_t wtile = get_tile_size(
        cb_w
    );  // bf16; the output may be fp32 partials
    const auto out = TensorAccessor(out_args, out_addr, tile);
    const uint32_t taps = s.kh * s.kw;
    const auto w = TensorAccessor(w_args, w_addr, taps * 32 * s.cg * 2);

    cb_reserve_back(cb_zero, 1);
    const uint32_t zero = get_write_ptr(cb_zero);
    bool zeroed = false;
    const auto zero_tile = [&] {
        if (!zeroed) {
            for (uint32_t e = 0; e < tile / 4; ++e) {
                reinterpret_cast<volatile uint32_t*>(zero)[e] = 0;
            }
            zeroed = true;
        }
    };
    uint32_t wzero = 0, pack = 0;
    if (packed) {
        cb_reserve_back(cb_wzero, 1);
        cb_reserve_back(cb_pack, 1);
        wzero = get_write_ptr(cb_wzero);
        pack = get_write_ptr(cb_pack);
        for (uint32_t e = 0; e < wtile / 4; ++e) {
            reinterpret_cast<volatile uint32_t*>(wzero)[e] = 0;
        }
    }

    const uint32_t items = get_arg_val<uint32_t>(0);
    for (uint32_t it = 0; it < items; ++it) {
        const Item item = read_item(it);
        const uint32_t m0 = item.m0, m1 = item.m1, n0 = item.n0, nt = item.nt;
        if (packed) {
            // Weight tile (jl, t, k) at CB position (jl taps + t) Kmax + k; only the
            // taps [t0, t1) of the item's tap rows (a split item: kernel rows a0 .. a1
            // - 1).
            const uint32_t stride = s.kmax * wtile;
            const bool rows = item.a0 > 0 || item.a1 < s.rows_h;
            const uint32_t t0 = rows ? item.a0 * s.kw : 0;
            const uint32_t t1 = rows ? item.a1 * s.kw : taps;
            cb_reserve_back(cb_w, s.nt_max * taps * s.kmax);
            const uint32_t w_base = get_write_ptr(cb_w);
            const uint64_t zero_noc = get_noc_addr(wzero);
            for (uint32_t jl = 0; jl < nt; ++jl) {
                for (
                    uint32_t p = (jl * taps + t0) * s.kmax;
                    p < (jl * taps + t1) * s.kmax;
                    ++p
                ) {
                    noc_async_read(zero_noc, w_base + p * wtile, wtile);
                }
            }
            for (uint32_t jl = 0; jl < nt; ++jl) {
                const uint32_t j = n0 + jl;
                uint32_t k0, k1;
                channel_tiles(j, s, k0, k1);
                noc_async_read_page(j, w, pack);
                noc_async_read_barrier();
                // Entry (t, cc, q): output channel 32 j + cc, input channel group Cg +
                // q.
                const auto* row = reinterpret_cast<const uint16_t*>(pack);
                // Output channel 32 j + cc is in group grp, rem channels in.
                uint32_t grp = (32 * j) / s.og, rem = (32 * j) % s.og;
                for (uint32_t cc = 0; cc < 32 && 32 * j + cc < s.o; ++cc) {
                    const uint32_t first = grp * s.cg;
                    if (++rem == s.og) {
                        rem = 0;
                        ++grp;
                    }
                    for (uint32_t q = 0; q < s.cg; ++q) {
                        const uint32_t ch = first + q;
                        auto* dst = reinterpret_cast<uint16_t*>(
                                        w_base +
                                        ((jl * taps + t0) * s.kmax + ch / 32 - k0) *
                                            wtile
                                    ) +
                            face_offset(ch % 32, cc);
                        const uint16_t* src = row + (t0 * 32 + cc) * s.cg + q;
                        for (uint32_t t = t0; t < t1; ++t) {
                            *dst = *src;
                            dst += stride / 2;
                            src += 32 * s.cg;
                        }
                    }
                }
            }
            cb_push_back(cb_w, s.nt_max * taps * s.kmax);
        }
        for (uint32_t m = m0; m < m1; ++m) {
            if (phase_taps(s, m, item.a0, item.a1) == 0) {
                zero_tile();
                for (uint32_t j = 0; j < nt; ++j) {
                    noc_async_write_page((item.mb + m) * s.ot + n0 + j, out, zero);
                }
                noc_async_write_barrier();
                continue;
            }
            cb_wait_front(cb_out, s.nt_max);
            uint32_t src = get_read_ptr(cb_out);
            for (uint32_t j = 0; j < nt; ++j) {
                noc_async_write_page((item.mb + m) * s.ot + n0 + j, out, src);
                src += tile;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, s.nt_max);
        }
    }
}
