// Sparse conv weights from the torch weight W = TILE [O, Cg, KH, KW] bf16 (KH, KW <= 32:
// one tile per (o, q)): per unit (output column tile j, kernel row i, channel tile k),
// the KW output tiles (j, t = i KW + jj, k) of TILE [Ot taps Kmax 32, 32] (page
// (j taps + t) Kmax + k). Entry (rr, cc) of tap t holds W[32 j + cc, q, i, jj] for input
// channel c = (k0(j) + k) 32 + rr of output channel 32 j + cc's group (q = c - group
// Cg), zero elsewhere and for k >= k1(j) - k0(j); each entry's kernel row is read once.
// Transposed mode (the input-gradient gather of a forward conv with weight W): entry
// (rr, cc) holds W[c, oc mod Og, i, jj] for c = (k0(j) + k) 32 + rr, oc = 32 j + cc
// (this conv's C, O, Cg, Og are the forward's O, C, Og, Cg). 1-D mode: W = TILE [O, Cg,
// KW] (KH = 1), entry (oc, q) at tile row q mod 32 of page oc ceil(Cg / 32) + q / 32.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "../ktt_conv/args.hpp"

namespace {

    constexpr uint32_t kTileBytes = 2048;

    // Element offset of (r, c) in a 32x32 tile stored as four 16x16 faces.
    inline uint32_t face_offset(uint32_t r, uint32_t c) {
        return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
    }

    // Input channel tiles [k0, k1) of output column tile j (groups of cg input and og
    // output channels, o outputs).
    inline void channel_tiles(
        uint32_t j, uint32_t cg, uint32_t og, uint32_t o, uint32_t& k0, uint32_t& k1
    ) {
        const uint32_t lo = 32 * j, hi = 32 * j + 31 < o ? 32 * j + 31 : o - 1;
        k0 = (lo / og) * cg / 32;
        k1 = ((hi / og + 1) * cg - 1) / 32 + 1;
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of W, then of the output, then the
// argument-layout version. Runtime args (per core: first unit; the rest common):
// version, W address, output address, unit step, Ot, KH, KW, Kmax, Cg, Og, O, C,
// transposed (0 / 1), 1-D (0 / 1). Scratch: CB 0 holds 1024 entries' row segments (the
// 64-byte aligned block of each 32-byte face row segment, two when KW > 16), CB 1 KW
// output tiles.
void kernel_main() {
    constexpr auto w_args = TensorAccessorArgs<0>();
    constexpr auto out_args = (
        TensorAccessorArgs<w_args.next_compile_time_args_offset()>()
    );
    if (get_compile_time_arg_val(out_args.next_compile_time_args_offset()) !=
            ktt_conv::kArgsVersion ||
        get_common_arg_val<uint32_t>(0) != ktt_conv::kArgsVersion) {
        return;
    }
    uint32_t a = 1;
    const uint32_t w_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t step = get_common_arg_val<uint32_t>(a++);
    const uint32_t ot = get_common_arg_val<uint32_t>(a++);
    const uint32_t kh = get_common_arg_val<uint32_t>(a++);
    const uint32_t kw = get_common_arg_val<uint32_t>(a++);
    const uint32_t kmax = get_common_arg_val<uint32_t>(a++);
    const uint32_t cg = get_common_arg_val<uint32_t>(a++);
    const uint32_t og = get_common_arg_val<uint32_t>(a++);
    const uint32_t o = get_common_arg_val<uint32_t>(a++);
    const uint32_t c = get_common_arg_val<uint32_t>(a++);
    const uint32_t transposed = get_common_arg_val<uint32_t>(a++);
    const uint32_t one_d = get_common_arg_val<uint32_t>(a++);
    const uint32_t start = get_arg_val<uint32_t>(0);
    // Page and tile row of weight entry (first index e0, second index e1 of n1) at
    // kernel row i.
    const auto locate = [&](uint32_t e0, uint32_t e1, uint32_t n1, uint32_t i,
                            uint32_t& page, uint32_t& row) {
        if (one_d) {
            page = e0 * ((n1 + 31) / 32) + e1 / 32;
            row = e1 % 32;
        } else {
            page = e0 * n1 + e1;
            row = i;
        }
    };

    constexpr uint32_t cb_rows = 0, cb_out = 1;
    const auto w = TensorAccessor(w_args, w_addr, kTileBytes);
    const auto out = TensorAccessor(out_args, out_addr, kTileBytes);

    cb_reserve_back(cb_rows, 1);
    cb_reserve_back(cb_out, 1);
    const uint32_t rows_l1 = (get_write_ptr(cb_rows) + 63) & ~63u;
    const uint32_t out_l1 = get_write_ptr(cb_out);
    const uint32_t halves = kw > 16 ? 2 : 1;
    const uint32_t taps = kh * kw;

    const uint32_t units = ot * kh * kmax;
    for (uint32_t u = start; u < units; u += step) {
        const uint32_t j = u / (kh * kmax), i = (u / kmax) % kh, k = u % kmax;
        auto* tiles = reinterpret_cast<uint32_t*>(out_l1);
        for (uint32_t e = 0; e < kw * kTileBytes / 4; ++e) {
            tiles[e] = 0;
        }
        uint32_t k0, k1;
        channel_tiles(j, cg, og, o, k0, k1);
        if (k < k1 - k0) {
            // Kernel row i of each entry's (o, q) tile: its face row segment(s).
            for (uint32_t rr = 0; rr < 32; ++rr) {
                const uint32_t ch = (k0 + k) * 32 + rr;
                for (uint32_t cc = 0; cc < 32; ++cc) {
                    const uint32_t oc = 32 * j + cc;
                    if (ch >= c || oc >= o || ch / cg != oc / og) {
                        continue;
                    }
                    uint32_t page, row;
                    if (transposed) {
                        locate(ch, oc % og, og, i, page, row);
                    } else {
                        locate(oc, ch - (oc / og) * cg, cg, i, page, row);
                    }
                    for (uint32_t hf = 0; hf < halves; ++hf) {
                        const uint32_t offset = face_offset(row, hf * 16) * 2;
                        noc_async_read(
                            w.get_noc_addr(page, offset & ~63u),
                            rows_l1 + ((rr * 32 + cc) * halves + hf) * 64,
                            64
                        );
                    }
                }
            }
            noc_async_read_barrier();
            const auto* rows = reinterpret_cast<const uint16_t*>(rows_l1);
            auto* dst = reinterpret_cast<uint16_t*>(out_l1);
            for (uint32_t rr = 0; rr < 32; ++rr) {
                const uint32_t ch = (k0 + k) * 32 + rr;
                for (uint32_t cc = 0; cc < 32; ++cc) {
                    const uint32_t oc = 32 * j + cc;
                    if (ch >= c || oc >= o || ch / cg != oc / og) {
                        continue;
                    }
                    uint32_t page, row;
                    if (transposed) {
                        locate(ch, oc % og, og, i, page, row);
                    } else {
                        locate(oc, ch - (oc / og) * cg, cg, i, page, row);
                    }
                    // Element offset of the entry's row segment in its 64-byte block.
                    const uint32_t within = (face_offset(row, 0) * 2 % 64) / 2;
                    const uint16_t* seg = rows + (rr * 32 + cc) * halves * 32 + within;
                    for (uint32_t jj = 0; jj < kw; ++jj) {
                        dst[jj * (kTileBytes / 2) + face_offset(rr, cc)] =
                            seg[(jj / 16) * 32 + jj % 16];
                    }
                }
            }
        }
        for (uint32_t jj = 0; jj < kw; ++jj) {
            noc_async_write_page(
                (j * taps + i * kw + jj) * kmax + k, out, out_l1 + jj * kTileBytes
            );
        }
        noc_async_write_barrier();
    }
}
