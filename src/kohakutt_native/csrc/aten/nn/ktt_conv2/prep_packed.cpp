// Packed conv weights (groups of Cg < 32 input channels) from the torch weight W = TILE
// [O, Cg, KH, KW] bf16 (or [O, Cg, KW] in 1-D mode, see prep.cpp): ROW_MAJOR [Ot, taps
// 32 Cg] bf16, row j = per tap t, per column cc of output tile j, the Cg weights
// W[32 j + cc, q, i, jj] (t = i KW + jj; zero past O). Transposed mode as in prep.cpp:
// W[c, oc mod Og, i, jj] for this conv's input channel c = group(oc) Cg + q. Per unit
// (j, kernel row i): each entry's kernel row read once, the row's KW taps assembled,
// written into row j.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "../ktt_conv/args.hpp"

namespace {

    constexpr uint32_t kTileBytes = 2048;

    // Element offset of (r, c) in a 32x32 tile stored as four 16x16 faces.
    inline uint32_t face_offset(uint32_t r, uint32_t c) {
        return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of W, then of the output, then the
// argument-layout version. Runtime args (per core: first unit; the rest common):
// version, W address, output address, unit step, Ot, KH, KW, Cg, Og, O, transposed
// (0 / 1), 1-D (0 / 1). Scratch: CB 0 holds 32 Cg entries' row segments (64-byte blocks,
// two per entry when KW > 16), CB 1 the KW taps' 32 Cg values.
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
    const uint32_t cg = get_common_arg_val<uint32_t>(a++);
    const uint32_t og = get_common_arg_val<uint32_t>(a++);
    const uint32_t o = get_common_arg_val<uint32_t>(a++);
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
    const uint32_t taps = kh * kw;
    const uint32_t entries = 32 * cg;
    const auto w = TensorAccessor(w_args, w_addr, kTileBytes);
    const auto out = TensorAccessor(out_args, out_addr, taps * entries * 2);

    cb_reserve_back(cb_rows, 1);
    cb_reserve_back(cb_out, 1);
    const uint32_t rows_l1 = (get_write_ptr(cb_rows) + 63) & ~63u;
    const uint32_t out_l1 = get_write_ptr(cb_out);
    const uint32_t halves = kw > 16 ? 2 : 1;

    const uint32_t units = ot * kh;
    for (uint32_t u = start; u < units; u += step) {
        const uint32_t j = u / kh, i = u % kh;
        // Kernel row i of each entry (cc, q): its face row segment(s).
        for (uint32_t e = 0; e < entries; ++e) {
            const uint32_t cc = e / cg, q = e % cg, oc = 32 * j + cc;
            if (oc >= o) {
                continue;
            }
            const uint32_t ch = (oc / og) * cg + q;
            uint32_t page, row;
            if (transposed) {
                locate(ch, oc % og, og, i, page, row);
            } else {
                locate(oc, q, cg, i, page, row);
            }
            for (uint32_t hf = 0; hf < halves; ++hf) {
                const uint32_t offset = face_offset(row, hf * 16) * 2;
                noc_async_read(
                    w.get_noc_addr(page, offset & ~63u),
                    rows_l1 + (e * halves + hf) * 64,
                    64
                );
            }
        }
        noc_async_read_barrier();
        const auto* rows = reinterpret_cast<const uint16_t*>(rows_l1);
        auto* dst = reinterpret_cast<uint16_t*>(out_l1);
        for (uint32_t e = 0; e < entries; ++e) {
            const uint32_t cc = e / cg, q = e % cg, oc = 32 * j + cc;
            if (oc >= o) {
                for (uint32_t jj = 0; jj < kw; ++jj) {
                    dst[jj * entries + e] = 0;
                }
                continue;
            }
            const uint32_t ch = (oc / og) * cg + q;
            uint32_t page, row;
            if (transposed) {
                locate(ch, oc % og, og, i, page, row);
            } else {
                locate(oc, q, cg, i, page, row);
            }
            // Element offset of the entry's row segment in its 64-byte block.
            const uint32_t within = (face_offset(row, 0) * 2 % 64) / 2;
            const uint16_t* seg = rows + e * halves * 32 + within;
            for (uint32_t jj = 0; jj < kw; ++jj) {
                dst[jj * entries + e] = seg[(jj / 16) * 32 + jj % 16];
            }
        }
        // Row j, taps i KW .. i KW + KW - 1.
        noc_async_write(out_l1, out.get_noc_addr(j, i * kw * entries * 2), kw * entries * 2);
        noc_async_write_barrier();
    }
}
