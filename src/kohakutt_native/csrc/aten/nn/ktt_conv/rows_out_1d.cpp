// 1-D conv output layout: TILE bf16 [rows, O] (pixel rows x channels, see OutRows in
// program.hpp with OH = 1) -> NCL TILE [N, O, OL]. Per unit (image, channel tile,
// length tile): per column phase the one or two pixel-row tiles holding its positions,
// transposed into the output tile (padding zero), written.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "args.hpp"

namespace {

    constexpr uint32_t kTileBytes = 2048;
    constexpr uint32_t kMaxPhases = 8;

    // Element offset of (r, c) in a 32x32 tile stored as four 16x16 faces.
    inline uint32_t face_offset(uint32_t r, uint32_t c) {
        return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of the conv output, then of the NCL output,
// then the argument-layout version. Runtime args (per core: start; the rest common):
// version, conv output address, NCL output address, first unit, unit step, N, O, OL,
// pitch, Sw, rows per phase.
void kernel_main() {
    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr auto
        dst_args = (TensorAccessorArgs<src_args.next_compile_time_args_offset()>());
    if (get_compile_time_arg_val(dst_args.next_compile_time_args_offset()) !=
            ktt_conv::kArgsVersion ||
        get_common_arg_val<uint32_t>(0) != ktt_conv::kArgsVersion) {
        return;
    }
    uint32_t a = 1;
    const uint32_t src_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t dst_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t step = get_common_arg_val<uint32_t>(a++);
    const uint32_t n = get_common_arg_val<uint32_t>(a++);
    const uint32_t o = get_common_arg_val<uint32_t>(a++);
    const uint32_t ol = get_common_arg_val<uint32_t>(a++);
    const uint32_t pitch = get_common_arg_val<uint32_t>(a++);
    const uint32_t sw = get_common_arg_val<uint32_t>(a++);
    const uint32_t phase_rows = get_common_arg_val<uint32_t>(a++);

    constexpr uint32_t cb_src = 0, cb_out = 1;
    const auto src = TensorAccessor(src_args, src_addr, kTileBytes);
    const auto dst = TensorAccessor(dst_args, dst_addr, kTileBytes);

    // Scratch: source tiles (Sw phases x 2), the output tile.
    cb_reserve_back(cb_src, 1);
    cb_reserve_back(cb_out, 1);
    const uint32_t src_l1 = get_write_ptr(cb_src);
    const uint32_t out_l1 = get_write_ptr(cb_out);
    auto* out = reinterpret_cast<uint16_t*>(out_l1);

    const auto row_of = [&](uint32_t img, uint32_t x) {
        return (x % sw) * phase_rows + img * pitch + x / sw;
    };

    const uint32_t ot = (o + 31) / 32, lt = (ol + 31) / 32;
    const uint32_t units = n * ot * lt;
    uint32_t first[kMaxPhases];
    for (uint32_t u = start; u < units; u += step) {
        const uint32_t img = u / (ot * lt), g = (u / lt) % ot, tl = u % lt;
        const uint32_t x0 = tl * 32;
        const uint32_t cols = ol - x0 < 32 ? ol - x0 : 32;
        const uint32_t channels = o - g * 32 < 32 ? o - g * 32 : 32;
        for (uint32_t f = 0; f < sw && f < cols; ++f) {
            const uint32_t lo = row_of(img, x0 + f);
            const uint32_t hi = row_of(img, x0 + f + (cols - 1 - f) / sw * sw);
            const uint32_t slot = src_l1 + f * 2 * kTileBytes;
            first[f] = lo / 32;
            noc_async_read_page((lo / 32) * ot + g, src, slot);
            if (hi / 32 != lo / 32) {
                noc_async_read_page((hi / 32) * ot + g, src, slot + kTileBytes);
            }
        }
        if (cols < 32 || channels < 32) {
            for (uint32_t e = 0; e < kTileBytes / 4; ++e) {
                reinterpret_cast<uint32_t*>(out_l1)[e] = 0;
            }
        }
        noc_async_read_barrier();
        // Position x0 + l: phase xf, quotient xq; f = l mod Sw (its source slot).
        uint32_t xf = x0 % sw, xq = x0 / sw, f = 0;
        for (uint32_t l = 0; l < cols; ++l) {
            const uint32_t p = xf * phase_rows + img * pitch + xq;
            const auto* t = reinterpret_cast<const uint16_t*>(
                src_l1 + f * 2 * kTileBytes + ((p >> 5) == first[f] ? 0 : kTileBytes)
            );
            const uint32_t r = p & 31;
            for (uint32_t k = 0; k < channels; ++k) {
                out[face_offset(k, l)] = t[face_offset(r, k)];
            }
            if (++xf == sw) {
                xf = 0;
                ++xq;
            }
            if (++f == sw) {
                f = 0;
            }
        }
        noc_async_write_page((img * ot + g) * lt + tl, dst, out_l1);
        noc_async_write_barrier();
    }
}
