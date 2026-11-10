// Conv output layout: TILE bf16 [rows, O] (pixel rows x channels, see OutRows in
// program.hpp) -> NCHW TILE [N, O, OH, OW]. Per unit (image, output row tile, output
// column tile, group of 32 channels, R of the tile's rows): per output row and column
// phase the one or two pixel-row tiles holding its pixels, scattered into the group's
// 32 channel tiles (padding zero); the unit's face rows written.
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

// Compile-time args: TensorAccessorArgs of the conv output, then of the NCHW output,
// then the argument-layout version. Runtime args (per core: start; the rest common):
// version, conv output address, NCHW output address, first unit, unit step, N, O, OH,
// OW, pitch, OHq, Sh, Sw, rows per phase, tile rows per unit (R: 2, 4 or 8).
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
    const uint32_t oh = get_common_arg_val<uint32_t>(a++);
    const uint32_t ow = get_common_arg_val<uint32_t>(a++);
    const uint32_t pitch = get_common_arg_val<uint32_t>(a++);
    const uint32_t ohq = get_common_arg_val<uint32_t>(a++);
    const uint32_t sh = get_common_arg_val<uint32_t>(a++);
    const uint32_t sw = get_common_arg_val<uint32_t>(a++);
    const uint32_t phase_rows = get_common_arg_val<uint32_t>(a++);
    const uint32_t rpu = get_common_arg_val<uint32_t>(a++);
    const uint32_t parts = 32 / rpu;

    constexpr uint32_t cb_src = 0, cb_out = 1;
    const auto src = TensorAccessor(src_args, src_addr, kTileBytes);
    const auto dst = TensorAccessor(dst_args, dst_addr, kTileBytes);

    // Scratch buffers (one page each): source tiles (8 rows x Sw phases x 2), output.
    cb_reserve_back(cb_src, 1);
    cb_reserve_back(cb_out, 1);
    const uint32_t src_l1 = get_write_ptr(cb_src);
    const uint32_t out_l1 = get_write_ptr(cb_out);
    auto* out = reinterpret_cast<uint16_t*>(out_l1);

    // Source row of output pixel (img, y, x).
    const auto row_of = [&](uint32_t img, uint32_t y, uint32_t x) {
        const uint32_t phase = (y % sh) * sw + x % sw;
        return phase * phase_rows + (img * ohq + y / sh) * pitch + x / sw;
    };

    const uint32_t ot = (o + 31) / 32, oht = (oh + 31) / 32, owt = (ow + 31) / 32;
    const uint32_t units = n * oht * owt * ot * parts;
    uint32_t first[8][kMaxPhases];
    for (uint32_t u = start; u < units; u += step) {
        const uint32_t q = u % parts, g = (u / parts) % ot;
        const uint32_t tile = u / parts / ot;
        const uint32_t tw = tile % owt, th = (tile / owt) % oht, img = tile / owt / oht;
        const uint32_t rv = oh - th * 32 < 32 ? oh - th * 32 : 32;
        const uint32_t cv = ow - tw * 32 < 32 ? ow - tw * 32 : 32;
        const uint32_t r0 = q * rpu;
        // Rows r0 .. r0 + R - 1 of every channel tile: zero where partial, then the
        // valid pixels.
        if (cv < 32 || r0 + rpu > rv) {
            for (uint32_t k = 0; k < 32; ++k) {
                for (uint32_t r = r0; r < r0 + rpu; ++r) {
                    for (uint32_t col = 0; col < 32; col += 2) {
                        *reinterpret_cast<uint32_t*>(
                            out + k * (kTileBytes / 2) + face_offset(r, col)
                        ) = 0;
                    }
                }
            }
        }
        // Per row and column phase: the source tiles of its pixels (at most two).
        const uint32_t r_end = r0 + rpu < rv ? r0 + rpu : rv;
        const uint32_t x0 = tw * 32;
        for (uint32_t r = r0; r < r_end; ++r) {
            const uint32_t y = th * 32 + r;
            for (uint32_t f = 0; f < sw && f < cv; ++f) {
                const uint32_t lo = row_of(img, y, x0 + f);
                const uint32_t last = x0 + f + (cv - 1 - f) / sw * sw;
                const uint32_t hi = row_of(img, y, last);
                const uint32_t slot = src_l1 + ((r - r0) * sw + f) * 2 * kTileBytes;
                first[r - r0][f] = lo / 32;
                noc_async_read_page((lo / 32) * ot + g, src, slot);
                if (hi / 32 != lo / 32) {
                    noc_async_read_page((hi / 32) * ot + g, src, slot + kTileBytes);
                }
            }
        }
        noc_async_read_barrier();
        constexpr uint32_t kTileWords = kTileBytes / 4;
        // Pixel row p's 32 channels in a source tile: words of channels (0, 1) .. (14,
        // 15) at its face-0/1 row, (16, 17) .. (30, 31) 256 elements on.
        const auto channels_of = [&](uint32_t slot, uint32_t p) {
            return reinterpret_cast<const uint32_t*>(
                reinterpret_cast<const uint16_t*>(slot) + face_offset(p & 31, 0)
            );
        };
        for (uint32_t r = r0; r < r_end; ++r) {
            // Source rows of this output row: phase (y mod Sh, x mod Sw) base plus x /
            // Sw, walked incrementally along x.
            const uint32_t y = th * 32 + r;
            const uint32_t row_y = ((y % sh) * sw) * phase_rows +
                (img * ohq + y / sh) * pitch;
            const uint32_t slots = src_l1 + (r - r0) * sw * 2 * kTileBytes;
            // Column x0 + col: phase xf, quotient xq; cf = col mod Sw (its source
            // slot).
            uint32_t xf = x0 % sw, xq = x0 / sw, cf = 0;
            const auto source = [&]() {
                const uint32_t p = row_y + xf * phase_rows + xq;
                const uint32_t slot = slots + cf * 2 * kTileBytes;
                const auto* words = channels_of(
                    slot + ((p >> 5) == first[r - r0][cf] ? 0 : kTileBytes), p
                );
                if (++xf == sw) {
                    xf = 0;
                    ++xq;
                }
                if (++cf == sw) {
                    cf = 0;
                }
                return words;
            };
            // 2 pixels x 2 channels per pair of 32-bit loads and stores; past cv zero.
            for (uint32_t col = 0; col < cv; col += 2) {
                const auto* t0 = source();
                const bool pair = col + 1 < cv;
                const auto* t1 = pair ? source() : t0;
                auto* dst = reinterpret_cast<uint32_t*>(out + face_offset(r, col));
                for (uint32_t k = 0; k < 32; k += 2) {
                    const uint32_t at = k < 16 ? k / 2 : 128 + (k - 16) / 2;
                    const uint32_t va = t0[at], vb = pair ? t1[at] : 0;
                    dst[0] = (va & 0xFFFFu) | (vb << 16);
                    dst[kTileWords] = (va >> 16) | (vb & 0xFFFF0000u);
                    dst += 2 * kTileWords;
                }
            }
        }
        // The unit's rows: R face rows (R 32 bytes) in each of two faces per tile.
        const uint32_t face = (r0 / 16) * 2, row = r0 % 16;
        const uint32_t channels = o - g * 32 < 32 ? o - g * 32 : 32;
        for (uint32_t k = 0; k < channels; ++k) {
            const uint32_t page = ((img * o + g * 32 + k) * oht + th) * owt + tw;
            for (uint32_t fc = face; fc < face + 2; ++fc) {
                const uint32_t offset = (fc * 256 + row * 16) * 2;
                noc_async_write(
                    out_l1 + k * kTileBytes + offset,
                    dst.get_noc_addr(page, offset),
                    rpu * 32
                );
            }
        }
        noc_async_write_barrier();
    }
}
