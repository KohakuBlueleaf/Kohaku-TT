// Conv input layout: NCHW TILE bf16 x -> zero-padded channels-last rows [N Hp Wp, Cp]
// (Cp: C rounded up to 32, channels from C on zero). Per unit (image, row tile, column
// tile, group of 32 channels, R of the tile's rows): the 32 channel tiles' R rows into
// L1, the pixels regrouped, each pixel's 64-byte channel segment written into its row.
// Border rows / columns (top / left: the padding; bottom / right: the rest of Hp x Wp)
// are written as zeros.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "args.hpp"

namespace {

    constexpr uint32_t kTileBytes = 2048;

    // Element offset of (r, c) in a 32x32 tile stored as four 16x16 faces.
    inline uint32_t face_offset(uint32_t r, uint32_t c) {
        return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of x, then of the rows, then the argument-layout
// version. Runtime args (per core: first unit; the rest common): version, x address, rows
// address, unit step, N, C, Cp, H, W, Hp, Wp, pad top, pad left, pixels per rows page (P,
// a power of two), tile rows per unit (R: 2, 4 or 8).
void kernel_main() {
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto rows_args = (
        TensorAccessorArgs<x_args.next_compile_time_args_offset()>()
    );
    if (get_compile_time_arg_val(rows_args.next_compile_time_args_offset()) !=
            ktt_conv::kArgsVersion ||
        get_common_arg_val<uint32_t>(0) != ktt_conv::kArgsVersion) {
        return;
    }
    uint32_t a = 1;
    const uint32_t x_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t rows_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t step = get_common_arg_val<uint32_t>(a++);
    const uint32_t n = get_common_arg_val<uint32_t>(a++);
    const uint32_t c = get_common_arg_val<uint32_t>(a++);
    const uint32_t cp = get_common_arg_val<uint32_t>(a++);
    const uint32_t h = get_common_arg_val<uint32_t>(a++);
    const uint32_t w = get_common_arg_val<uint32_t>(a++);
    const uint32_t hp = get_common_arg_val<uint32_t>(a++);
    const uint32_t wp = get_common_arg_val<uint32_t>(a++);
    const uint32_t ph = get_common_arg_val<uint32_t>(a++);
    const uint32_t pw = get_common_arg_val<uint32_t>(a++);
    const uint32_t per_page = get_common_arg_val<uint32_t>(a++);
    const uint32_t rpu = get_common_arg_val<uint32_t>(a++);
    const uint32_t parts = 32 / rpu;

    constexpr uint32_t cb_tiles = 0, cb_stage = 1, cb_zero = 2;
    const uint32_t row_bytes = cp * 2;
    const auto x = TensorAccessor(x_args, x_addr, kTileBytes);
    const auto rows = TensorAccessor(rows_args, rows_addr, per_page * row_bytes);
    // NOC address of byte `offset` of pixel row `pixel`.
    const uint32_t page_shift = __builtin_ctz(per_page), page_mask = per_page - 1;
    const auto pixel_addr = [&](uint32_t pixel, uint32_t offset) {
        return rows.get_noc_addr(
            pixel >> page_shift, (pixel & page_mask) * row_bytes + offset
        );
    };

    // Scratch buffers (one page each).
    cb_reserve_back(cb_tiles, 1);
    cb_reserve_back(cb_stage, 1);
    cb_reserve_back(cb_zero, 1);
    const uint32_t tiles = get_write_ptr(cb_tiles);
    auto* stage = reinterpret_cast<uint16_t*>(get_write_ptr(cb_stage));
    const uint32_t zero = get_write_ptr(cb_zero);
    for (uint32_t i = 0; i < row_bytes / 4; ++i) {
        reinterpret_cast<uint32_t*>(zero)[i] = 0;
    }

    const uint32_t ht = (h + 31) / 32, wt = (w + 31) / 32, groups = cp / 32;
    const uint32_t units = n * ht * wt * groups * parts;
    for (uint32_t u = start; u < units; u += step) {
        const uint32_t q = u % parts, g = (u / parts) % groups;
        const uint32_t tile = u / parts / groups;
        const uint32_t img = tile / (ht * wt), th = (tile / wt) % ht, tw = tile % wt;
        const uint32_t rv = h - th * 32 < 32 ? h - th * 32 : 32;
        const uint32_t cv = w - tw * 32 < 32 ? w - tw * 32 : 32;
        const uint32_t r0 = q * rpu, r1 = r0 + rpu < rv ? r0 + rpu : rv;
        if (r0 >= r1) {
            continue;
        }
        // The unit's R face rows (R 32 bytes) in each of the two faces of its half.
        const uint32_t face = (r0 / 16) * 2, frow = r0 % 16;
        for (uint32_t k = 0; k < 32; ++k) {
            const uint32_t page = ((img * c + g * 32 + k) * ht + th) * wt + tw;
            for (uint32_t f = face; f < face + 2; ++f) {
                const uint32_t offset = (f * 256 + frow * 16) * 2;
                if (g * 32 + k < c) {
                    noc_async_read(
                        x.get_noc_addr(page, offset),
                        tiles + k * kTileBytes + offset,
                        rpu * 32
                    );
                } else {
                    auto* z = reinterpret_cast<uint32_t*>(tiles + k * kTileBytes + offset);
                    for (uint32_t e = 0; e < rpu * 8; ++e) {
                        z[e] = 0;
                    }
                }
            }
        }
        noc_async_read_barrier();
        // stage[pixel][k]: pixel = r * 32 + col, 32 channels (64 bytes) each; 2 pixels
        // x 2 channels per pair of 32-bit loads and stores.
        auto* stage32 = reinterpret_cast<uint32_t*>(stage);
        constexpr uint32_t kTileWords = kTileBytes / 4;
        for (uint32_t r = r0; r < r1; ++r) {
            for (uint32_t col = 0; col < cv; col += 2) {
                // Channels k, k + 1 of pixels col, col + 1: one word of tiles k and k + 1.
                const auto* src = reinterpret_cast<const uint32_t*>(
                    reinterpret_cast<const uint16_t*>(tiles) + face_offset(r, col)
                );
                uint32_t* dst = stage32 + (r * 32 + col) * 16;
                for (uint32_t k = 0; k < 32; k += 2) {
                    const uint32_t va = src[0], vb = src[kTileWords];
                    dst[0] = (va & 0xFFFFu) | (vb << 16);
                    dst[16] = (va >> 16) | (vb & 0xFFFF0000u);
                    src += 2 * kTileWords;
                    ++dst;
                }
            }
        }
        for (uint32_t r = r0; r < r1; ++r) {
            const uint32_t base = (
                img * hp * wp + (th * 32 + r + ph) * wp + tw * 32 + pw
            );
            for (uint32_t col = 0; col < cv; ++col) {
                noc_async_write(
                    reinterpret_cast<uint32_t>(stage + (r * 32 + col) * 32),
                    pixel_addr(base + col, g * 64),
                    64
                );
            }
        }
        noc_async_write_barrier();
    }

    // Border: zero pages for padded rows and the padded columns of image rows.
    const uint32_t padded_rows = n * hp;
    for (uint32_t pr = start; pr < padded_rows; pr += step) {
        const uint32_t y = pr % hp;
        const uint32_t base = pr * wp;
        const bool full = y < ph || y >= h + ph;
        for (uint32_t col = 0; col < wp; ++col) {
            if (full || col < pw || col >= w + pw) {
                noc_async_write(zero, pixel_addr(base + col, 0), row_bytes);
            }
        }
        noc_async_write_barrier();
    }
}
