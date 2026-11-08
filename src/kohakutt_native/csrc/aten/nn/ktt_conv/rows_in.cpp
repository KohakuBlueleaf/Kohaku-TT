// Conv input layout: NCHW TILE bf16 x -> zero-padded channels-last rows [N Hp Wp, C]
// (one page per pixel). Per unit (image, row tile, column tile, group of 32 channels,
// quarter of the tile's rows): the 32 channel tiles into L1, the quarter's pixels
// regrouped, each pixel's 64-byte channel segment written into its row. Border rows /
// columns are written as zeros.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    constexpr uint32_t kTileBytes = 2048;

    // Element offset of (r, c) in a 32x32 tile stored as four 16x16 faces.
    inline uint32_t face_offset(uint32_t r, uint32_t c) {
        return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of x, then of the rows. Runtime args: x
// address, rows address, first unit, unit step, N, C, H, W, Hp, Wp, pad top, pad left,
// pixels per rows page (P).
void kernel_main() {
    uint32_t a = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(a++);
    const uint32_t rows_addr = get_arg_val<uint32_t>(a++);
    const uint32_t start = get_arg_val<uint32_t>(a++);
    const uint32_t step = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++);
    const uint32_t c = get_arg_val<uint32_t>(a++);
    const uint32_t h = get_arg_val<uint32_t>(a++);
    const uint32_t w = get_arg_val<uint32_t>(a++);
    const uint32_t hp = get_arg_val<uint32_t>(a++);
    const uint32_t wp = get_arg_val<uint32_t>(a++);
    const uint32_t ph = get_arg_val<uint32_t>(a++);
    const uint32_t pw = get_arg_val<uint32_t>(a++);
    const uint32_t per_page = get_arg_val<uint32_t>(a++);

    constexpr uint32_t cb_tiles = 0, cb_stage = 1, cb_zero = 2;
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto rows_args = (
        TensorAccessorArgs<x_args.next_compile_time_args_offset()>()
    );
    const uint32_t row_bytes = c * 2;
    const auto x = TensorAccessor(x_args, x_addr, kTileBytes);
    const auto rows = TensorAccessor(rows_args, rows_addr, per_page * row_bytes);
    // NOC address of byte `offset` of pixel row `pixel`.
    const auto pixel_addr = [&](uint32_t pixel, uint32_t offset) {
        return (
            rows.get_noc_addr(pixel / per_page, (pixel % per_page) * row_bytes + offset)
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

    const uint32_t ht = (h + 31) / 32, wt = (w + 31) / 32, groups = c / 32;
    const uint32_t units = n * ht * wt * groups * 4;
    for (uint32_t u = start; u < units; u += step) {
        const uint32_t q = u % 4, g = (u / 4) % groups;
        const uint32_t tile = u / 4 / groups;
        const uint32_t img = tile / (ht * wt), th = (tile / wt) % ht, tw = tile % wt;
        const uint32_t rv = h - th * 32 < 32 ? h - th * 32 : 32;
        const uint32_t cv = w - tw * 32 < 32 ? w - tw * 32 : 32;
        const uint32_t r0 = q * 8, r1 = r0 + 8 < rv ? r0 + 8 : rv;
        if (r0 >= r1) {
            continue;
        }
        // The quarter's 8 face rows (256 bytes) in each of the two faces of its half.
        const uint32_t face = (r0 / 16) * 2, frow = r0 % 16;
        for (uint32_t k = 0; k < 32; ++k) {
            const uint32_t page = ((img * c + g * 32 + k) * ht + th) * wt + tw;
            for (uint32_t f = face; f < face + 2; ++f) {
                const uint32_t offset = (f * 256 + frow * 16) * 2;
                noc_async_read(
                    x.get_noc_addr(page, offset), tiles + k * kTileBytes + offset, 256
                );
            }
        }
        noc_async_read_barrier();
        // stage[pixel][k]: pixel = r * 32 + col, 32 channels (64 bytes) each; 2 pixels
        // x 2 channels per pair of 32-bit loads and stores.
        auto* stage32 = reinterpret_cast<uint32_t*>(stage);
        for (uint32_t k = 0; k < 32; k += 2) {
            const auto* t0 = reinterpret_cast<const uint16_t*>(tiles + k * kTileBytes);
            const auto* t1 = t0 + kTileBytes / 2;
            for (uint32_t r = r0; r < r1; ++r) {
                for (uint32_t col = 0; col < cv; col += 2) {
                    const uint32_t f = face_offset(r, col);
                    const uint32_t a = *reinterpret_cast<const uint32_t*>(t0 + f);
                    const uint32_t b = *reinterpret_cast<const uint32_t*>(t1 + f);
                    const uint32_t s = ((r * 32 + col) * 32 + k) / 2;
                    stage32[s] = (a & 0xFFFFu) | (b << 16);
                    stage32[s + 16] = (a >> 16) | (b & 0xFFFF0000u);
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
