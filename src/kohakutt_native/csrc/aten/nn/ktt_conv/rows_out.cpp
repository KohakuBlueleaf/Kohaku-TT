// Conv output layout: TILE bf16 [N OH Wp, O] (padded-width pixel rows x channels) ->
// NCHW TILE [N, O, OH, OW]. Per unit (image, output row tile, output column tile, group
// of 32 channels, quarter of the tile's rows): per output row the one or two pixel-row
// tiles holding its 32 pixels, scattered into the group's 32 channel tiles (padding
// zero); the quarter's face rows written.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    constexpr uint32_t kTileBytes = 2048;

    // Element offset of (r, c) in a 32x32 tile stored as four 16x16 faces.
    inline uint32_t face_offset(uint32_t r, uint32_t c) {
        return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of the conv output, then of the NCHW output.
// Runtime args: conv output address, NCHW output address, first unit, unit step, N, O,
// OH, OW, Wp.
void kernel_main() {
    uint32_t a = 0;
    const uint32_t src_addr = get_arg_val<uint32_t>(a++);
    const uint32_t dst_addr = get_arg_val<uint32_t>(a++);
    const uint32_t start = get_arg_val<uint32_t>(a++);
    const uint32_t step = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++);
    const uint32_t o = get_arg_val<uint32_t>(a++);
    const uint32_t oh = get_arg_val<uint32_t>(a++);
    const uint32_t ow = get_arg_val<uint32_t>(a++);
    const uint32_t wp = get_arg_val<uint32_t>(a++);

    constexpr uint32_t cb_src = 0, cb_out = 1;
    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr auto dst_args = (
        TensorAccessorArgs<src_args.next_compile_time_args_offset()>()
    );
    const auto src = TensorAccessor(src_args, src_addr, kTileBytes);
    const auto dst = TensorAccessor(dst_args, dst_addr, kTileBytes);

    // Scratch buffers (one page each).
    cb_reserve_back(cb_src, 1);
    cb_reserve_back(cb_out, 1);
    const uint32_t src_l1 = get_write_ptr(cb_src);
    const uint32_t out_l1 = get_write_ptr(cb_out);
    auto* out = reinterpret_cast<uint16_t*>(out_l1);

    const uint32_t ot = (o + 31) / 32, oht = (oh + 31) / 32, owt = (ow + 31) / 32;
    const uint32_t units = n * oht * owt * ot * 4;
    for (uint32_t u = start; u < units; u += step) {
        const uint32_t q = u % 4, g = (u / 4) % ot;
        const uint32_t tile = u / 4 / ot;
        const uint32_t tw = tile % owt, th = (tile / owt) % oht, img = tile / owt / oht;
        const uint32_t rv = oh - th * 32 < 32 ? oh - th * 32 : 32;
        const uint32_t cv = ow - tw * 32 < 32 ? ow - tw * 32 : 32;
        const uint32_t r0 = q * 8;
        // Rows r0 .. r0 + 7 of every channel tile: zero where partial, then the valid
        // pixels.
        if (cv < 32 || r0 + 8 > rv) {
            for (uint32_t k = 0; k < 32; ++k) {
                for (uint32_t r = r0; r < r0 + 8; ++r) {
                    for (uint32_t col = 0; col < 32; col += 2) {
                        *reinterpret_cast<uint32_t*>(
                            out + k * (kTileBytes / 2) + face_offset(r, col)
                        ) = 0;
                    }
                }
            }
        }
        // Source tiles of the quarter's rows (two per row) read together.
        const uint32_t r_end = r0 + 8 < rv ? r0 + 8 : rv;
        for (uint32_t r = r0; r < r_end; ++r) {
            const uint32_t p0 = img * oh * wp + (th * 32 + r) * wp + tw * 32;
            const uint32_t m0 = p0 / 32, m1 = (p0 + cv - 1) / 32;
            const uint32_t slot = src_l1 + (r - r0) * 2 * kTileBytes;
            noc_async_read_page(m0 * ot + g, src, slot);
            if (m1 != m0) {
                noc_async_read_page(m1 * ot + g, src, slot + kTileBytes);
            }
        }
        noc_async_read_barrier();
        for (uint32_t r = r0; r < r_end; ++r) {
            // Pixels p0 .. p0 + cv - 1 of this output row in the padded-width rows.
            const uint32_t p0 = img * oh * wp + (th * 32 + r) * wp + tw * 32;
            const uint32_t m0 = p0 / 32;
            const uint32_t slot = src_l1 + (r - r0) * 2 * kTileBytes;
            // 2 pixels x 2 channels per pair of 32-bit loads and stores; past cv zero.
            for (uint32_t col = 0; col < cv; col += 2) {
                const uint32_t p = p0 + col;
                const auto* t0 = (
                    reinterpret_cast<const uint16_t*>(
                        slot + (p / 32 == m0 ? 0 : kTileBytes)
                    )
                );
                const bool pair = col + 1 < cv;
                const auto* t1 = (
                    reinterpret_cast<const uint16_t*>(
                        slot + ((p + 1) / 32 == m0 ? 0 : kTileBytes)
                    )
                );
                const uint32_t f = face_offset(r, col);
                for (uint32_t k = 0; k < 32; k += 2) {
                    const uint32_t a = (
                        *reinterpret_cast<const uint32_t*>(t0 + face_offset(p & 31, k))
                    );
                    const uint32_t b =
                        (pair ? *reinterpret_cast<const uint32_t*>(
                                    t1 + face_offset((p + 1) & 31, k)
                                )
                              : 0);
                    *reinterpret_cast<uint32_t*>(
                        out + k * (kTileBytes / 2) + f
                    ) = (a & 0xFFFFu) | (b << 16);
                    *reinterpret_cast<uint32_t*>(
                        out + (k + 1) * (kTileBytes / 2) + f
                    ) = (a >> 16) | (b & 0xFFFF0000u);
                }
            }
        }
        // The quarter's rows: 8 face rows (256 bytes) in each of two faces per tile.
        const uint32_t face = (r0 / 16) * 2, row = r0 % 16;
        const uint32_t channels = o - g * 32 < 32 ? o - g * 32 : 32;
        for (uint32_t k = 0; k < channels; ++k) {
            const uint32_t page = ((img * o + g * 32 + k) * oht + th) * owt + tw;
            for (uint32_t f = face; f < face + 2; ++f) {
                const uint32_t offset = (f * 256 + row * 16) * 2;
                noc_async_write(
                    out_l1 + k * kTileBytes + offset,
                    dst.get_noc_addr(page, offset),
                    256
                );
            }
        }
        noc_async_write_barrier();
    }
}
