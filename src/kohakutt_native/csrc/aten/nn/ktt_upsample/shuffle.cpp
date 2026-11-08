// Pixel shuffle by r dividing 32 of a TILE [P, C r^2, H, W] tensor -> [P, C, r H, r W]:
// per output tile, the r^2 input tiles (channels c r^2 + i r + j) at one spatial tile,
// gathered in L1 (out[R][Q] = in[c r^2 + (R % r) r + Q % r][R / r][Q / r]) and written.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    // Element offset of (r, c) in a 32x32 tile stored as four 16x16 faces.
    inline uint32_t face_offset(uint32_t r, uint32_t c) {
        return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
    }

    template <typename T>
    void gather(
        uint32_t in_l1,
        uint32_t out_l1,
        uint32_t tile_elems,
        uint32_t r,
        uint32_t h0,
        uint32_t w0
    ) {
        auto* out = reinterpret_cast<T*>(out_l1);
        for (uint32_t rr = 0; rr < 32; ++rr) {
            for (uint32_t cc = 0; cc < 32; ++cc) {
                const auto* t =
                    (reinterpret_cast<const T*>(in_l1) +
                     ((rr % r) * r + cc % r) * tile_elems);
                out[face_offset(rr, cc)] = t[face_offset(h0 + rr / r, w0 + cc / r)];
            }
        }
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of the input, then of the output. Runtime args:
// input address, output address, first unit, unit step, output planes (P C), C, input
// row / column tiles, output row / column tiles, r, element bytes (2 or 4).
void kernel_main() {
    uint32_t a = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t start = get_arg_val<uint32_t>(a++);
    const uint32_t step = get_arg_val<uint32_t>(a++);
    const uint32_t planes = get_arg_val<uint32_t>(a++);
    const uint32_t c = get_arg_val<uint32_t>(a++);
    const uint32_t iht = get_arg_val<uint32_t>(a++);
    const uint32_t iwt = get_arg_val<uint32_t>(a++);
    const uint32_t oht = get_arg_val<uint32_t>(a++);
    const uint32_t owt = get_arg_val<uint32_t>(a++);
    const uint32_t r = get_arg_val<uint32_t>(a++);
    const uint32_t elem = get_arg_val<uint32_t>(a++);

    constexpr uint32_t cb_in = 0, cb_out = 1;
    const uint32_t tile_bytes = 1024 * elem;
    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto out_args = (
        TensorAccessorArgs<in_args.next_compile_time_args_offset()>()
    );
    const auto in = TensorAccessor(in_args, in_addr, tile_bytes);
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    // Scratch buffers (one page each): r^2 input tiles, one output tile.
    cb_reserve_back(cb_in, 1);
    cb_reserve_back(cb_out, 1);
    const uint32_t in_l1 = get_write_ptr(cb_in);
    const uint32_t out_l1 = get_write_ptr(cb_out);

    const uint32_t units = planes * oht * owt;
    for (uint32_t u = start; u < units; u += step) {
        const uint32_t plane = u / (oht * owt), th = (u / owt) % oht, tw = u % owt;
        const uint32_t prefix = plane / c, ch = plane % c;
        const uint32_t h0 = th * 32 / r, w0 = tw * 32 / r;
        for (uint32_t k = 0; k < r * r; ++k) {
            const uint32_t in_plane = (prefix * c + ch) * r * r + k;
            const uint32_t page = (in_plane * iht + h0 / 32) * iwt + w0 / 32;
            noc_async_read_page(page, in, in_l1 + k * tile_bytes);
        }
        noc_async_read_barrier();
        if (elem == 4) {
            gather<uint32_t>(in_l1, out_l1, 1024, r, h0 % 32, w0 % 32);
        } else {
            gather<uint16_t>(in_l1, out_l1, 1024, r, h0 % 32, w0 % 32);
        }
        noc_async_write_page(u, out, out_l1);
        noc_async_write_barrier();
    }
}
