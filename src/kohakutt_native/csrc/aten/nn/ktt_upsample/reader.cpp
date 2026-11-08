// Nearest upsampling by integer factors (kh, kw dividing 32) of a TILE [N C, H, W]
// tensor: per output tile, the one input tile it reads, gathered in L1
// (out[r][c] = in[(r0 + r) / kh][(c0 + c) / kw]) and written.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    // Element offset of (r, c) in a 32x32 tile stored as four 16x16 faces.
    inline uint32_t face_offset(uint32_t r, uint32_t c) {
        return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
    }

    // out tile = gather of the input tile; T: 2- or 4-byte elements.
    template <typename T>
    void gather(
        uint32_t in_l1,
        uint32_t out_l1,
        uint32_t r_off,
        uint32_t c_off,
        uint32_t kh,
        uint32_t kw
    ) {
        const auto* in = reinterpret_cast<const T*>(in_l1);
        auto* out = reinterpret_cast<T*>(out_l1);
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t sr = r_off + r / kh;
            for (uint32_t c = 0; c < 32; ++c) {
                out[face_offset(r, c)] = in[face_offset(sr, c_off + c / kw)];
            }
        }
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of the input, then of the output. Common
// runtime args: input address, output address, unit step, planes (N C), input row /
// column tiles, output row / column tiles, kh, kw, element bytes (2 or 4); per core:
// first unit.
void kernel_main() {
    uint32_t a = 0;
    const uint32_t in_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t step = get_common_arg_val<uint32_t>(a++);
    const uint32_t planes = get_common_arg_val<uint32_t>(a++);
    const uint32_t iht = get_common_arg_val<uint32_t>(a++);
    const uint32_t iwt = get_common_arg_val<uint32_t>(a++);
    const uint32_t oht = get_common_arg_val<uint32_t>(a++);
    const uint32_t owt = get_common_arg_val<uint32_t>(a++);
    const uint32_t kh = get_common_arg_val<uint32_t>(a++);
    const uint32_t kw = get_common_arg_val<uint32_t>(a++);
    const uint32_t elem = get_common_arg_val<uint32_t>(a++);
    const uint32_t start = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in = 0, cb_out = 1;
    const uint32_t tile_bytes = 1024 * elem;
    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto out_args = (
        TensorAccessorArgs<in_args.next_compile_time_args_offset()>()
    );
    const auto in = TensorAccessor(in_args, in_addr, tile_bytes);
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    // Scratch buffers (one page each).
    cb_reserve_back(cb_in, 1);
    cb_reserve_back(cb_out, 1);
    const uint32_t in_l1 = get_write_ptr(cb_in);
    const uint32_t out_l1 = get_write_ptr(cb_out);

    const uint32_t units = planes * oht * owt;
    uint32_t loaded = 0xFFFFFFFF;
    for (uint32_t u = start; u < units; u += step) {
        const uint32_t plane = u / (oht * owt), th = (u / owt) % oht, tw = u % owt;
        // Input rows / columns of this output tile start at th 32 / kh, tw 32 / kw.
        const uint32_t r0 = th * 32 / kh, c0 = tw * 32 / kw;
        const uint32_t src = (plane * iht + r0 / 32) * iwt + c0 / 32;
        if (src != loaded) {
            noc_async_read_page(src, in, in_l1);
            noc_async_read_barrier();
            loaded = src;
        }
        if (elem == 4) {
            gather<uint32_t>(in_l1, out_l1, r0 % 32, c0 % 32, kh, kw);
        } else {
            gather<uint16_t>(in_l1, out_l1, r0 % 32, c0 % 32, kh, kw);
        }
        noc_async_write_page(u, out, out_l1);
        noc_async_write_barrier();
    }
}
