// Every step-th column of a TILE [P, H, W] tensor from column `first` -> [P, H, count]:
// per output tile, the input tiles its columns come from, gathered in L1
// (out[r][c] = in[r][first + step (c0 + c)]; columns past `count` are zero).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    // out tile = the columns of the loaded input tiles (tile 0 is input tile `t0`).
    template <typename T>
    void gather(
        uint32_t in_l1,
        uint32_t out_l1,
        uint32_t c0,
        uint32_t t0,
        uint32_t first,
        uint32_t step,
        uint32_t count
    ) {
        const auto* in = reinterpret_cast<const T*>(in_l1);
        auto* out = reinterpret_cast<T*>(out_l1);
        for (uint32_t c = 0; c < 32; ++c) {
            const bool real = c0 + c < count;
            const uint32_t src = first + step * (c0 + c);
            const uint32_t sc = src % 32;
            const T* from = in + (real ? src / 32 - t0 : 0) * 1024 + (sc >> 4) * 256 +
                (sc & 15);
            T* to = out + (c >> 4) * 256 + (c & 15);
            // Rows 0 .. 15 in faces 0 / 1, rows 16 .. 31 two faces on.
            for (uint32_t half = 0; half < 2; ++half) {
                for (uint32_t r = 0; r < 16; ++r) {
                    to[half * 512 + r * 16] = real ? from[half * 512 + r * 16] : T(0);
                }
            }
        }
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of the input, then of the output. Common
// runtime args: input address, output address, unit step, planes, row tiles, input
// column tiles, output column tiles, first, step, count, element bytes (2 or 4); per
// core: first unit.
void kernel_main() {
    uint32_t a = 0;
    const uint32_t in_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t unit_step = get_common_arg_val<uint32_t>(a++);
    const uint32_t planes = get_common_arg_val<uint32_t>(a++);
    const uint32_t ht = get_common_arg_val<uint32_t>(a++);
    const uint32_t iwt = get_common_arg_val<uint32_t>(a++);
    const uint32_t owt = get_common_arg_val<uint32_t>(a++);
    const uint32_t first = get_common_arg_val<uint32_t>(a++);
    const uint32_t step = get_common_arg_val<uint32_t>(a++);
    const uint32_t count = get_common_arg_val<uint32_t>(a++);
    const uint32_t elem = get_common_arg_val<uint32_t>(a++);
    const uint32_t start = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in = 0, cb_out = 1;
    const uint32_t tile_bytes = 1024 * elem;
    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto
        out_args = (TensorAccessorArgs<in_args.next_compile_time_args_offset()>());
    const auto in = TensorAccessor(in_args, in_addr, tile_bytes);
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    // Scratch buffers (one page each): step + 1 input tiles, one output tile.
    cb_reserve_back(cb_in, 1);
    cb_reserve_back(cb_out, 1);
    const uint32_t in_l1 = get_write_ptr(cb_in);
    const uint32_t out_l1 = get_write_ptr(cb_out);

    const uint32_t units = planes * ht * owt;
    for (uint32_t u = start; u < units; u += unit_step) {
        const uint32_t row = u / owt, tw = u % owt;
        const uint32_t c0 = tw * 32;
        const uint32_t last = c0 + 31 < count ? c0 + 31 : count - 1;
        const uint32_t t0 = (first + step * c0) / 32;
        const uint32_t t1 = (first + step * last) / 32;
        for (uint32_t t = t0; t <= t1 && t < iwt; ++t) {
            noc_async_read_page(row * iwt + t, in, in_l1 + (t - t0) * tile_bytes);
        }
        noc_async_read_barrier();
        // The previous unit's write (overlapping this unit's reads) leaves the buffer.
        noc_async_write_barrier();
        if (elem == 4) {
            gather<uint32_t>(in_l1, out_l1, c0, t0, first, step, count);
        } else {
            gather<uint16_t>(in_l1, out_l1, c0, t0, first, step, count);
        }
        noc_async_write_page(u, out, out_l1);
    }
    noc_async_write_barrier();
}
