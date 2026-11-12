// The columns of k TILE [P, H, N] tensors interleaved -> [P, H, N k] (torch.stack(xs,
// -1) as [.., N k]): per output tile, the input tiles its columns come from, gathered
// in L1 (out[r][c] = x_{c % k}[r][c / k]; columns past N k are zero).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    // Part j's input tiles t0 and t0 + 1 sit at tiles 2 j and 2 j + 1 of in_l1.
    template <typename T>
    void gather(
        uint32_t in_l1,
        uint32_t out_l1,
        uint32_t c0,
        uint32_t k,
        uint32_t t0,
        uint32_t n
    ) {
        const auto* in = reinterpret_cast<const T*>(in_l1);
        auto* out = reinterpret_cast<T*>(out_l1);
        for (uint32_t c = 0; c < 32; ++c) {
            const uint32_t oc = c0 + c, j = oc % k, src = oc / k;
            const bool real = src < n;
            const uint32_t sc = src % 32;
            const T* from = in + (2 * j + (real ? src / 32 - t0 : 0)) * 1024 +
                (sc >> 4) * 256 + (sc & 15);
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

// Compile-time args: TensorAccessorArgs of the inputs (one layout), then of the output,
// then k (2 .. 8). Common runtime args: k input addresses, output address, unit step,
// planes, row tiles, input column tiles, output column tiles, N, element bytes (2 or
// 4); per core: first unit.
void kernel_main() {
    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto
        out_args = (TensorAccessorArgs<in_args.next_compile_time_args_offset()>());
    constexpr uint32_t k = get_compile_time_arg_val(
        out_args.next_compile_time_args_offset()
    );
    uint32_t a = k + 1;
    const uint32_t out_addr = get_common_arg_val<uint32_t>(k);
    const uint32_t unit_step = get_common_arg_val<uint32_t>(a++);
    const uint32_t planes = get_common_arg_val<uint32_t>(a++);
    const uint32_t ht = get_common_arg_val<uint32_t>(a++);
    const uint32_t iwt = get_common_arg_val<uint32_t>(a++);
    const uint32_t owt = get_common_arg_val<uint32_t>(a++);
    const uint32_t n = get_common_arg_val<uint32_t>(a++);
    const uint32_t elem = get_common_arg_val<uint32_t>(a++);
    const uint32_t start = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in = 0, cb_out = 1;
    const uint32_t tile_bytes = 1024 * elem;
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    // Scratch buffers (one page each): two input tiles per part, one output tile.
    cb_reserve_back(cb_in, 1);
    cb_reserve_back(cb_out, 1);
    const uint32_t in_l1 = get_write_ptr(cb_in);
    const uint32_t out_l1 = get_write_ptr(cb_out);

    const uint32_t units = planes * ht * owt;
    for (uint32_t u = start; u < units; u += unit_step) {
        const uint32_t row = u / owt, tw = u % owt;
        const uint32_t c0 = tw * 32;
        const uint32_t t0 = c0 / k / 32;
        const uint32_t t1 = (c0 + 31) / k / 32;
        for (uint32_t j = 0; j < k; ++j) {
            const auto in = TensorAccessor(
                in_args, get_common_arg_val<uint32_t>(j), tile_bytes
            );
            for (uint32_t t = t0; t <= t1 && t < iwt; ++t) {
                noc_async_read_page(
                    row * iwt + t, in, in_l1 + (2 * j + t - t0) * tile_bytes
                );
            }
        }
        noc_async_read_barrier();
        // The previous unit's write (overlapping this unit's reads) leaves the buffer.
        noc_async_write_barrier();
        if (elem == 4) {
            gather<uint32_t>(in_l1, out_l1, c0, k, t0, n);
        } else {
            gather<uint16_t>(in_l1, out_l1, c0, k, t0, n);
        }
        noc_async_write_page(u, out, out_l1);
    }
    noc_async_write_barrier();
}
