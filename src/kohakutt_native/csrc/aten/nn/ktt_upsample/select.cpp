// Member j of a small last dim: TILE [P, R, N, d] (d < 32) -> [P, R, N] (x[..., j]):
// per output tile, the input tiles of its 32 rows (one per R index), column j of each
// read as a row (out[r][c] = in[r][c][j]; past R or N zero).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    // Output rows `rows` .. `filled` - 1 set to zero (rows past R stay zero after).
    template <typename T>
    void clear(uint32_t out_l1, uint32_t rows, uint32_t filled) {
        auto* out = reinterpret_cast<T*>(out_l1);
        for (uint32_t r = rows; r < filled; ++r) {
            T* to = out + (r >> 4) * 512 + (r & 15) * 16;
            for (uint32_t c = 0; c < 32; ++c) {
                to[(c >> 4) * 256 + (c & 15)] = T(0);
            }
        }
    }

    // Row r < rows of the output tile = column j of input tile r (its 32 rows).
    template <typename T>
    void gather(uint32_t in_l1, uint32_t out_l1, uint32_t rows, uint32_t j) {
        const auto* in = reinterpret_cast<const T*>(in_l1);
        auto* out = reinterpret_cast<T*>(out_l1);
        const uint32_t from_col = (j >> 4) * 256 + (j & 15);
        for (uint32_t r = 0; r < rows; ++r) {
            const T* tile = in + r * 1024 + from_col;
            T* to = out + (r >> 4) * 512 + (r & 15) * 16;
            // Input rows c (faces 0 / 2) become output columns c (faces 0 / 1).
            for (uint32_t c = 0; c < 32; ++c) {
                to[(c >> 4) * 256 + (c & 15)] = tile[(c >> 4) * 512 + (c & 15) * 16];
            }
        }
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of the input, then of the output. Common
// runtime args: input address, output address, unit step, planes P, R, input row tiles
// (of N), output row tiles (of R), output column tiles (of N), j, element bytes (2 or
// 4); per core: first unit.
void kernel_main() {
    uint32_t a = 0;
    const uint32_t in_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t unit_step = get_common_arg_val<uint32_t>(a++);
    const uint32_t planes = get_common_arg_val<uint32_t>(a++);
    const uint32_t r_count = get_common_arg_val<uint32_t>(a++);
    const uint32_t nt = get_common_arg_val<uint32_t>(a++);
    const uint32_t oht = get_common_arg_val<uint32_t>(a++);
    const uint32_t owt = get_common_arg_val<uint32_t>(a++);
    const uint32_t j = get_common_arg_val<uint32_t>(a++);
    const uint32_t elem = get_common_arg_val<uint32_t>(a++);
    const uint32_t start = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in = 0, cb_out = 1;
    const uint32_t tile_bytes = 1024 * elem;
    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto
        out_args = (TensorAccessorArgs<in_args.next_compile_time_args_offset()>());
    const auto in = TensorAccessor(in_args, in_addr, tile_bytes);
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    // Scratch buffers (one page each): 32 input tiles, one output tile.
    cb_reserve_back(cb_in, 1);
    cb_reserve_back(cb_out, 1);
    const uint32_t in_l1 = get_write_ptr(cb_in);
    const uint32_t out_l1 = get_write_ptr(cb_out);

    const uint32_t units = planes * oht * owt;
    uint32_t filled = 32;  // rows of the output buffer that may be non-zero
    for (uint32_t u = start; u < units; u += unit_step) {
        const uint32_t p = u / (oht * owt), rt = (u / owt) % oht, ct = u % owt;
        const uint32_t r0 = rt * 32;
        const uint32_t rows = r_count - r0 < 32 ? r_count - r0 : 32;
        // Input tile (p R + r, ct): its [N, d] plane has nt row tiles, one column.
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t page = (p * r_count + r0 + r) * nt + ct;
            noc_async_read_page(page, in, in_l1 + r * tile_bytes);
        }
        noc_async_read_barrier();
        // The previous unit's write (overlapping this unit's reads) leaves the buffer.
        noc_async_write_barrier();
        if (elem == 4) {
            clear<uint32_t>(out_l1, rows, filled);
            gather<uint32_t>(in_l1, out_l1, rows, j);
        } else {
            clear<uint16_t>(out_l1, rows, filled);
            gather<uint16_t>(in_l1, out_l1, rows, j);
        }
        filled = rows;
        noc_async_write_page(u, out, out_l1);
    }
    noc_async_write_barrier();
}
