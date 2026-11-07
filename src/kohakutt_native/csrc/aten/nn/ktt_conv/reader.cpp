// Conv reader: this core's weight block (all K = taps x Ct weight rows of its output
// column tiles) into CB 1 once; then per output row tile m and tap (i, j), the 32 input
// rows the tap reads (shifted by i dh Wp + j dw within the image) into CB 0, row-major,
// for the compute kernel to tilize.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Compile-time args: TensorAccessorArgs of x_rows, then of w. Runtime args: x address,
// w address, first / end output row tile, first output column tile, column tiles,
// output column tiles in all (Ot), Ct, KH, KW, dh, dw, Wp, input rows per image (Hp
// Wp), output rows per image (OH Wp), input rows in all, output rows in all, x row
// bytes.
void kernel_main() {
    uint32_t a = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(a++);
    const uint32_t w_addr = get_arg_val<uint32_t>(a++);
    const uint32_t m0 = get_arg_val<uint32_t>(a++);
    const uint32_t m1 = get_arg_val<uint32_t>(a++);
    const uint32_t n0 = get_arg_val<uint32_t>(a++);
    const uint32_t nt = get_arg_val<uint32_t>(a++);
    const uint32_t ot = get_arg_val<uint32_t>(a++);
    const uint32_t ct = get_arg_val<uint32_t>(a++);
    const uint32_t kh = get_arg_val<uint32_t>(a++);
    const uint32_t kw = get_arg_val<uint32_t>(a++);
    const uint32_t dh = get_arg_val<uint32_t>(a++);
    const uint32_t dw = get_arg_val<uint32_t>(a++);
    const uint32_t wp = get_arg_val<uint32_t>(a++);
    const uint32_t in_per_image = get_arg_val<uint32_t>(a++);
    const uint32_t out_per_image = get_arg_val<uint32_t>(a++);
    const uint32_t in_rows = get_arg_val<uint32_t>(a++);
    const uint32_t out_rows = get_arg_val<uint32_t>(a++);
    const uint32_t row_bytes = get_arg_val<uint32_t>(a++);

    constexpr uint32_t cb_rows = 0, cb_w = 1;
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto w_args = (
        TensorAccessorArgs<x_args.next_compile_time_args_offset()>()
    );
    const auto x = TensorAccessor(x_args, x_addr, row_bytes);
    const uint32_t tile = get_tile_size(cb_w);
    const auto w = TensorAccessor(w_args, w_addr, tile);

    // Weight block: tile (k, n0 + j) at CB position k nt + j.
    const uint32_t k_tiles = kh * kw * ct;
    cb_reserve_back(cb_w, k_tiles * nt);
    uint32_t addr = get_write_ptr(cb_w);
    for (uint32_t k = 0; k < k_tiles; ++k) {
        for (uint32_t j = 0; j < nt; ++j) {
            noc_async_read_page(k * ot + n0 + j, w, addr);
            addr += tile;
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_w, k_tiles * nt);

    for (uint32_t m = m0; m < m1; ++m) {
        for (uint32_t i = 0; i < kh; ++i) {
            for (uint32_t j = 0; j < kw; ++j) {
                const uint32_t shift = i * dh * wp + j * dw;
                cb_reserve_back(cb_rows, ct);
                uint32_t dst = get_write_ptr(cb_rows);
                for (uint32_t q = 0; q < 32; ++q) {
                    // Rows past the output (the last tile's padding) read any valid
                    // row.
                    uint32_t r = m * 32 + q;
                    r = r < out_rows ? r : out_rows - 1;
                    const uint32_t image = r / out_per_image;
                    uint32_t src = image * in_per_image + r % out_per_image + shift;
                    src = src < in_rows ? src : in_rows - 1;
                    noc_async_read_page(src, x, dst);
                    dst += row_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_rows, ct);
            }
        }
    }
}
