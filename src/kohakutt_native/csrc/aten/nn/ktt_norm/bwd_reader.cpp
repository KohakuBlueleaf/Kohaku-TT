// Norm backward reader: once, the core's gamma column tiles; then per tile row the
// row's mean (layer norm) / rstd tiles and the core's segments of x and dy, read from
// the core's rotation.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "args.hpp"

// Compile-time args: TensorAccessorArgs of x, dy, mean, rstd, gamma (mean / gamma: of x
// when absent), then the common arg count. Common runtime args: x, dy, mean, rstd,
// gamma addresses, Wt, tile rows, groups, has mean, has gamma, the x / dy ring pages
// per row. Per core: group, first column tile, column tiles, read rotation.
void kernel_main() {
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto
        dy_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto
        m_args = TensorAccessorArgs<dy_args.next_compile_time_args_offset()>();
    constexpr auto
        r_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    constexpr auto
        g_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    if (!ktt_norm::args_ok<g_args.next_compile_time_args_offset()>(12)) {
        return;
    }
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t dy_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t mean_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t rstd_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t g_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t wt = get_common_arg_val<uint32_t>(5);
    const uint32_t rows = get_common_arg_val<uint32_t>(6);
    const uint32_t groups = get_common_arg_val<uint32_t>(7);
    const bool has_mean = get_common_arg_val<uint32_t>(8) != 0;
    const bool has_gamma = get_common_arg_val<uint32_t>(9) != 0;
    const uint32_t stride = get_common_arg_val<uint32_t>(10);
    const uint32_t gid = get_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t seg = get_arg_val<uint32_t>(2);
    const uint32_t rot = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_x = 0, cb_g = 1, cb_st = 8, cb_dy = 9;
    constexpr uint32_t kF32Tile = 4096;
    const uint32_t x_tile = get_tile_size(cb_x);
    const auto x = TensorAccessor(x_args, x_addr, x_tile);
    const auto dy = TensorAccessor(dy_args, dy_addr, x_tile);
    const auto mean = TensorAccessor(m_args, mean_addr, kF32Tile);
    const auto rstd = TensorAccessor(r_args, rstd_addr, kF32Tile);

    if (has_gamma) {
        const uint32_t tile = get_tile_size(cb_g);
        const auto g = TensorAccessor(g_args, g_addr, tile);
        cb_reserve_back(cb_g, seg);
        const uint32_t base = get_write_ptr(cb_g);
        for (uint32_t j = 0; j < seg; ++j) {
            noc_async_read_page(first + j, g, base + j * tile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_g, seg);
    }

    for (uint32_t r = gid; r < rows; r += groups) {
        cb_reserve_back(cb_st, 2);
        cb_reserve_back(cb_x, stride);
        cb_reserve_back(cb_dy, stride);
        const uint32_t st = get_write_ptr(cb_st);
        if (has_mean) {
            noc_async_read_page(r, mean, st);
        }
        noc_async_read_page(r, rstd, st + kF32Tile);
        const uint32_t xb = get_write_ptr(cb_x), db = get_write_ptr(cb_dy);
        const uint32_t page = r * wt + first;
        uint32_t j = rot;
        for (uint32_t i = 0; i < seg; ++i) {
            noc_async_read_page(page + j, x, xb + j * x_tile);
            noc_async_read_page(page + j, dy, db + j * x_tile);
            j = j + 1 == seg ? 0 : j + 1;
        }
        noc_async_read_barrier();
        cb_push_back(cb_st, 2);
        cb_push_back(cb_x, stride);
        cb_push_back(cb_dy, stride);
    }
}
