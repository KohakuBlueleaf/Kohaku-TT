// Blocked tile reader of the Adam step: tiles [start, start + count) of the parameter,
// gradient, first and second moment into CB 0 - 3, `block` tiles of each per CB
// transaction with one read barrier per block.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    // Reserves n pages of `cb` and queues the reads of tiles [t, t + n) of `src` into
    // them.
    template <typename Accessor>
    inline void read(
        uint32_t cb, const Accessor& src, uint32_t t, uint32_t n, uint32_t page
    ) {
        cb_reserve_back(cb, n);
        uint32_t addr = get_write_ptr(cb);
        for (uint32_t k = 0; k < n; ++k) {
            noc_async_read_page(t + k, src, addr);
            addr += page;
        }
    }

}  // namespace

// Compile-time args: the four accessors. Common runtime args: the four addresses, block
// size; per core: first tile, tile count.
void kernel_main() {
    constexpr auto p_args = TensorAccessorArgs<0>();
    constexpr auto
        g_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto
        m_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto
        v_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    const uint32_t block = get_common_arg_val<uint32_t>(4);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);

    const uint32_t page = get_tile_size(0);
    const auto p = TensorAccessor(p_args, get_common_arg_val<uint32_t>(0), page);
    const auto g = TensorAccessor(g_args, get_common_arg_val<uint32_t>(1), page);
    const auto m = TensorAccessor(m_args, get_common_arg_val<uint32_t>(2), page);
    const auto v = TensorAccessor(v_args, get_common_arg_val<uint32_t>(3), page);

    const uint32_t end = start + count;
    for (uint32_t t = start; t < end; t += block) {
        const uint32_t n = end - t < block ? end - t : block;
        read(0, p, t, n, page);
        read(1, g, t, n, page);
        read(2, m, t, n, page);
        read(3, v, t, n, page);
        noc_async_read_barrier();
        for (uint32_t cb = 0; cb < 4; ++cb) {
            cb_push_back(cb, n);
        }
    }
}
