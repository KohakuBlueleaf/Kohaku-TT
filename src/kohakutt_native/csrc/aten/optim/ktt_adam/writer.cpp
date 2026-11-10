// Blocked tile writer of the Adam step: tiles [start, start + count) of the new
// parameter, first and second moment from CB 16 - 18, `block` tiles of each per CB
// transaction.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    // Waits for n pages of `cb` and queues their writes to tiles [t, t + n) of `dst`.
    template <typename Accessor>
    inline void write(
        uint32_t cb, const Accessor& dst, uint32_t t, uint32_t n, uint32_t page
    ) {
        cb_wait_front(cb, n);
        uint32_t addr = get_read_ptr(cb);
        for (uint32_t k = 0; k < n; ++k) {
            noc_async_write_page(t + k, dst, addr);
            addr += page;
        }
    }

}  // namespace

// Compile-time args: the three accessors. Common runtime args: the three addresses,
// block size; per core: first tile, tile count.
void kernel_main() {
    constexpr auto p_args = TensorAccessorArgs<0>();
    constexpr auto
        m_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto
        v_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    const uint32_t block = get_common_arg_val<uint32_t>(3);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);

    constexpr uint32_t first = 16;
    const uint32_t page = get_tile_size(first);
    const auto p = TensorAccessor(p_args, get_common_arg_val<uint32_t>(0), page);
    const auto m = TensorAccessor(m_args, get_common_arg_val<uint32_t>(1), page);
    const auto v = TensorAccessor(v_args, get_common_arg_val<uint32_t>(2), page);

    const uint32_t end = start + count;
    for (uint32_t t = start; t < end; t += block) {
        const uint32_t n = end - t < block ? end - t : block;
        write(first, p, t, n, page);
        write(first + 1, m, t, n, page);
        write(first + 2, v, t, n, page);
        noc_async_writes_flushed();
        for (uint32_t i = 0; i < 3; ++i) {
            cb_pop_front(first + i, n);
        }
    }
    noc_async_write_barrier();
}
