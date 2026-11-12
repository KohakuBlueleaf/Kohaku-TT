// Index copy in place: rows (or planes) of dst at int32 indices replaced by the
// source's in index order; one unit per (batch, column tile) or (outer, plane tile).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    // Element offset of (r, c) in a 32x32 tile stored as four 16x16 faces.
    inline uint32_t face_offset(uint32_t r, uint32_t c) {
        return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
    }

    // Row `from` of the tile at `src` over row `to` of the tile at `dst`.
    inline void copy_row(
        uint32_t src, uint32_t from, uint32_t dst, uint32_t to, uint32_t elem
    ) {
        const uint32_t words = 4 * elem;  // 16 elements
        for (uint32_t h = 0; h < 2; ++h) {
            const auto* s = reinterpret_cast<volatile uint32_t*>(
                src + face_offset(from, h * 16) * elem
            );
            auto* d = reinterpret_cast<volatile uint32_t*>(
                dst + face_offset(to, h * 16) * elem
            );
            for (uint32_t i = 0; i < words; ++i) {
                d[i] = s[i];
            }
        }
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of the source, the index, dst. Common runtime
// args: see program.cpp; per core: first unit.
void kernel_main() {
    uint32_t a = 0;
    const uint32_t src_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t index_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t dst_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t step = get_common_arg_val<uint32_t>(a++);
    const uint32_t units = get_common_arg_val<uint32_t>(a++);
    const uint32_t n = get_common_arg_val<uint32_t>(a++);
    const uint32_t planes = get_common_arg_val<uint32_t>(a++);
    const uint32_t dt = get_common_arg_val<uint32_t>(a++);
    const uint32_t lt = get_common_arg_val<uint32_t>(a++);
    const uint32_t length = get_common_arg_val<uint32_t>(a++);
    const uint32_t tt = get_common_arg_val<uint32_t>(a++);
    const uint32_t src_wt = get_common_arg_val<uint32_t>(a++);
    const uint32_t src_coff = get_common_arg_val<uint32_t>(a++);
    const uint32_t src_hstride = get_common_arg_val<uint32_t>(a++);
    const uint32_t heads = get_common_arg_val<uint32_t>(a++);
    const uint32_t elem = get_common_arg_val<uint32_t>(a++);
    const uint32_t start = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_index = 0, cb_dst = 1, cb_src = 2;
    const uint32_t tile_bytes = 1024 * elem;
    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr auto
        index_args = (TensorAccessorArgs<src_args.next_compile_time_args_offset()>());
    constexpr auto
        dst_args = (TensorAccessorArgs<index_args.next_compile_time_args_offset()>());
    const auto src = TensorAccessor(src_args, src_addr, tile_bytes);
    const auto index = TensorAccessor(index_args, index_addr, 4096);
    const auto dst = TensorAccessor(dst_args, dst_addr, tile_bytes);

    cb_reserve_back(cb_index, 1);
    cb_reserve_back(cb_dst, 1);
    cb_reserve_back(cb_src, 1);
    const uint32_t index_l1 = get_write_ptr(cb_index);
    const uint32_t dst_l1 = get_write_ptr(cb_dst);
    const uint32_t src_l1 = get_write_ptr(cb_src);
    const auto* ids = reinterpret_cast<volatile uint32_t*>(index_l1);

    constexpr uint32_t kNone = 0xFFFFFFFFu;
    for (uint32_t u = start; u < units; u += step) {
        uint32_t loaded = kNone, held = kNone, read = kNone;
        for (uint32_t k = 0; k < n; ++k) {
            if (k / 32 != loaded) {
                loaded = k / 32;
                noc_async_read_page(loaded, index, index_l1);
                noc_async_read_barrier();
            }
            const uint32_t v = ids[face_offset(0, k % 32)];
            if (v >= length) {
                continue;
            }
            if (planes) {
                // Planes: tile u % dt of plane (outer, k) to plane (outer, v).
                const uint32_t j = u % dt, outer = u / dt;
                noc_async_read_page((outer * n + k) * dt + j, src, dst_l1);
                noc_async_read_barrier();
                noc_async_write_page((outer * length + v) * dt + j, dst, dst_l1);
                noc_async_write_barrier();
                continue;
            }
            const uint32_t col = u % dt, b = u / dt;
            const uint32_t page = (b * lt + v / 32) * dt + col;
            if (page != held) {
                if (held != kNone) {
                    noc_async_write_page(held, dst, dst_l1);
                    noc_async_write_barrier();
                }
                noc_async_read_page(page, dst, dst_l1);
                noc_async_read_barrier();
                held = page;
            }
            const uint32_t from = ((b / heads) * tt + k / 32) * src_wt + src_coff +
                (b % heads) * src_hstride + col;
            if (from != read) {
                noc_async_read_page(from, src, src_l1);
                noc_async_read_barrier();
                read = from;
            }
            copy_row(src_l1, k % 32, dst_l1, v % 32, elem);
        }
        if (held != kNone) {
            noc_async_write_page(held, dst, dst_l1);
            noc_async_write_barrier();
        }
    }
}
