// Gather along the last dim: out[b, h, j] = x[b, h, idx[b, h, j]], indices clamped to
// [0, W). One output tile per unit, dealt over both data-movement RISCs.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    // Element offset of (r, c) in a 32x32 tile stored as four 16x16 faces.
    inline uint32_t face_offset(uint32_t r, uint32_t c) {
        return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
    }

    constexpr uint32_t kSentinel = 0x6A710001u;

}  // namespace

// Compile-time args: TensorAccessorArgs of x, the index, the output. Common runtime
// args: see program.cpp; per core: the core's index.
void kernel_main() {
    uint32_t a = 0;
    const uint32_t x_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t index_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t workers = get_common_arg_val<uint32_t>(a++);
    const uint32_t units = get_common_arg_val<uint32_t>(a++);
    const uint32_t wt_out = get_common_arg_val<uint32_t>(a++);
    const uint32_t ht_out = get_common_arg_val<uint32_t>(a++);
    const uint32_t rows = get_common_arg_val<uint32_t>(a++);
    const uint32_t cols = get_common_arg_val<uint32_t>(a++);
    const uint32_t ht_in = get_common_arg_val<uint32_t>(a++);
    const uint32_t wt_in = get_common_arg_val<uint32_t>(a++);
    const uint32_t width = get_common_arg_val<uint32_t>(a++);
    const uint32_t elem = get_common_arg_val<uint32_t>(a++);
    const uint32_t sentinel = get_common_arg_val<uint32_t>(a++);
    if (sentinel != kSentinel || width == 0) {
        return;
    }
    const uint32_t worker = get_arg_val<uint32_t>(0) * 2 + ROLE;

    constexpr uint32_t cb_index = ROLE * 3, cb_out = ROLE * 3 + 1,
                       cb_stage = ROLE * 3 + 2;
    const uint32_t tile_bytes = 1024 * elem;
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto
        index_args = (TensorAccessorArgs<x_args.next_compile_time_args_offset()>());
    constexpr auto
        out_args = (TensorAccessorArgs<index_args.next_compile_time_args_offset()>());
    const auto x = TensorAccessor(x_args, x_addr, tile_bytes);
    const auto index = TensorAccessor(index_args, index_addr, 4096);
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    cb_reserve_back(cb_index, 1);
    cb_reserve_back(cb_out, 1);
    cb_reserve_back(cb_stage, 1);
    const uint32_t index_l1 = get_write_ptr(cb_index);
    const uint32_t out_l1 = get_write_ptr(cb_out);
    const uint32_t stage_l1 = (get_write_ptr(cb_stage) + 63) &
        ~63u;  // 1024 slots of 64 B
    const auto* ids = reinterpret_cast<volatile uint32_t*>(index_l1);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);

    for (uint32_t u = worker; u < units; u += workers) {
        const uint32_t tw = u % wt_out, rest = u / wt_out;
        const uint32_t th = rest % ht_out, b = rest / ht_out;
        const uint32_t nr = rows - th * 32 < 32 ? rows - th * 32 : 32;
        const uint32_t nc = cols - tw * 32 < 32 ? cols - tw * 32 : 32;
        if (nr >= 8) {
            noc_async_read_page(u, index, index_l1);
        } else {
            for (uint32_t r = 0; r < nr; ++r) {
                for (uint32_t h = 0; h * 16 < nc; ++h) {
                    const uint32_t at = face_offset(r, h * 16) * 4;
                    noc_async_read(index.get_noc_addr(u, at), index_l1 + at, 64);
                }
            }
        }
        if (nr < 32 || nc < 32) {
            for (uint32_t off = 0; off < tile_bytes; off += MEM_ZEROS_SIZE) {
                noc_async_read(zeros, out_l1 + off, MEM_ZEROS_SIZE);
            }
        }
        noc_async_read_barrier();

        // One 64-byte read per source segment; equal consecutive segments share it.
        const uint32_t row_page = (b * ht_in + th) * wt_in;
        uint32_t prev = 0xFFFFFFFFu, slot = 0;
        for (uint32_t r = 0; r < nr; ++r) {
            for (uint32_t c = 0; c < nc; ++c) {
                uint32_t v = ids[face_offset(r, c)];
                v = v < width ? v : width - 1;
                const uint32_t page = row_page + (v >> 5);
                const uint32_t off = face_offset(r, v & 31) * elem;
                const uint32_t key = (page << 6) | (off >> 6);
                if (key != prev) {
                    noc_async_read(
                        x.get_noc_addr(page, off & ~63u), stage_l1 + slot * 64, 64
                    );
                    prev = key;
                    ++slot;
                }
            }
        }
        noc_async_read_barrier();

        prev = 0xFFFFFFFFu;
        uint32_t seg = stage_l1 - 64;
        for (uint32_t r = 0; r < nr; ++r) {
            for (uint32_t c = 0; c < nc; ++c) {
                uint32_t v = ids[face_offset(r, c)];
                v = v < width ? v : width - 1;
                const uint32_t page = row_page + (v >> 5);
                const uint32_t off = face_offset(r, v & 31) * elem;
                const uint32_t key = (page << 6) | (off >> 6);
                if (key != prev) {
                    prev = key;
                    seg += 64;
                }
                const uint32_t src = seg + (off & 63),
                               dst = out_l1 + face_offset(r, c) * elem;
                if (elem == 4) {
                    *reinterpret_cast<volatile uint32_t*>(
                        dst
                    ) = *reinterpret_cast<volatile uint32_t*>(src);
                } else {
                    *reinterpret_cast<volatile uint16_t*>(
                        dst
                    ) = *reinterpret_cast<volatile uint16_t*>(src);
                }
            }
        }
        noc_async_write_page(u, out, out_l1);
        noc_async_write_barrier();
    }
}
