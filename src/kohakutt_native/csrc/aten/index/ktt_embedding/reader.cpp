// Row gather: out [B, n, D] = table [B, V, D] rows at int32 indices [1, n] (all TILE),
// one output tile (32 rows, one column tile) per unit.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    // Element offset of (r, c) in a 32x32 tile stored as four 16x16 faces.
    inline uint32_t face_offset(uint32_t r, uint32_t c) {
        return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of the table, indices, output. Runtime args:
// table, index and output addresses, first unit, unit step, n, D tiles, B, V tiles,
// element bytes (2 / 4).
void kernel_main() {
    uint32_t a = 0;
    const uint32_t table_addr = get_arg_val<uint32_t>(a++);
    const uint32_t index_addr = get_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t start = get_arg_val<uint32_t>(a++);
    const uint32_t step = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++);
    const uint32_t dt = get_arg_val<uint32_t>(a++);
    const uint32_t batch = get_arg_val<uint32_t>(a++);
    const uint32_t vt = get_arg_val<uint32_t>(a++);
    const uint32_t elem = get_arg_val<uint32_t>(a++);

    constexpr uint32_t cb_index = 0, cb_out = 1, cb_stage = 2;
    const uint32_t tile_bytes = 1024 * elem, half = 16 * elem;
    constexpr auto table_args = TensorAccessorArgs<0>();
    constexpr auto index_args = (
        TensorAccessorArgs<table_args.next_compile_time_args_offset()>()
    );
    constexpr auto out_args = (
        TensorAccessorArgs<index_args.next_compile_time_args_offset()>()
    );
    const auto table = TensorAccessor(table_args, table_addr, tile_bytes);
    const auto index = TensorAccessor(index_args, index_addr, 4096);
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    cb_reserve_back(cb_index, 1);
    cb_reserve_back(cb_out, 1);
    cb_reserve_back(cb_stage, 1);
    const uint32_t index_l1 = get_write_ptr(cb_index);
    const uint32_t out_l1 = get_write_ptr(cb_out);
    const uint32_t stage_l1 = get_write_ptr(cb_stage);  // 64 slots of 64 bytes
    const auto* ids = reinterpret_cast<volatile uint32_t*>(index_l1);

    const uint32_t blocks = (n + 31) / 32;
    uint32_t loaded = 0xFFFFFFFFu;
    for (uint32_t u = start; u < batch * blocks * dt; u += step) {
        const uint32_t col = u % dt, block = (u / dt) % blocks, b = u / dt / blocks;
        if (block != loaded) {
            noc_async_read_page(block, index, index_l1);
            noc_async_read_barrier();
            loaded = block;
        }
        const uint32_t rows = n - block * 32 < 32 ? n - block * 32 : 32;
        for (uint32_t k = 0; k < rows; ++k) {
            const uint32_t id = ids[face_offset(0, k)];
            const uint32_t page = (b * vt + id / 32) * dt + col, r = id % 32;
            for (uint32_t h = 0; h < 2; ++h) {
                const uint32_t src = face_offset(r, h * 16) * elem;
                const uint32_t dst = face_offset(k, h * 16) * elem;
                // bf16 half rows (32 bytes) land in a 64-byte slot at the source's
                // alignment, then are copied into place.
                const uint32_t at =
                    (elem == 4 ? out_l1 + dst
                               : stage_l1 + (k * 2 + h) * 64 + (src & 63));
                noc_async_read(table.get_noc_addr(page, src), at, half);
            }
        }
        noc_async_read_barrier();
        if (elem == 2) {
            for (uint32_t k = 0; k < rows; ++k) {
                const uint32_t r = ids[face_offset(0, k)] % 32;
                for (uint32_t h = 0; h < 2; ++h) {
                    const uint32_t src = face_offset(r, h * 16) * 2;
                    const auto* from = (
                        reinterpret_cast<volatile uint32_t*>(
                            stage_l1 + (k * 2 + h) * 64 + (src & 63)
                        )
                    );
                    auto* to = (
                        reinterpret_cast<volatile uint32_t*>(
                            out_l1 + face_offset(k, h * 16) * 2
                        )
                    );
                    for (uint32_t i = 0; i < 8; ++i) {
                        to[i] = from[i];
                    }
                }
            }
        }
        noc_async_write_page((b * blocks + block) * dt + col, out, out_l1);
        noc_async_write_barrier();
    }
}
