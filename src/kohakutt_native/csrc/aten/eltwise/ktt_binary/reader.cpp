// Broadcast binary reader: walks (j, r) pairs j-major, output tile r * sr + j * sj.
// Reads b's tile j into CB 1 once per j (row 0 / column 0 filled across the tile when
// broadcast inside it), and a's tiles of the pair range into CB 0, in blocks.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Copies row 0 (fill 1) or column 0 (fill 2) of a 32x32 tile across the tile, in L1.
template <typename T>
void fill_tile(uint32_t addr, uint32_t fill) {
    volatile T* t = reinterpret_cast<volatile T*>(addr);
    constexpr uint32_t face = 256;  // 16 x 16 elements per face, faces 0..3
    for (uint32_t f = 0; f < 4; ++f) {
        for (uint32_t i = 0; i < 16; ++i) {
            for (uint32_t c = 0; c < 16; ++c) {
                if (fill == 1) {
                    // Row 0 of the face column above: face 0 (f = 0, 2), face 1 (1, 3).
                    t[f * face + i * 16 + c] = t[(f % 2) * face + c];
                } else {
                    // Column 0 of the face row: face 0 (f = 0, 1), face 2 (2, 3).
                    t[f * face + i * 16 + c] = t[(f / 2) * 2 * face + i * 16];
                }
            }
        }
    }
}

// Common runtime args: a address, b address, repeats per j (nr), a stride per r (sr),
// a stride per j (sj), fill mode (0 none, 1 row, 2 column), block size, element
// bytes (2 or 4); per core: first pair, pair count.
void kernel_main() {
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = (
        TensorAccessorArgs<a_args.next_compile_time_args_offset()>()
    );
    const uint32_t a_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t nr = get_common_arg_val<uint32_t>(2);
    const uint32_t sr = get_common_arg_val<uint32_t>(3);
    const uint32_t sj = get_common_arg_val<uint32_t>(4);
    const uint32_t fill = get_common_arg_val<uint32_t>(5);
    const uint32_t block = get_common_arg_val<uint32_t>(6);
    const uint32_t elem = get_common_arg_val<uint32_t>(7);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_a = 0, cb_b = 1;
    const uint32_t page = get_tile_size(cb_a);
    const auto a = TensorAccessor(a_args, a_addr, page);
    const auto b = TensorAccessor(b_args, b_addr, page);

    const uint32_t end = start + count;
    uint32_t idx = start;
    while (idx < end) {
        const uint32_t j = idx / nr, r = idx % nr;

        // A new b tile at the start of each j group (and of this core's range).
        if (r == 0 || idx == start) {
            cb_reserve_back(cb_b, 1);
            const uint32_t addr = get_write_ptr(cb_b);
            noc_async_read_page(j, b, addr);
            noc_async_read_barrier();
            if (fill != 0) {
                if (elem == 4) {
                    fill_tile<uint32_t>(addr, fill);
                } else {
                    fill_tile<uint16_t>(addr, fill);
                }
            }
            cb_push_back(cb_b, 1);
        }

        // A block of a's tiles within this j group.
        uint32_t n = nr - r;
        n = n < block ? n : block;
        n = n < end - idx ? n : end - idx;
        cb_reserve_back(cb_a, n);
        uint32_t addr = get_write_ptr(cb_a);
        for (uint32_t k = 0; k < n; ++k) {
            noc_async_read_page((r + k) * sr + j * sj, a, addr);
            addr += page;
        }
        noc_async_read_barrier();
        cb_push_back(cb_a, n);
        idx += n;
    }
}
