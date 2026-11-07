// Fused-chain reader: this core's output tiles column by column (each column's tile
// rows in order). Per input layout: full (the output's tile), matrix (one [R, C] matrix
// over the batch), row ([.., 1, C]), column ([.., R, 1]), scalar (tile 0). Row / scalar
// inputs without a batch are persistent: one tile per column, kept by the compute
// kernel for the column. Broadcasting within a tile is done by the compute kernel.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

enum Layout : uint32_t { FULL = 0, MATRIX = 1, ROW = 2, COLUMN = 3, SCALAR = 4 };

// The page of input `layout` for global tile row `gr` (batch b = gr / rt), column c.
uint32_t page_of(
    uint32_t layout, uint32_t batched, uint32_t gr, uint32_t c, uint32_t rt, uint32_t ct
) {
    const uint32_t b = gr / rt, r = gr % rt, bi = batched ? b : 0;
    switch (layout) {
        case FULL:
            return gr * ct + c;
        case MATRIX:
            return r * ct + c;
        case ROW:
            return bi * ct + c;
        case COLUMN:
            return bi * rt + r;
        default:
            return 0;
    }
}

inline bool persistent(uint32_t layout, uint32_t batched) {
    return (layout == ROW || layout == SCALAR) && !batched;
}

template <typename Accessor>
void read_one(const Accessor& in, uint32_t cb, uint32_t page) {
    cb_reserve_back(cb, 1);
    noc_async_read_page(page, in, get_write_ptr(cb));
    noc_async_read_barrier();
    cb_push_back(cb, 1);
}

// Pages for tile rows gr0 .. gr0 + n - 1 of column c into CB `cb` (capacity `cap`
// tiles, `filled` tiles pushed so far): one transaction per stretch up to the CB's end,
// so none wraps.
template <typename Accessor>
void read_run(
    const Accessor& in,
    uint32_t cb,
    uint32_t cap,
    uint32_t& filled,
    uint32_t layout,
    uint32_t batched,
    uint32_t gr0,
    uint32_t n,
    uint32_t c,
    uint32_t rt,
    uint32_t ct
) {
    const uint32_t page = get_tile_size(cb);
    while (n > 0) {
        const uint32_t room = cap - filled % cap;
        const uint32_t m = n < room ? n : room;
        cb_reserve_back(cb, m);
        uint32_t addr = get_write_ptr(cb);
        for (uint32_t k = 0; k < m; ++k) {
            noc_async_read_page(page_of(layout, batched, gr0 + k, c, rt, ct), in, addr);
            addr += page;
        }
        noc_async_read_barrier();
        cb_push_back(cb, m);
        filled += m;
        gr0 += m;
        n -= m;
    }
}

// Compile-time args: TensorAccessorArgs of inputs 0 .. NUM_INPUTS - 1. Runtime args:
// first tile, tile count, block, rt (tile rows per matrix), ct (tile columns), then per
// input: address, layout, batched (0 / 1).
void kernel_main() {
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    const uint32_t block = get_arg_val<uint32_t>(2);
    const uint32_t rt = get_arg_val<uint32_t>(3);
    const uint32_t ct = get_arg_val<uint32_t>(4);
    auto arg = [](uint32_t i, uint32_t k) {
        return get_arg_val<uint32_t>(5 + 3 * i + k);
    };

    constexpr auto args0 = TensorAccessorArgs<0>();
    const auto in0 = TensorAccessor(args0, arg(0, 0), get_tile_size(0));
#if NUM_INPUTS > 1
    constexpr auto args1 = TensorAccessorArgs<args0.next_compile_time_args_offset()>();
    const auto in1 = TensorAccessor(args1, arg(1, 0), get_tile_size(1));
#endif
#if NUM_INPUTS > 2
    constexpr auto args2 = TensorAccessorArgs<args1.next_compile_time_args_offset()>();
    const auto in2 = TensorAccessor(args2, arg(2, 0), get_tile_size(2));
#endif
#if NUM_INPUTS > 3
    constexpr auto args3 = TensorAccessorArgs<args2.next_compile_time_args_offset()>();
    const auto in3 = TensorAccessor(args3, arg(3, 0), get_tile_size(3));
#endif

    const uint32_t end = start + count;
    const uint32_t first_row = start / ct, last_row = (end - 1) / ct;
    const uint32_t cap = 2 * block;  // tiles in a streamed input's CB
    uint32_t filled[4] = {0, 0, 0, 0};
    for (uint32_t c = 0; c < ct; ++c) {
        // This column's tile rows within [start, end).
        const uint32_t gr0 = first_row * ct + c >= start ? first_row : first_row + 1;
        const uint32_t gr1 = last_row * ct + c < end ? last_row : last_row - 1;
        if (gr1 < gr0 || gr1 == UINT32_MAX) {
            continue;
        }
#define KTT_PERSIST(i)                                             \
    if (i < NUM_INPUTS && persistent(arg(i, 1), arg(i, 2))) {      \
        read_one(in##i, i, page_of(arg(i, 1), 0, gr0, c, rt, ct)); \
    }
        KTT_PERSIST(0)
#if NUM_INPUTS > 1
        KTT_PERSIST(1)
#endif
#if NUM_INPUTS > 2
        KTT_PERSIST(2)
#endif
#if NUM_INPUTS > 3
        KTT_PERSIST(3)
#endif
        for (uint32_t gr = gr0; gr <= gr1; gr += block) {
            const uint32_t n = gr1 + 1 - gr < block ? gr1 + 1 - gr : block;
#define KTT_STREAM(i)                                                               \
    if (!persistent(arg(i, 1), arg(i, 2))) {                                        \
        read_run(in##i, i, cap, filled[i], arg(i, 1), arg(i, 2), gr, n, c, rt, ct); \
    }
            KTT_STREAM(0)
#if NUM_INPUTS > 1
            KTT_STREAM(1)
#endif
#if NUM_INPUTS > 2
            KTT_STREAM(2)
#endif
#if NUM_INPUTS > 3
            KTT_STREAM(3)
#endif
        }
    }
}
