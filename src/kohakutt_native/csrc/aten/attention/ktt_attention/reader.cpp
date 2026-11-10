// Attention reader: the mask tile (0 for keys < valid, -inf past them) and a ones tile
// once; per unit (batch-head, q chunk) the Q chunk, per K chunk the K and V chunks.
// Rows past the sequence in a last Q / K / V tile are zeroed in L1.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    constexpr uint32_t kTileBytes = 2048;

    // Zeroes rows [first, 32) of the bf16 tile at addr.
    void zero_rows(uint32_t addr, uint32_t first) {
        auto* t = reinterpret_cast<volatile uint16_t*>(addr);
        for (uint32_t r = first; r < 32; ++r) {
            const uint32_t face = (r / 16) * 2, row = r % 16;
            for (uint32_t f = face; f < face + 2; ++f) {
                for (uint32_t c = 0; c < 16; ++c) {
                    t[f * 256 + row * 16 + c] = 0;
                }
            }
        }
    }

    // Fills the bf16 tile at addr: `value` where column c >= valid or c > r + diag
    // (diag 32: no diagonal), else `base`.
    void fill_mask(
        uint32_t addr, uint32_t valid, uint32_t diag, uint16_t base, uint16_t value
    ) {
        auto* t = reinterpret_cast<volatile uint16_t*>(addr);
        for (uint32_t r = 0; r < 32; ++r) {
            for (uint32_t c = 0; c < 32; ++c) {
                const uint32_t f = (r / 16) * 2 + c / 16;
                const bool masked = c >= valid || c > r + diag;
                t[f * 256 + (r % 16) * 16 + c % 16] = masked ? value : base;
            }
        }
    }

    // Reads row_tiles x cols tiles from page `first` into a `slot`-tile push of CB
    // `cb`: tile (r, c) at r cols + c, or at c stride + r when `stride` (transposed
    // order) is nonzero; zeroes rows from valid_rows of row tile `ragged_tile` (>=
    // row_tiles: none).
    template <typename Accessor>
    void read_block(
        const Accessor& src,
        uint32_t cb,
        uint32_t slot,
        uint32_t first,
        uint32_t row_tiles,
        uint32_t cols,
        uint32_t ragged_tile,
        uint32_t valid_rows,
        uint32_t stride = 0
    ) {
        cb_reserve_back(cb, slot);
        const uint32_t base = get_write_ptr(cb);
        auto at = [&](uint32_t r, uint32_t c) {
            return base + (stride ? c * stride + r : r * cols + c) * kTileBytes;
        };
        for (uint32_t r = 0; r < row_tiles; ++r) {
            for (uint32_t c = 0; c < cols; ++c) {
                noc_async_read_page(first + r * cols + c, src, at(r, c));
            }
        }
        noc_async_read_barrier();
        if (ragged_tile < row_tiles) {
            for (uint32_t c = 0; c < cols; ++c) {
                zero_rows(at(ragged_tile, c), valid_rows);
            }
        }
        cb_push_back(cb, slot);
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of q, k, v. Common runtime args: q / k / v
// addresses, unit step, units, heads, heads per K / V head, Sq tiles, Sk tiles, D
// tiles, Dv tiles, q chunk tiles, k chunk tiles, valid rows of the last Q tile, valid
// keys of the last K tile (32 when full), causal (key tiles past the diagonal are not
// read); per core: first unit. K chunks are stored transposed: tile (key j, d) at
// d kc + j.
void kernel_main() {
    uint32_t a = 0;
    const uint32_t q_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t k_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t v_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t step = get_common_arg_val<uint32_t>(a++);
    const uint32_t units = get_common_arg_val<uint32_t>(a++);
    const uint32_t heads = get_common_arg_val<uint32_t>(a++);
    const uint32_t group = get_common_arg_val<uint32_t>(a++);
    const uint32_t sqt = get_common_arg_val<uint32_t>(a++);
    const uint32_t skt = get_common_arg_val<uint32_t>(a++);
    const uint32_t dt = get_common_arg_val<uint32_t>(a++);
    const uint32_t dvt = get_common_arg_val<uint32_t>(a++);
    const uint32_t qc = get_common_arg_val<uint32_t>(a++);
    const uint32_t kc = get_common_arg_val<uint32_t>(a++);
    const uint32_t q_valid = get_common_arg_val<uint32_t>(a++);
    const uint32_t k_valid = get_common_arg_val<uint32_t>(a++);
    const bool causal = get_common_arg_val<uint32_t>(a++) != 0;
    const uint32_t start = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_mask = 3, cb_ones = 4;
    constexpr auto q_args = TensorAccessorArgs<0>();
    constexpr auto
        k_args = (TensorAccessorArgs<q_args.next_compile_time_args_offset()>());
    constexpr auto
        v_args = (TensorAccessorArgs<k_args.next_compile_time_args_offset()>());
    const auto q = TensorAccessor(q_args, q_addr, kTileBytes);
    const auto k = TensorAccessor(k_args, k_addr, kTileBytes);
    const auto v = TensorAccessor(v_args, v_addr, kTileBytes);

    // Masks: ragged last key tile, causal diagonal tile, all -inf.
    constexpr uint16_t kZero = 0x0000, kOne = 0x3F80, kNegInf = 0xFF80;
    cb_reserve_back(cb_mask, 3);
    const uint32_t masks = get_write_ptr(cb_mask);
    fill_mask(masks, k_valid, 32, kZero, kNegInf);
    fill_mask(masks + kTileBytes, 32, 0, kZero, kNegInf);
    fill_mask(masks + 2 * kTileBytes, 0, 32, kZero, kNegInf);
    cb_push_back(cb_mask, 3);
    cb_reserve_back(cb_ones, 1);
    fill_mask(get_write_ptr(cb_ones), 32, 32, kOne, kOne);
    cb_push_back(cb_ones, 1);

    const uint32_t q_chunks = (sqt + qc - 1) / qc;
    for (uint32_t u = start; u < units; u += step) {
        const uint32_t bh = u / q_chunks, chunk = u % q_chunks;
        const uint32_t b = bh / heads, h = bh % heads;
        const uint32_t kv_bh = b * (heads / group) + h / group;
        const uint32_t q0 = chunk * qc, nq = sqt - q0 < qc ? sqt - q0 : qc;
        // The ragged last Q tile, as an index within this chunk (>= nq: none).
        const uint32_t q_ragged = q_valid < 32 && q0 + nq == sqt ? nq - 1 : nq;
        read_block(q, cb_q, qc * dt, (bh * sqt + q0) * dt, nq, dt, q_ragged, q_valid);
        const uint32_t k_end = causal && q0 + nq < skt ? q0 + nq : skt;
        for (uint32_t k0 = 0; k0 < k_end; k0 += kc) {
            const uint32_t nk = k_end - k0 < kc ? k_end - k0 : kc;
            const uint32_t k_ragged = k_valid < 32 && k0 + nk == skt ? nk - 1 : nk;
            read_block(
                k, cb_k, kc * dt, (kv_bh * skt + k0) * dt, nk, dt, k_ragged, k_valid, kc
            );
            read_block(
                v, cb_v, kc * dvt, (kv_bh * skt + k0) * dvt, nk, dvt, k_ragged, k_valid
            );
        }
    }
}
