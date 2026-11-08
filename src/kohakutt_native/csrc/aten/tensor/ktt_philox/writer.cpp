// Philox writer: per output tile, each element's uniform in [0, 1) (24 bits) from word
// `word` of Philox4x32-10 at counter (its global row-major index, 0) under the key;
// tile padding and elements beyond the global sizes are 0.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    constexpr uint32_t kM0 = 0xD2511F53u, kM1 = 0xCD9E8D57u;
    constexpr uint32_t kW0 = 0x9E3779B9u, kW1 = 0xBB67AE85u;
    constexpr uint32_t kMaxRank = 8;

    inline uint32_t mulhi(uint32_t a, uint32_t b) {
        return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32);
    }

    // Word `word` of Philox4x32-10 at counter (c0, c1, 0, 0) under key (k0, k1).
    inline uint32_t philox(
        uint32_t c0, uint32_t c1, uint32_t k0, uint32_t k1, uint32_t word
    ) {
        uint32_t x0 = c0, x1 = c1, x2 = 0, x3 = 0;
        for (uint32_t round = 0; round < 10; ++round) {
            const uint32_t h0 = mulhi(kM0, x0), l0 = kM0 * x0;
            const uint32_t h1 = mulhi(kM1, x2), l1 = kM1 * x2;
            x0 = h1 ^ x1 ^ k0;
            x1 = l1;
            x2 = h0 ^ x3 ^ k1;
            x3 = l0;
            k0 += kW0;
            k1 += kW1;
        }
        switch (word) {
            case 0:
                return x0;
            case 1:
                return x1;
            case 2:
                return x2;
            default:
                return x3;
        }
    }

    // fp32 bits of (u >> 8) * 2^-24.
    inline uint32_t unit_float(uint32_t u) {
        const uint32_t m = u >> 8;
        if (m == 0) {
            return 0;
        }
        const uint32_t e = 31 - static_cast<uint32_t>(__builtin_clz(m));
        return ((e + 127 - 24) << 23) | ((m << (23 - e)) & 0x7FFFFFu);
    }

    // Element offset of (r, c) in a 32x32 tile stored as four 16x16 faces.
    inline uint32_t face_offset(uint32_t r, uint32_t c) {
        return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of the output, then of the params. Common
// runtime args: output address, params address, tile step, tiles, element bytes (2 /
// 4), rank R, local sizes [R], global sizes [R], global strides [R] as (low, high)
// words; per core: first tile. Params (int32 TILE, this chip's row): key low, key high,
// word, global offsets [R].
void kernel_main() {
    uint32_t a = 0;
    const uint32_t out_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t params_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t step = get_common_arg_val<uint32_t>(a++);
    const uint32_t tiles = get_common_arg_val<uint32_t>(a++);
    const uint32_t elem = get_common_arg_val<uint32_t>(a++);
    const uint32_t rank = get_common_arg_val<uint32_t>(a++);
    uint32_t local[kMaxRank], global[kMaxRank], off[kMaxRank];
    uint64_t stride[kMaxRank];
    for (uint32_t k = 0; k < rank; ++k) {
        local[k] = get_common_arg_val<uint32_t>(a++);
    }
    for (uint32_t k = 0; k < rank; ++k) {
        global[k] = get_common_arg_val<uint32_t>(a++);
    }
    for (uint32_t k = 0; k < rank; ++k) {
        const uint32_t lo = get_common_arg_val<uint32_t>(a++);
        const uint32_t hi = get_common_arg_val<uint32_t>(a++);
        stride[k] = (static_cast<uint64_t>(hi) << 32) | lo;
    }
    const uint32_t start = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_params = 0, cb_out = 1;
    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr auto params_args = (
        TensorAccessorArgs<out_args.next_compile_time_args_offset()>()
    );
    const auto out = TensorAccessor(out_args, out_addr, 1024 * elem);
    const auto params = TensorAccessor(params_args, params_addr, 4096);

    cb_reserve_back(cb_params, 1);
    cb_reserve_back(cb_out, 1);
    const uint32_t params_l1 = get_write_ptr(cb_params);
    const uint32_t out_l1 = get_write_ptr(cb_out);
    noc_async_read_page(0, params, params_l1);
    noc_async_read_barrier();
    const auto* p = reinterpret_cast<volatile uint32_t*>(params_l1);
    const uint32_t key0 = p[0], key1 = p[1], word = p[2];
    for (uint32_t k = 0; k < rank; ++k) {
        off[k] = p[3 + k];
    }

    const uint32_t hr = rank - 2, wr = rank - 1;
    const uint32_t ht = (local[hr] + 31) / 32, wt = (local[wr] + 31) / 32;
    auto* out32 = reinterpret_cast<volatile uint32_t*>(out_l1);
    auto* out16 = reinterpret_cast<volatile uint16_t*>(out_l1);

    for (uint32_t u = start; u < tiles; u += step) {
        const uint32_t lead = u / (ht * wt), rem = u % (ht * wt);
        const uint32_t th = rem / wt, tw = rem % wt;
        uint64_t base = 0;
        bool inside = true;
        uint32_t q = lead;
        for (uint32_t k = hr; k-- > 0;) {
            const uint32_t g = q % local[k] + off[k];
            q /= local[k];
            inside = inside && g < global[k];
            base += static_cast<uint64_t>(g) * stride[k];
        }
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t i = th * 32 + r, gi = i + off[hr];
            const bool row = inside && i < local[hr] && gi < global[hr];
            const uint64_t row_base = base + static_cast<uint64_t>(gi) * stride[hr];
            for (uint32_t c = 0; c < 32; ++c) {
                const uint32_t j = tw * 32 + c, gj = j + off[wr];
                uint32_t bits = 0;
                if (row && j < local[wr] && gj < global[wr]) {
                    const uint64_t index = (
                        row_base + static_cast<uint64_t>(gj) * stride[wr]
                    );
                    bits = (
                        unit_float(philox(
                            static_cast<uint32_t>(index),
                            static_cast<uint32_t>(index >> 32),
                            key0,
                            key1,
                            word
                        ))
                    );
                }
                if (elem == 4) {
                    out32[face_offset(r, c)] = bits;
                } else {
                    out16[face_offset(r, c)] = static_cast<uint16_t>(bits >> 16);
                }
            }
        }
        noc_async_write_page(u, out, out_l1);
        noc_async_write_barrier();
    }
}
