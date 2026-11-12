// Group norm statistics: per (n, group) unit, dealt over the cores, mean and rstd from
// the planes' sums s1 / s2 and per plane A = rstd * w, B = bias - mean * A; core 0
// collects every unit's mean / rstd and writes them as [N, G] tiles.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    constexpr uint32_t kChunk = 64, kSlot = 64;

    // Element offset of (r, c) in a 32x32 tile stored as four 16x16 faces.
    inline uint32_t face_offset(uint32_t r, uint32_t c) {
        return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
    }

    inline float as_float(uint32_t u) {
        return __builtin_bit_cast(float, u);
    }

    inline uint32_t as_bits(float f) {
        return __builtin_bit_cast(uint32_t, f);
    }

    // 1 / sqrt(x) for x > 0: bit-level guess, then Newton steps in double.
    double rsqrt(double x) {
        const float xf = static_cast<float>(x);
        double y = as_float(0x5F3759DFu - (as_bits(xf) >> 1));
        for (int i = 0; i < 5; ++i) {
            y = y * (1.5 - 0.5 * x * y * y);
        }
        return y;
    }

    // Value `v` stored as element `index` of a tile of `elem` bytes per element.
    void store(uint32_t tile, uint32_t index, float v, uint32_t elem) {
        if (elem == 4) {
            reinterpret_cast<volatile float*>(tile)[index] = v;
            return;
        }
        uint32_t u = as_bits(v);
        u += 0x7FFFu + ((u >> 16) & 1u);
        reinterpret_cast<volatile uint16_t*>(tile)[index] = static_cast<uint16_t>(
            u >> 16
        );
    }

    // Reads the 64 B segment holding channel `ch` of a [1, C] TILE vector of
    // `elem`-byte values (row 0 of its tile) to `l1` (no barrier).
    template <typename Accessor>
    void read_channel(const Accessor& v, uint32_t ch, uint32_t elem, uint32_t l1) {
        const uint32_t offset = face_offset(0, ch % 32) * elem;
        noc_async_read(v.get_noc_addr(ch / 32, offset & ~63u), l1, 64);
    }

    // The value of channel `ch` in the segment read_channel() put at `l1`.
    float channel_value(uint32_t ch, uint32_t elem, uint32_t l1) {
        const uint32_t at = l1 + (face_offset(0, ch % 32) * elem & 63u);
        if (elem == 4) {
            return *reinterpret_cast<volatile float*>(at);
        }
        return (as_float(
            static_cast<uint32_t>(*reinterpret_cast<volatile uint16_t*>(at)) << 16
        ));
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of s1, s2, weight, bias, a, b, mean, rstd.
// Runtime args: s1, s2, weight, bias, a, b, mean, rstd addresses, N, C, groups, count
// (elements per group) bits as float, eps bits, weight / bias / output element bytes
// (weight, bias: 0 when absent), then per core: its index, the core count, NoC x / y of
// core 0. Semaphore 0: core 0 counts the cores whose parts arrived.
void kernel_main() {
    uint32_t i = 0;
    const uint32_t s1_addr = get_arg_val<uint32_t>(i++);
    const uint32_t s2_addr = get_arg_val<uint32_t>(i++);
    const uint32_t w_addr = get_arg_val<uint32_t>(i++);
    const uint32_t bias_addr = get_arg_val<uint32_t>(i++);
    const uint32_t a_addr = get_arg_val<uint32_t>(i++);
    const uint32_t b_addr = get_arg_val<uint32_t>(i++);
    const uint32_t mean_addr = get_arg_val<uint32_t>(i++);
    const uint32_t rstd_addr = get_arg_val<uint32_t>(i++);
    const uint32_t n = get_arg_val<uint32_t>(i++);
    const uint32_t c = get_arg_val<uint32_t>(i++);
    const uint32_t groups = get_arg_val<uint32_t>(i++);
    const double count = as_float(get_arg_val<uint32_t>(i++));
    const double eps = as_float(get_arg_val<uint32_t>(i++));
    const uint32_t w_elem = get_arg_val<uint32_t>(i++);
    const uint32_t bias_elem = get_arg_val<uint32_t>(i++);
    const uint32_t out_elem = get_arg_val<uint32_t>(i++);
    const uint32_t core = get_arg_val<uint32_t>(i++);
    const uint32_t cores = get_arg_val<uint32_t>(i++);
    const uint32_t head_x = get_arg_val<uint32_t>(i++);
    const uint32_t head_y = get_arg_val<uint32_t>(i++);

    constexpr auto s1_args = TensorAccessorArgs<0>();
    constexpr auto
        s2_args = (TensorAccessorArgs<s1_args.next_compile_time_args_offset()>());
    constexpr auto
        w_args = (TensorAccessorArgs<s2_args.next_compile_time_args_offset()>());
    constexpr auto
        bias_args = (TensorAccessorArgs<w_args.next_compile_time_args_offset()>());
    constexpr auto
        a_args = (TensorAccessorArgs<bias_args.next_compile_time_args_offset()>());
    constexpr auto
        b_args = (TensorAccessorArgs<a_args.next_compile_time_args_offset()>());
    constexpr auto
        mean_args = (TensorAccessorArgs<b_args.next_compile_time_args_offset()>());
    constexpr auto
        rstd_args = (TensorAccessorArgs<mean_args.next_compile_time_args_offset()>());
    const auto s1 = TensorAccessor(s1_args, s1_addr, 4096);
    const auto s2 = TensorAccessor(s2_args, s2_addr, 4096);
    const auto wv = TensorAccessor(w_args, w_addr, 1024 * (w_elem ? w_elem : 4));
    const auto bv = (TensorAccessor(
        bias_args, bias_addr, 1024 * (bias_elem ? bias_elem : 4)
    ));
    const auto av = TensorAccessor(a_args, a_addr, 4096);
    const auto bo = TensorAccessor(b_args, b_addr, 4096);
    const auto mean_out = TensorAccessor(mean_args, mean_addr, 1024 * out_elem);
    const auto rstd_out = TensorAccessor(rstd_args, rstd_addr, 1024 * out_elem);

    constexpr uint32_t cb_sums = 0, cb_vec = 1, cb_ab = 2, cb_mean = 3, cb_rstd = 4,
                       cb_parts = 5;
    cb_reserve_back(cb_sums, 1);
    cb_reserve_back(cb_vec, 1);
    cb_reserve_back(cb_ab, 1);
    cb_reserve_back(cb_mean, 1);
    cb_reserve_back(cb_rstd, 1);
    cb_reserve_back(cb_parts, 1);
    const uint32_t sums = get_write_ptr(cb_sums);  // 2 x kChunk slots
    const uint32_t vec = get_write_ptr(cb_vec);    // 2 x kChunk slots: weight, bias
    const uint32_t ab = get_write_ptr(cb_ab);      // 2 x kChunk slots
    const uint32_t mean_tile = get_write_ptr(cb_mean);
    const uint32_t rstd_tile = get_write_ptr(cb_rstd);
    const uint32_t parts = get_write_ptr(cb_parts);  // (mean, rstd) per unit, 16 B

    const uint32_t cg = c / groups;
    for (uint32_t u = core; u < n * groups; u += cores) {
        const uint32_t img = u / groups, g = u % groups;
        const uint32_t first = img * c + g * cg;
        double t1 = 0, t2 = 0;
        for (uint32_t base = 0; base < cg; base += kChunk) {
            const uint32_t m = cg - base < kChunk ? cg - base : kChunk;
            for (uint32_t k = 0; k < m; ++k) {
                noc_async_read(
                    s1.get_noc_addr(first + base + k), sums + k * kSlot, kSlot
                );
                noc_async_read(
                    s2.get_noc_addr(first + base + k),
                    sums + (kChunk + k) * kSlot,
                    kSlot
                );
            }
            noc_async_read_barrier();
            for (uint32_t k = 0; k < m; ++k) {
                t1 += *reinterpret_cast<volatile float*>(sums + k * kSlot);
                t2 += *reinterpret_cast<volatile float*>(sums + (kChunk + k) * kSlot);
            }
        }
        const double mean = t1 / count;
        double var = t2 / count - mean * mean;
        var = var > 0 ? var : 0;
        const double rs = rsqrt(var + eps);
        auto* part = reinterpret_cast<volatile float*>(parts + u * 16);
        part[0] = static_cast<float>(mean);
        part[1] = static_cast<float>(rs);
        for (uint32_t base = 0; base < cg; base += kChunk) {
            const uint32_t m = cg - base < kChunk ? cg - base : kChunk;
            for (uint32_t k = 0; k < m; ++k) {
                const uint32_t ch = g * cg + base + k;
                if (w_elem) {
                    read_channel(wv, ch, w_elem, vec + k * kSlot);
                }
                if (bias_elem) {
                    read_channel(bv, ch, bias_elem, vec + (kChunk + k) * kSlot);
                }
            }
            noc_async_read_barrier();
            for (uint32_t k = 0; k < m; ++k) {
                const uint32_t ch = g * cg + base + k;
                const double w = w_elem ? channel_value(ch, w_elem, vec + k * kSlot)
                                        : 1.0;
                const double bias =
                    (bias_elem
                         ? channel_value(ch, bias_elem, vec + (kChunk + k) * kSlot)
                         : 0.0);
                const float a = static_cast<float>(rs * w);
                const float b = static_cast<float>(bias - mean * a);
                *reinterpret_cast<volatile float*>(ab + k * kSlot) = a;
                *reinterpret_cast<volatile float*>(ab + (kChunk + k) * kSlot) = b;
            }
            for (uint32_t k = 0; k < m; ++k) {
                noc_async_write(
                    ab + k * kSlot, av.get_noc_addr(first + base + k), kSlot
                );
                noc_async_write(
                    ab + (kChunk + k) * kSlot, bo.get_noc_addr(first + base + k), kSlot
                );
            }
            noc_async_write_barrier();
        }
    }

    if (core != 0) {
        // This core's parts to the same place on core 0, then its count there.
        for (uint32_t u = core; u < n * groups; u += cores) {
            noc_async_write(
                parts + u * 16, get_noc_addr(head_x, head_y, parts + u * 16), 16
            );
        }
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(head_x, head_y, get_semaphore(0)), 1);
        noc_async_atomic_barrier();
        return;
    }
    if (cores > 1) {
        noc_semaphore_wait_min(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), cores - 1
        );
        invalidate_l1_cache();
    }
    const uint32_t nt = (n + 31) / 32, gt = (groups + 31) / 32;
    for (uint32_t ti = 0; ti < nt; ++ti) {
        for (uint32_t tj = 0; tj < gt; ++tj) {
            for (uint32_t k = 0; k < 1024 * out_elem / 4; ++k) {
                reinterpret_cast<volatile uint32_t*>(mean_tile)[k] = 0;
                reinterpret_cast<volatile uint32_t*>(rstd_tile)[k] = 0;
            }
            for (uint32_t r = 0; r < 32 && ti * 32 + r < n; ++r) {
                for (uint32_t col = 0; col < 32 && tj * 32 + col < groups; ++col) {
                    const uint32_t u = (ti * 32 + r) * groups + tj * 32 + col;
                    const auto* part = reinterpret_cast<volatile float*>(
                        parts + u * 16
                    );
                    store(mean_tile, face_offset(r, col), part[0], out_elem);
                    store(rstd_tile, face_offset(r, col), part[1], out_elem);
                }
            }
            noc_async_write_page(ti * gt + tj, mean_out, mean_tile);
            noc_async_write_page(ti * gt + tj, rstd_out, rstd_tile);
            noc_async_write_barrier();
        }
    }
}
