// Weight gradient in the torch layout from the chunk sums (fp32 [B, 1024], flat): per
// output tile, (o, q) of TILE [O, Cg, KH, KW] bf16 with element (t / KW, t mod KW), or
// in 1-D mode (o, q tile) of TILE [O, Cg, KW] with element (q mod 32, t), each value
// read from the sums and rounded to bf16 (nearest even). Sum of weight (o, q, t):
// packed (Cg < 32), element ((j taps + t) 32 + o mod 32) Cg + q; else element
// face (o mod 32, c mod 32) of sum tile (j taps + t) Kmax + c / 32 - k0(j), for j =
// o / 32 and input channel c = group(o) Cg + q.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "../ktt_conv/args.hpp"

namespace {

    constexpr uint32_t kTileBytes = 2048;

    inline uint32_t face_offset(uint32_t r, uint32_t c) {
        return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
    }

    inline void channel_tiles(
        uint32_t j, uint32_t cg, uint32_t og, uint32_t o, uint32_t& k0, uint32_t& k1
    ) {
        const uint32_t lo = 32 * j, hi = 32 * j + 31 < o ? 32 * j + 31 : o - 1;
        k0 = (lo / og) * cg / 32;
        k1 = ((hi / og + 1) * cg - 1) / 32 + 1;
    }

    inline uint16_t to_bf16(uint32_t bits) {
        return static_cast<uint16_t>((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of the sums, then of the output, then the
// argument-layout version. Common runtime args: version, sums address, output address,
// unit step, packed (0 / 1), 1-D (0 / 1), KH, KW, Cg, Og, O, Kmax. Per core: first
// output tile. Scratch: CB 0 holds taps 64-byte blocks, CB 1 one output tile.
void kernel_main() {
    constexpr auto s_args = TensorAccessorArgs<0>();
    constexpr auto out_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    if (get_compile_time_arg_val(out_args.next_compile_time_args_offset()) !=
            ktt_conv::kArgsVersion ||
        get_common_arg_val<uint32_t>(0) != ktt_conv::kArgsVersion) {
        return;
    }
    uint32_t a = 1;
    const uint32_t sums_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t step = get_common_arg_val<uint32_t>(a++);
    const uint32_t packed = get_common_arg_val<uint32_t>(a++);
    const uint32_t one_d = get_common_arg_val<uint32_t>(a++);
    const uint32_t kh = get_common_arg_val<uint32_t>(a++);
    const uint32_t kw = get_common_arg_val<uint32_t>(a++);
    const uint32_t cg = get_common_arg_val<uint32_t>(a++);
    const uint32_t og = get_common_arg_val<uint32_t>(a++);
    const uint32_t o = get_common_arg_val<uint32_t>(a++);
    const uint32_t kmax = get_common_arg_val<uint32_t>(a++);
    const uint32_t start = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_blocks = 0, cb_out = 1;
    const auto sums = TensorAccessor(s_args, sums_addr, 4096);
    const auto out = TensorAccessor(out_args, out_addr, kTileBytes);
    cb_reserve_back(cb_blocks, 1);
    cb_reserve_back(cb_out, 1);
    const uint32_t blocks = (get_write_ptr(cb_blocks) + 63) & ~63u;
    auto* tile = reinterpret_cast<uint16_t*>(get_write_ptr(cb_out));
    const uint32_t taps = kh * kw;

    // Flat element of the sum of weight (oc, q, t).
    const auto element = [&](uint32_t oc, uint32_t q, uint32_t t) {
        const uint32_t j = oc / 32;
        if (packed) {
            return ((j * taps + t) * 32 + oc % 32) * cg + q;
        }
        uint32_t k0, k1;
        channel_tiles(j, cg, og, o, k0, k1);
        const uint32_t c = (oc / og) * cg + q;
        return ((j * taps + t) * kmax + c / 32 - k0) * 1024 + face_offset(oc % 32, c % 32);
    };

    const uint32_t q_tiles = (cg + 31) / 32;
    const uint32_t count = one_d ? o * q_tiles : o * cg;
    for (uint32_t p = start; p < count; p += step) {
        const uint32_t oc = one_d ? p / q_tiles : p / cg;
        const uint32_t q0 = one_d ? (p % q_tiles) * 32 : p % cg;
        const uint32_t q1 = one_d ? (q0 + 32 < cg ? q0 + 32 : cg) : q0 + 1;
        for (uint32_t e = 0; e < kTileBytes / 4; ++e) {
            reinterpret_cast<uint32_t*>(tile)[e] = 0;
        }
        for (uint32_t q = q0; q < q1; ++q) {
            for (uint32_t t = 0; t < taps; ++t) {
                const uint32_t e = element(oc, q, t);
                noc_async_read(
                    sums.get_noc_addr(e / 1024, (e % 1024) * 4 & ~63u),
                    blocks + t * 64,
                    64
                );
            }
            noc_async_read_barrier();
            for (uint32_t t = 0; t < taps; ++t) {
                const uint32_t e = element(oc, q, t);
                const uint32_t bits = reinterpret_cast<const uint32_t*>(
                    blocks + t * 64
                )[(e % 16)];
                const uint32_t at = (
                    one_d ? face_offset(q % 32, t) : face_offset(t / kw, t % kw)
                );
                tile[at] = to_bf16(bits);
            }
        }
        noc_async_write_page(p, out, reinterpret_cast<uint32_t>(tile));
        noc_async_write_barrier();
    }
}
