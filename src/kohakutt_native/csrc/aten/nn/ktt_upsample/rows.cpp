// Nearest upsampling by integer factors as row copies on both data-movement RISCs: an
// output row is its source row (pixels repeated kw times), built once per kh rows.
#include <stdint.h>

#include <algorithm>

#include "api/dataflow/dataflow_api.h"

namespace {

    constexpr uint32_t kSlots = SLOTS;  // input cache tiles (4, or 2 when L1 is short)

    // Element offset of (r, c) in a 32x32 tile stored as four 16x16 faces.
    inline uint32_t face_offset(uint32_t r, uint32_t c) {
        return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
    }

    // The L1 tile cache: kSlots input tiles, replaced round robin; prefetched tiles
    // are waited for on first use.
    struct Cache {
        uint32_t base, tile_bytes, next = 0, pending = 0;
        uint32_t ids[kSlots];

        Cache(uint32_t b, uint32_t t) : base(b), tile_bytes(t) {
            for (uint32_t s = 0; s < kSlots; ++s) {
                ids[s] = 0xFFFFFFFFu;
            }
        }

        uint32_t find(uint32_t id) const {
            for (uint32_t s = 0; s < kSlots; ++s) {
                if (ids[s] == id) {
                    return s;
                }
            }
            return kSlots;
        }

        template <typename Accessor>
        uint32_t fetch(const Accessor& in, uint32_t id) {
            const uint32_t s = next;
            next = (next + 1) % kSlots;
            if (pending & (1u << s)) {
                noc_async_read_barrier();
                pending = 0;
            }
            ids[s] = id;
            noc_async_read_page(id, in, base + s * tile_bytes);
            return s;
        }

        template <typename Accessor>
        uint32_t get(const Accessor& in, uint32_t id) {
            uint32_t s = find(id);
            if (s == kSlots) {
                s = fetch(in, id);
                pending |= 1u << s;
            }
            if (pending & (1u << s)) {
                noc_async_read_barrier();
                pending = 0;
            }
            return base + s * tile_bytes;
        }

        template <typename Accessor>
        void prefetch(const Accessor& in, uint32_t id) {
            if (find(id) == kSlots) {
                pending |= 1u << fetch(in, id);
            }
        }
    };

    inline void zero_row(uint32_t dst, uint32_t to, uint32_t elem) {
        const uint32_t words = 4 * elem;
        for (uint32_t half = 0; half < 2; ++half) {
            auto* d = reinterpret_cast<uint32_t*>(
                dst + face_offset(to, half * 16) * elem
            );
            for (uint32_t k = 0; k < words; ++k) {
                d[k] = 0;
            }
        }
    }

    // Element `at` of a tile at `src` (bf16 or fp32) as a 32-bit word.
    inline uint32_t element(uint32_t src, uint32_t at, uint32_t elem) {
        return elem == 4 ? reinterpret_cast<const uint32_t*>(src)[at]
                         : reinterpret_cast<const uint16_t*>(src)[at];
    }

    // One face row (16 elements, W words: 8 bf16, 16 fp32) held in registers.
    template <uint32_t W>
    struct FaceRow {
        static constexpr uint32_t kElem = W / 4;
        uint32_t v[W];

        void load(uint32_t addr) {
            const auto* s = reinterpret_cast<const uint32_t*>(addr);
#pragma GCC unroll 16
            for (uint32_t k = 0; k < W; ++k) {
                v[k] = s[k];
            }
        }

        // The 8 elements at `addr`, each twice.
        void widen2(uint32_t addr) {
            const auto* s = reinterpret_cast<const uint32_t*>(addr);
            if constexpr (W == 8) {
                uint32_t x[4];
#pragma GCC unroll 4
                for (uint32_t k = 0; k < 4; ++k) {
                    x[k] = s[k];
                }
#pragma GCC unroll 4
                for (uint32_t k = 0; k < 4; ++k) {
                    const uint32_t lo = x[k] & 0xFFFFu, hi = x[k] >> 16;
                    v[2 * k] = lo | (lo << 16);
                    v[2 * k + 1] = hi | (hi << 16);
                }
            } else {
#pragma GCC unroll 8
                for (uint32_t k = 0; k < 8; ++k) {
                    v[2 * k] = v[2 * k + 1] = s[k];
                }
            }
        }

        // Element c from column (co + c) / kw of row `sr` of the tile at `src`.
        void widen(uint32_t src, uint32_t sr, uint32_t co, uint32_t kw) {
            for (uint32_t c = 0; c < 16; ++c) {
                const uint32_t x = element(src, face_offset(sr, (co + c) / kw), kElem);
                if constexpr (W == 16) {
                    v[c] = x;
                } else if (c & 1) {
                    v[c / 2] |= x << 16;
                } else {
                    v[c / 2] = x;
                }
            }
        }

        // To face row `half` of `n` rows from row `to` of the tile at `dst`.
        void store(uint32_t dst, uint32_t to, uint32_t n, uint32_t half) const {
            for (uint32_t i = 0; i < n; ++i) {
                auto* d = reinterpret_cast<uint32_t*>(
                    dst + face_offset(to + i, half * 16) * kElem
                );
#pragma GCC unroll 16
                for (uint32_t k = 0; k < W; ++k) {
                    d[k] = v[k];
                }
            }
        }
    };

    // Output tile (a, b) of the input tile at `src` (32 % kh == 0, 32 % kw == 0),
    // its first row at output row `top`; rows from `ho` on are zero.
    template <uint32_t W>
    void build_tile(
        uint32_t src,
        uint32_t a,
        uint32_t b,
        uint32_t top,
        uint32_t kh,
        uint32_t kw,
        uint32_t ho,
        uint32_t o
    ) {
        constexpr uint32_t elem = W / 4;
        for (uint32_t r = 0; r < 32;) {
            if (top + r >= ho) {
                zero_row(o, r++, elem);
                continue;
            }
            const uint32_t n = std::min(
                std::min(kh - (a * 32 + r) % kh, 32 - r), ho - top - r
            );
            const uint32_t sr = (a * 32 + r) / kh;
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t co = b * 32 + half * 16;
                FaceRow<W> f;
                if (kw == 1) {
                    f.load(src + face_offset(sr, co) * elem);
                } else if (kw == 2) {
                    f.widen2(src + face_offset(sr, co / 2) * elem);
                } else {
                    f.widen(src, sr, co, kw);
                }
                f.store(o, r, n, half);
            }
            r += n;
        }
    }

    struct Geometry {
        uint32_t kh, kw, h, w, iht, iwt, oht, owt, count;
    };

    // The other slot's writes have left L1, then `o` is written to output tile `id`.
    template <typename Out>
    inline void send(const Out& out, uint32_t id, uint32_t o) {
        asm volatile("" ::: "memory");
        noc_async_writes_flushed();
        noc_async_write_page(id, out, o);
    }

    // Mode 2, planes, 32 % kh == 0 == 32 % kw: unit = input tile, its kh x kw outputs.
    template <uint32_t W, typename In, typename Out>
    void planes_by_tile(
        const Geometry& g,
        const In& in,
        const Out& out,
        Cache& cache,
        uint32_t out_base,
        uint32_t first,
        uint32_t last
    ) {
        constexpr uint32_t elem = W / 4;
        const uint32_t ho = g.h * g.kh, kh = g.kh, kw = g.kw;
        uint32_t slot = 0;
        for (uint32_t u = first; u < last; ++u) {
            const uint32_t src = cache.get(in, u);
            if (u + 1 < last) {
                cache.prefetch(in, u + 1);
            }
            const uint32_t plane = u / (g.iht * g.iwt);
            const uint32_t ih = (u / g.iwt) % g.iht, iw = u % g.iwt;
            for (uint32_t a = 0; a < kh && ih * kh + a < g.oht; ++a) {
                const uint32_t th = ih * kh + a;
                for (uint32_t b = 0; b < kw && iw * kw + b < g.owt; ++b) {
                    const uint32_t o = out_base + slot * 1024 * elem;
                    build_tile<W>(src, a, b, th * 32, kh, kw, ho, o);
                    send(out, (plane * g.oht + th) * g.owt + iw * kw + b, o);
                    slot ^= 1;
                }
            }
        }
    }

    // Mode 3, rows, W kw % 32 == 0: unit = (channel tile, input image row); each tile
    // is built once and written for the kh output rows.
    template <uint32_t W, typename In, typename Out>
    void rows_by_image_row(
        const Geometry& g,
        const In& in,
        const Out& out,
        Cache& cache,
        uint32_t out_base,
        uint32_t first,
        uint32_t last
    ) {
        constexpr uint32_t elem = W / 4;
        const uint32_t kw = g.kw, wo = g.w * kw, rows = g.count * g.h;
        uint32_t slot = 0;
        for (uint32_t u = first; u < last; ++u) {
            const uint32_t ct = u / rows, ir = u % rows;
            const uint32_t q0 = ir * g.w, p0 = ir * g.kh * wo;
            uint32_t touched = 0xFFFFFFFFu;
            for (uint32_t k = 0; k < wo / 32; ++k) {
                const uint32_t o = out_base + slot * 1024 * elem;
                for (uint32_t r = 0; r < 32;) {
                    const uint32_t x = (k * 32 + r) / kw;
                    const uint32_t n = std::min(kw - (k * 32 + r) % kw, 32 - r);
                    const uint32_t q = q0 + x, pt = q / 32;
                    const uint32_t src = cache.get(in, pt * g.iwt + ct);
                    if (pt != touched) {
                        touched = pt;
                        if (pt + 1 < g.iht) {
                            cache.prefetch(in, (pt + 1) * g.iwt + ct);
                        }
                    }
                    for (uint32_t half = 0; half < 2; ++half) {
                        FaceRow<W> f;
                        f.load(src + face_offset(q % 32, half * 16) * elem);
                        f.store(o, r, n, half);
                    }
                    r += n;
                }
                for (uint32_t a = 0; a < g.kh; ++a) {
                    send(out, ((p0 + a * wo) / 32 + k) * g.owt + ct, o);
                }
                slot ^= 1;
            }
        }
    }

    // Mode 0, planes, any factors: unit = output tile; runs of output rows of one
    // source built once.
    template <uint32_t W, typename In, typename Out>
    void planes_any(
        const Geometry& g,
        const In& in,
        const Out& out,
        Cache& cache,
        uint32_t out_base,
        uint32_t first,
        uint32_t last
    ) {
        constexpr uint32_t elem = W / 4;
        const uint32_t ho = g.h * g.kh, wo = g.w * g.kw, kh = g.kh, kw = g.kw;
        for (uint32_t u = first; u < last; ++u) {
            const uint32_t o = out_base + (u & 1) * 1024 * elem;
            const uint32_t plane = u / (g.oht * g.owt);
            const uint32_t th = (u / g.owt) % g.oht, tw = u % g.owt;
            for (uint32_t r = 0; r < 32;) {
                const uint32_t ro = th * 32 + r;
                if (ro >= ho) {
                    zero_row(o, r++, elem);
                    continue;
                }
                const uint32_t s = ro / kh;
                const uint32_t n = std::min(std::min(kh - ro % kh, 32 - r), ho - ro);
                const uint32_t row_base = (plane * g.iht + s / 32) * g.iwt;
                uint32_t tile = 0xFFFFFFFFu, src = 0;
                for (uint32_t half = 0; half < 2; ++half) {
                    FaceRow<W> f;
                    for (uint32_t c = 0; c < 16; ++c) {
                        const uint32_t co = tw * 32 + half * 16 + c, sc = co / kw;
                        uint32_t x = 0;
                        if (co < wo) {
                            if (sc / 32 != tile) {
                                tile = sc / 32;
                                src = cache.get(in, row_base + tile);
                            }
                            x = element(src, face_offset(s % 32, sc % 32), elem);
                        }
                        if constexpr (W == 16) {
                            f.v[c] = x;
                        } else if (c & 1) {
                            f.v[c / 2] |= x << 16;
                        } else {
                            f.v[c / 2] = x;
                        }
                    }
                    f.store(o, r, n, half);
                }
                r += n;
            }
            send(out, u, o);
        }
    }

    // Mode 1, rows, any factors: unit = output tile; output pixel p = (image, h', w')
    // reads input pixel (h' / kh, w' / kw).
    template <uint32_t W, typename In, typename Out>
    void rows_any(
        const Geometry& g,
        const In& in,
        const Out& out,
        Cache& cache,
        uint32_t out_base,
        uint32_t first,
        uint32_t last
    ) {
        constexpr uint32_t elem = W / 4;
        const uint32_t wo = g.w * g.kw, per_out = g.h * g.kh * wo;
        for (uint32_t u = first; u < last; ++u) {
            const uint32_t o = out_base + (u & 1) * 1024 * elem;
            const uint32_t pt = u / g.owt, ct = u % g.owt;
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t p = pt * 32 + r;
                if (p >= g.count * per_out) {
                    zero_row(o, r, elem);
                    continue;
                }
                const uint32_t img = p / per_out, rem = p % per_out;
                const uint32_t q = img * g.h * g.w + (rem / wo / g.kh) * g.w +
                    (rem % wo) / g.kw;
                const uint32_t src = cache.get(in, (q / 32) * g.iwt + ct);
                for (uint32_t half = 0; half < 2; ++half) {
                    FaceRow<W> f;
                    f.load(src + face_offset(q % 32, half * 16) * elem);
                    f.store(o, r, 1, half);
                }
            }
            send(out, u, o);
        }
    }

}  // namespace

// Compile-time args: the input's then the output's TensorAccessorArgs. Runtime args:
// see row_copy() and its callers in program.cpp (common, then the core index).
void kernel_main() {
    uint32_t a = 0;
    const uint32_t in_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t workers = get_common_arg_val<uint32_t>(a++);
    const uint32_t units = get_common_arg_val<uint32_t>(a++);
    const uint32_t mode = get_common_arg_val<uint32_t>(a++);
    const uint32_t elem = get_common_arg_val<uint32_t>(a++);
    const uint32_t kh = get_common_arg_val<uint32_t>(a++);
    const uint32_t kw = get_common_arg_val<uint32_t>(a++);
    const uint32_t h = get_common_arg_val<uint32_t>(a++);
    const uint32_t w = get_common_arg_val<uint32_t>(a++);
    const uint32_t iht = get_common_arg_val<uint32_t>(a++);
    const uint32_t iwt = get_common_arg_val<uint32_t>(a++);
    const uint32_t oht = get_common_arg_val<uint32_t>(a++);
    const uint32_t owt = get_common_arg_val<uint32_t>(a++);
    const uint32_t count = get_common_arg_val<uint32_t>(a++);
    const uint32_t worker = get_arg_val<uint32_t>(0) * 2 + ROLE;

    constexpr uint32_t cb_cache = ROLE * 2, cb_out = ROLE * 2 + 1;
    const uint32_t tile_bytes = 1024 * elem;
    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto
        out_args = (TensorAccessorArgs<in_args.next_compile_time_args_offset()>());
    const auto in = TensorAccessor(in_args, in_addr, tile_bytes);
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    cb_reserve_back(cb_cache, 1);
    cb_reserve_back(cb_out, 1);
    Cache cache{get_write_ptr(cb_cache), tile_bytes};
    const uint32_t out_base = get_write_ptr(cb_out);  // two output tiles

    const uint32_t first = static_cast<uint32_t>(
        (static_cast<uint64_t>(units) * worker) / workers
    );
    const uint32_t last = static_cast<uint32_t>(
        (static_cast<uint64_t>(units) * (worker + 1)) / workers
    );
    const Geometry g{kh, kw, h, w, iht, iwt, oht, owt, count};
    const bool bf16 = elem == 2;
    if (mode == 2) {
        bf16 ? planes_by_tile<8>(g, in, out, cache, out_base, first, last)
             : planes_by_tile<16>(g, in, out, cache, out_base, first, last);
    } else if (mode == 3) {
        bf16 ? rows_by_image_row<8>(g, in, out, cache, out_base, first, last)
             : rows_by_image_row<16>(g, in, out, cache, out_base, first, last);
    } else if (mode == 1) {
        bf16 ? rows_any<8>(g, in, out, cache, out_base, first, last)
             : rows_any<16>(g, in, out, cache, out_base, first, last);
    } else {
        bf16 ? planes_any<8>(g, in, out, cache, out_base, first, last)
             : planes_any<16>(g, in, out, cache, out_base, first, last);
    }
    noc_async_read_barrier();  // a read-ahead past the last unit (mode 3)
    noc_async_write_barrier();
}
