// Packed variable-length attention: the per-row plan and its mask tiles from the
// cumulative sequence offsets (no TT-NN or torch headers).
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ktt::bridge::ktt_flash::varlen {

    // The logit (scale x score) a mask tile gives a key a query does not see. A query
    // tile row can meet key tiles in which some of its tokens see no key at all; their
    // running softmax reference is then this value until their first visible key, and
    // it has to stay small enough for fp32 to carry the reference exactly (a -1e30 mask
    // does not). Scores must stay well below its magnitude.
    constexpr float kMaskedLogit = -5e5f;

    // One query tile row (32 packed query tokens, the same for every head): the key
    // tiles [kfirst, klast] its tokens see, of which the first `a` and the last `b`
    // take a mask tile (pages u0 .. u0 + a + b - 1 of the mask tiles, leading tiles
    // first); the tiles between are visible to all 32 tokens.
    struct Row {
        uint32_t kfirst, klast, a, b, u0;
    };

    struct Plan {
        std::vector<Row> rows;
        // Mask tiles, 1024 values each (row-major 32 x 32): 0 where the row's token
        // sees the tile's key, the masked value elsewhere.
        std::vector<float> masks;
        uint32_t queries = 0, keys = 0;
        uint32_t tiles = 0;  // mask tiles

        uint32_t mask_tiles() const {
            return tiles;
        }
    };

    // The keys [lo, hi) (packed key indices) each packed query token sees: its
    // sequence's keys; with `causal` the first min(Lk, r + 1 + Lk - Lq) of them for the
    // sequence's query r (the diagonal aligned at the last query and key).
    inline void intervals(
        const std::vector<int64_t>& cu_q,
        const std::vector<int64_t>& cu_k,
        bool causal,
        std::vector<int64_t>& lo,
        std::vector<int64_t>& hi
    ) {
        if (cu_q.size() < 2 || cu_q.size() != cu_k.size() || cu_q.front() != 0 ||
            cu_k.front() != 0) {
            throw std::invalid_argument(
                "kohakutt_native: sdpa_varlen needs cumulative offsets [0, ..., total] "
                "of equal count "
                "for queries and keys"
            );
        }
        lo.clear();
        hi.clear();
        for (size_t b = 0; b + 1 < cu_q.size(); ++b) {
            const int64_t lq = cu_q[b + 1] - cu_q[b], lk = cu_k[b + 1] - cu_k[b];
            if (lq < 0 || lk < 0 || (lq > 0 && lk == 0)) {
                throw std::invalid_argument(
                    "kohakutt_native: sdpa_varlen offsets must not decrease, and a "
                    "sequence with "
                    "queries needs keys"
                );
            }
            for (int64_t r = 0; r < lq; ++r) {
                const int64_t seen = causal ? std::min(lk, r + 1 + lk - lq) : lk;
                if (seen <= 0) {
                    throw std::invalid_argument(
                        "kohakutt_native: causal sdpa_varlen with more queries than "
                        "keys in a "
                        "sequence leaves queries without keys"
                    );
                }
                lo.push_back(cu_k[b]);
                hi.push_back(cu_k[b] + seen);
            }
        }
    }

    // The plan of offsets `cu_q` / `cu_k` ([0, ..., total], one entry per sequence
    // boundary). Rows past the last query in the last tile repeat the last token's
    // interval. `masked`: the mask tiles' value for an unseen key (kMaskedLogit /
    // scale); 0 builds only the row records.
    inline Plan plan(
        const std::vector<int64_t>& cu_q,
        const std::vector<int64_t>& cu_k,
        bool causal,
        float masked
    ) {
        const bool masks = masked != 0.0f;
        std::vector<int64_t> lo, hi;
        intervals(cu_q, cu_k, causal, lo, hi);
        Plan out;
        out.queries = static_cast<uint32_t>(lo.size());
        out.keys = static_cast<uint32_t>(cu_k.back());
        for (size_t i = 0; i < lo.size(); i += 32) {
            int64_t rlo[32], rhi[32];
            for (size_t r = 0; r < 32; ++r) {
                const size_t t = std::min(i + r, lo.size() - 1);
                rlo[r] = lo[t];
                rhi[r] = hi[t];
            }
            const int64_t lo_min = *std::min_element(rlo, rlo + 32);
            const int64_t lo_max = *std::max_element(rlo, rlo + 32);
            const int64_t hi_min = *std::min_element(rhi, rhi + 32);
            const int64_t hi_max = *std::max_element(rhi, rhi + 32);
            const int64_t kfirst = lo_min / 32, klast = (hi_max - 1) / 32,
                          nk = klast - kfirst + 1;
            // Tiles every token sees whole: [clear0, clear1).
            const int64_t clear0 = std::max(kfirst, (lo_max + 31) / 32);
            const int64_t clear1 = std::min(klast + 1, hi_min / 32);
            Row row{};
            row.kfirst = static_cast<uint32_t>(kfirst);
            row.klast = static_cast<uint32_t>(klast);
            if (clear0 < clear1) {
                row.a = static_cast<uint32_t>(clear0 - kfirst);
                row.b = static_cast<uint32_t>(klast + 1 - clear1);
            } else {
                row.a = static_cast<uint32_t>(nk);
                row.b = 0;
            }
            row.u0 = out.tiles;
            out.tiles += row.a + row.b;
            for (int64_t rel = 0; masks && rel < nk; ++rel) {
                if (rel >= row.a && rel < nk - row.b) {
                    continue;
                }
                const int64_t col0 = 32 * (kfirst + rel);
                for (size_t r = 0; r < 32; ++r) {
                    for (int64_t c = 0; c < 32; ++c) {
                        const bool seen = col0 + c >= rlo[r] && col0 + c < rhi[r];
                        out.masks.push_back(seen ? 0.0f : masked);
                    }
                }
            }
            out.rows.push_back(row);
        }
        return out;
    }

    // The plan seen from the keys (the backward's dK / dV program walks key tile rows):
    // one record per key tile row with, in Row's fields, the query tile rows [kfirst,
    // klast] that see it, of which the first `a` and the last `b` take a mask tile. The
    // mask tiles are the row plan's (query rows x key columns), in this plan's order. A
    // key tile no query sees has kfirst > klast (1, 0). `masked` as plan().
    inline Plan key_plan(
        const std::vector<int64_t>& cu_q,
        const std::vector<int64_t>& cu_k,
        bool causal,
        float masked
    ) {
        const bool masks = masked != 0.0f;
        std::vector<int64_t> lo, hi;
        intervals(cu_q, cu_k, causal, lo, hi);
        const Plan rows = plan(cu_q, cu_k, causal, 0.0f);
        const uint32_t qt = static_cast<uint32_t>(rows.rows.size());
        const uint32_t kt = static_cast<uint32_t>((cu_k.back() + 31) / 32);
        auto takes_mask = [&](uint32_t t, uint32_t c) {
            const Row& r = rows.rows[t];
            const uint32_t rel = c - r.kfirst, n = r.klast - r.kfirst + 1;
            return rel < r.a || rel >= n - r.b;
        };
        Plan out;
        out.queries = rows.queries;
        out.keys = rows.keys;
        uint32_t
            from = 0;  // first query tile row whose window reaches key tile c or beyond
        for (uint32_t c = 0; c < kt; ++c) {
            while (from < qt && rows.rows[from].klast < c) {
                ++from;
            }
            Row rec{1, 0, 0, 0, out.tiles};
            if (from < qt && rows.rows[from].kfirst <= c) {
                uint32_t last = from;
                while (last + 1 < qt && rows.rows[last + 1].kfirst <= c) {
                    ++last;
                }
                rec.kfirst = from;
                rec.klast = last;
                const uint32_t n = last - from + 1;
                while (rec.a < n && takes_mask(from + rec.a, c)) {
                    ++rec.a;
                }
                while (rec.a + rec.b < n && takes_mask(last - rec.b, c)) {
                    ++rec.b;
                }
                for (uint32_t t = from + rec.a; t + rec.b <= last; ++t) {
                    if (takes_mask(t, c)) {
                        throw std::logic_error(
                            "kohakutt_native: sdpa_varlen key plan: masked tiles "
                            "inside a window"
                        );
                    }
                }
                out.tiles += rec.a + rec.b;
                for (uint32_t t = from; masks && t <= last; ++t) {
                    if (t - from >= rec.a && t + rec.b <= last) {
                        continue;
                    }
                    for (size_t r = 0; r < 32; ++r) {
                        const size_t tok = std::min<size_t>(
                            32 * static_cast<size_t>(t) + r, lo.size() - 1
                        );
                        for (
                            int64_t col = 32 * static_cast<int64_t>(c);
                            col < 32 * static_cast<int64_t>(c) + 32;
                            ++col
                        ) {
                            out.masks.push_back(
                                col >= lo[tok] && col < hi[tok] ? 0.0f : masked
                            );
                        }
                    }
                }
            }
            out.rows.push_back(rec);
        }
        return out;
    }

}  // namespace ktt::bridge::ktt_flash::varlen
