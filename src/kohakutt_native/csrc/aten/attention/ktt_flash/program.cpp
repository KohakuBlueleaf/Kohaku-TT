// Host side of ktt_flash (forward, and the backward's dQ and dK / dV programs): block
// sizes under the program L1 budget, the lane plans (each core's jobs), circular
// buffers, kernels and runtime args.
#include "aten/attention/ktt_flash/program.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "aten/attention/ktt_flash/flash.hpp"
#include "aten/attention/ktt_flash/flash_bw.hpp"
#include "core/device/handle.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"

namespace ktt::bridge::ktt_flash {
    namespace {

        namespace dev = ::ktt_flash;

        constexpr uint32_t kTile = 2048;    // bf16 tile
        constexpr uint32_t kTile32 = 4096;  // fp32 tile
        constexpr uint32_t kToken = 64;     // token page
        constexpr uint32_t kSc = dev::kSubChunk;
        constexpr uint32_t kLaneSlots = 4;  // cb_lane tiles (steps run in pairs)
        constexpr uint32_t kStatSlots = 2;  // cb_stats tiles (one per finished row)
        // Jobs per core: the reader takes 17 + 7 per job runtime args, at most
        // tt::tt_metal::max_runtime_args (341).
        constexpr uint32_t kMaxJobs = 46;

        // Runtime arguments the kernel sources read: per core (the readers' key tail
        // after the jobs), then per job; the forward's dump variant adds its own.
        constexpr uint32_t kReaderArgs = 17, kReaderJobArgs = 7;  // reader.cpp
        constexpr uint32_t kWriterArgs = 4, kWriterJobArgs = 3,
                           kWriterDumpArgs = 3;  // writer.cpp
        constexpr uint32_t kComputeArgs = 2, kComputeJobArgs = 4,
                           kComputeDumpArgs = 2;  // compute.cpp
        constexpr uint32_t kDqReaderArgs = 19,
                           kDqReaderJobArgs = 5;                     // bw_dq_reader.cpp
        constexpr uint32_t kDqWriterArgs = 4, kDqWriterJobArgs = 3;  // bw_dq_writer.cpp
        constexpr uint32_t kDqComputeArgs = 2,
                           kDqComputeJobArgs = 4;  // bw_dq_compute.cpp
        constexpr uint32_t kDkvReaderArgs = 18,
                           kDkvReaderJobArgs = 6;  // bw_dkv_reader.cpp
        constexpr uint32_t kDkvWriterArgs = 4,
                           kDkvWriterJobArgs = 3;  // bw_dkv_writer.cpp
        constexpr uint32_t kDkvComputeArgs = 2,
                           kDkvComputeJobArgs = 5;  // bw_dkv_compute.cpp
        static_assert(
            kReaderArgs + kReaderJobArgs * kMaxJobs <= tt::tt_metal::max_runtime_args
        );
        static_assert(
            kDqReaderArgs + kDqReaderJobArgs * kMaxJobs <=
            tt::tt_metal::max_runtime_args
        );
        static_assert(
            kDkvReaderArgs + kDkvReaderJobArgs * kMaxJobs <=
            tt::tt_metal::max_runtime_args
        );

        uint32_t ceil_div(uint32_t a, uint32_t b) {
            return (a + b - 1) / b;
        }

        // Problem in tiles. Mask: batches / heads of the mask (1 or the full count),
        // `mask_rows` for a per-query-row mask, `mask_bcast` for a mask row broadcast
        // over the queries. `stats`: the forward variant that also writes the row
        // statistics.
        struct Dims {
            uint32_t b, h, hk, sqt, skt, dt, dvt;
            bool causal;
            uint32_t mb, mh;
            bool mask_rows, mask_bcast;
            bool stats;

            // Last key tile query tile row i sees.
            uint32_t last_key(uint32_t i) const {
                return causal ? std::min(i, skt - 1) : skt - 1;
            }
        };

        Dims dims_of(
            const ttnn::Tensor& q,
            const ttnn::Tensor& k,
            const ttnn::Tensor& v,
            const std::optional<ttnn::Tensor>& mask,
            bool causal,
            bool stats = false
        ) {
            const auto& qs = q.logical_shape();
            Dims s{};
            s.b = qs[0];
            s.h = qs[1];
            s.hk = k.logical_shape()[1];
            s.sqt = ceil_div(qs[2], 32);
            s.skt = ceil_div(k.logical_shape()[2], 32);
            s.dt = qs[3] / 32;
            s.dvt = v.logical_shape()[3] / 32;
            s.causal = causal;
            s.mask_rows = mask && mask->logical_shape()[2] != 1;
            s.mask_bcast = mask && !s.mask_rows;
            s.mb = mask ? mask->logical_shape()[0] : 1;
            s.mh = mask ? mask->logical_shape()[1] : 1;
            s.stats = stats;
            return s;
        }

        // One job of a core: up to R query tile rows of one head (first row `start`,
        // rows `stride` apart) against key tiles [0, nk); first pages of Q, K, V, O and
        // the mask.
        struct Job {
            uint32_t q0, k0, v0, o0, rows, start, nk, stride, u0;
        };

        // A query chunk of a lane: rows [r0, r0 + n) of head (bi, hi), K / V head g.
        struct Chunk {
            uint32_t bi, hi, g, r0, n;
        };

        // Jobs per core (index y * gx + x) for lanes of `length` cores down each grid
        // column (the sender at the top). A lane job is up to length * r query rows of
        // one head, split over the lane: contiguous rows, or (causal) rows `length`
        // apart. Non-causal lanes take even ranges of the (batch, head, row) order;
        // causal lanes take chunks largest work first onto the least loaded lane.
        std::vector<std::vector<Job>> plan(
            const Dims& s, uint32_t r, uint32_t gx, uint32_t gy, uint32_t length
        ) {
            std::vector<std::pair<uint32_t, uint32_t>> lanes;  // (x, y0)
            for (uint32_t x = 0; x < gx; ++x) {
                for (uint32_t y0 = 0; y0 + length <= gy; y0 += length) {
                    lanes.emplace_back(x, y0);
                }
            }
            const uint32_t total = s.b * s.h * s.sqt;
            const uint32_t group = s.h / s.hk;
            std::vector<std::vector<Chunk>> lane_chunks(lanes.size());
            if (s.causal) {
                std::vector<std::pair<uint64_t, Chunk>> pieces;
                for (uint32_t head = 0; head < s.b * s.h; ++head) {
                    const uint32_t bi = head / s.h, hi = head % s.h;
                    const uint32_t n_chunks = ceil_div(s.sqt, length * r);
                    for (uint32_t c = 0; c < n_chunks; ++c) {
                        const uint32_t r0 = c * s.sqt / n_chunks,
                                       r1 = (c + 1) * s.sqt / n_chunks;
                        const uint64_t work = static_cast<uint64_t>(
                                                  ceil_div(r1 - r0, length)
                                              ) *
                            (s.last_key(r0) + s.last_key(r1 - 1) + 2) / 2;
                        pieces.push_back(
                            {work, Chunk{bi, hi, hi / group, r0, r1 - r0}}
                        );
                    }
                }
                std::stable_sort(
                    pieces.begin(), pieces.end(), [](const auto& a, const auto& b) {
                        return a.first > b.first;
                    }
                );
                // Least loaded among the lanes below kMaxJobs chunks (any lane once all
                // are full).
                std::vector<uint64_t> load(lanes.size(), 0);
                for (const auto& [work, chunk] : pieces) {
                    size_t i = lanes.size();
                    for (size_t j = 0; j < lanes.size(); ++j) {
                        if (lane_chunks[j].size() < kMaxJobs &&
                            (i == lanes.size() || load[j] < load[i])) {
                            i = j;
                        }
                    }
                    if (i == lanes.size()) {
                        i = std::min_element(load.begin(), load.end()) - load.begin();
                    }
                    lane_chunks[i].push_back(chunk);
                    load[i] += work;
                }
            } else {
                for (size_t i = 0; i < lanes.size(); ++i) {
                    const uint32_t lo = static_cast<uint32_t>(
                        static_cast<uint64_t>(i) * total / lanes.size()
                    );
                    const uint32_t hi_ = static_cast<uint32_t>(
                        static_cast<uint64_t>(i + 1) * total / lanes.size()
                    );
                    for (uint32_t pos = lo; pos < hi_;) {
                        const uint32_t head = pos / s.sqt, r0 = pos % s.sqt;
                        const uint32_t n = std::min(
                            {hi_ - pos, s.sqt - r0, length * r}
                        );
                        const uint32_t bi = head / s.h, hi = head % s.h;
                        lane_chunks[i].push_back(Chunk{bi, hi, hi / group, r0, n});
                        pos += n;
                    }
                }
            }
            std::vector<std::vector<Job>> per_core(gx * gy);
            for (size_t i = 0; i < lanes.size(); ++i) {
                const auto [x, y0] = lanes[i];
                for (uint32_t y = y0; y < y0 + length; ++y) {
                    auto& jobs = per_core[y * gx + x];
                    for (const auto& c : lane_chunks[i]) {
                        uint32_t start, stride, rows;
                        if (s.causal) {
                            start = c.r0 + (y - y0);
                            stride = length;
                            rows = start < c.r0 + c.n
                                ? ceil_div(c.r0 + c.n - start, length)
                                : 0;
                            start = std::min(start, c.r0 + c.n);
                        } else {
                            const uint32_t per = ceil_div(c.n, length);
                            start = std::min(c.r0 + (y - y0) * per, c.r0 + c.n);
                            stride = 1;
                            rows = std::min(per, c.r0 + c.n - start);
                        }
                        const uint32_t nk = s.last_key(c.r0 + c.n - 1) + 1;
                        const uint32_t mbi = s.mb > 1 ? c.bi : 0,
                                       mhi = s.mh > 1 ? c.hi : 0;
                        const uint32_t u0 = s.mask_rows
                            ? ((mbi * s.mh + mhi) * s.sqt + start) * s.skt
                            : (mbi * s.mh + mhi) * s.skt;
                        const uint32_t qrow = (c.bi * s.h + c.hi) * s.sqt + start;
                        jobs.push_back(
                            Job{qrow * s.dt,
                                (c.bi * s.hk + c.g) * s.skt * s.dt,
                                (c.bi * s.hk + c.g) * s.skt * s.dvt,
                                qrow * s.dvt,
                                rows,
                                start,
                                nk,
                                stride,
                                u0}
                        );
                    }
                }
            }
            return per_core;
        }

        // The busiest core's score tiles.
        uint64_t max_tiles(
            const Dims& s, const std::vector<std::vector<Job>>& per_core
        ) {
            uint64_t best = 0;
            for (const auto& jobs : per_core) {
                uint64_t t = 0;
                for (const auto& j : jobs) {
                    for (uint32_t i = 0; i < j.rows; ++i) {
                        t += s.last_key(j.start + i * j.stride) + 1;
                    }
                }
                best = std::max(best, t);
            }
            return best;
        }

        // The most jobs on one core of a plan.
        size_t max_jobs(const std::vector<std::vector<Job>>& per_core) {
            size_t best = 0;
            for (const auto& jobs : per_core) {
                best = std::max(best, jobs.size());
            }
            return best;
        }

        // Circular-buffer bytes per core.
        uint32_t cb_bytes(const Dims& s, uint32_t r, uint32_t kb) {
            const uint32_t w = s.dvt + 1 + (s.stats ? 1 : 0);
            uint32_t bf16 = r * s.dt + 2 * s.dt * kb + 2 * kb * s.dvt + 1 + 2 * kSc +
                2 * s.dvt + 4;
            if (s.mask_rows) {
                bf16 += 4 * kSc;
            }
            if (s.mask_bcast) {
                bf16 += 2 * kb;
            }
            const uint32_t fp32 = r * w + 2 * r +
                (s.stats ? kStatSlots + kLaneSlots : 0);
            return bf16 * kTile + fp32 * kTile32 + (r + 8) * kToken;
        }

        // Rows per job (R) and tiles per streamed block (whole sub-chunks) with
        // bytes(r, block) under the circular buffer budget (allocator read: on the
        // stream worker); r = 0 when nothing fits.
        template <typename Bytes>
        std::pair<uint32_t, uint32_t> fit_blocks(const Bytes& bytes) {
            const uint32_t budget = program::cb_budget();
            uint32_t r = 13, kb = 12;
            while (bytes(r, kb) > budget) {
                if (kb > 6) {
                    kb -= kSc;
                } else if (r > 4) {
                    --r;
                } else if (kb > kSc) {
                    kb -= kSc;
                } else if (r > 1) {
                    --r;
                } else {
                    return {0, 0};
                }
            }
            return {r, kb};
        }

        // Forward R and KB.
        std::pair<uint32_t, uint32_t> blocks(const Dims& s) {
            return fit_blocks([&](uint32_t r, uint32_t kb) {
                return cb_bytes(s, r, kb);
            });
        }

        // True when lanes of one core (the most lanes) keep every core at kMaxJobs jobs
        // or fewer for `heads` heads of `rows` tile rows split in jobs of up to r rows.
        // Non-causal: a lane's m = ceil(T / cores) consecutive rows give at most m / r
        // full chunks, one per head boundary and the last. Causal: ceil(rows / r)
        // chunks per head, capped at kMaxJobs per lane.
        bool args_fit(
            uint64_t heads, uint32_t rows, bool causal, uint32_t r, uint32_t cores
        ) {
            if (causal) {
                return heads * ceil_div(rows, r) <=
                    static_cast<uint64_t>(kMaxJobs) * cores;
            }
            const uint64_t m = (heads * rows + cores - 1) / cores;
            return m / r + (m + rows - 1) / rows + 1 <= kMaxJobs;
        }

        bool args_fit(const Dims& s, uint32_t r, uint32_t cores) {
            return args_fit(
                static_cast<uint64_t>(s.b) * s.h, s.sqt, s.causal, r, cores
            );
        }

        // Lane length and per-core jobs of the forward / dQ plan: the fewest score
        // tiles on the busiest core, then the longest lane (fewest K / V reads); empty
        // when no plan keeps kMaxJobs.
        std::pair<uint32_t, std::vector<std::vector<Job>>> choose_lanes(
            const Dims& s, uint32_t r, uint32_t gx, uint32_t gy
        ) {
            uint32_t length = 1;
            std::vector<std::vector<Job>> per_core;
            uint64_t best = UINT64_MAX;
            for (uint32_t len = 1; len <= gy; ++len) {
                if (gy % len != 0) {
                    continue;
                }
                auto p = plan(s, r, gx, gy, len);
                const uint64_t t = max_tiles(s, p);
                if (max_jobs(p) <= kMaxJobs && t <= best) {
                    best = t;
                    length = len;
                    per_core = std::move(p);
                }
            }
            return {length, std::move(per_core)};
        }

        // Physical NoC coordinates of logical worker core (x, y).
        std::pair<uint32_t, uint32_t> phys(uint32_t x, uint32_t y) {
            const auto c = detail::device()->worker_core_from_logical_core(
                tt::tt_metal::CoreCoord(x, y)
            );
            return {static_cast<uint32_t>(c.x), static_cast<uint32_t>(c.y)};
        }

        // Reader runtime args after the tensor addresses: iterations (1), the lane's
        // multicast rectangle below the sender, receivers, the sender, the semaphore
        // ids, the sender flag and the job count.
        std::vector<uint32_t> lane_args(
            uint32_t x, uint32_t y, uint32_t length, uint32_t gy, size_t jobs
        ) {
            const uint32_t y0 = y - y % length;
            const auto top = phys(x, y0), first = phys(x, std::min(y0 + 1, gy - 1));
            const auto last = phys(x, y0 + length - 1);
            return {
                1u,
                first.first,
                first.second,
                last.first,
                last.second,
                length - 1,
                top.first,
                top.second,
                0u,
                1u,
                y == y0 ? 1u : 0u,
                static_cast<uint32_t>(jobs)
            };
        }

        namespace bw = ::ktt_flash_bw;
        constexpr uint32_t kChunk = bw::kChunk;

        // Statistics tiles per query tile row between the dQ and dK / dV programs: the
        // row tile, or in the key-row orientation the two column tiles.
        uint32_t stat_tiles(bool key_rows) {
            return key_rows ? 2 : 1;
        }

        // dQ program circular-buffer bytes per core (R query rows, KB key tiles per
        // block).
        uint32_t dq_cb_bytes(const Dims& s, uint32_t r, uint32_t kb, bool key_rows) {
            const uint32_t bf16 = r * s.dt + 2 * s.dt * kb + 2 * kb * s.dvt +
                2 * r * s.dvt + 1 + s.dvt + 2 * kChunk + 1 + 2 * s.dt + 3;
            const uint32_t fp32 = 2 * r + r * s.dt + 2 * stat_tiles(key_rows);
            return bf16 * kTile + fp32 * kTile32 + 2 * kToken;
        }

        // dK / dV program circular-buffer bytes per core (R key rows, QB query tiles
        // per block). The default orientation adds a chunk of Q^T and dO^T and a
        // finished row.
        uint32_t dkv_cb_bytes(const Dims& s, uint32_t r, uint32_t qb, bool key_rows) {
            const uint32_t w = s.dt + s.dvt;
            const uint32_t bf16 = r * w + 2 * qb * w + 3 + 4 * kChunk + 2 * w +
                (key_rows ? 0 : (kChunk + 1) * w);
            const uint32_t fp32 = 2 * stat_tiles(key_rows) * qb + r * w;
            return bf16 * kTile + fp32 * kTile32 + 2 * kToken;
        }

        std::pair<uint32_t, uint32_t> dq_blocks(const Dims& s, bool key_rows) {
            return fit_blocks([&](uint32_t r, uint32_t kb) {
                return dq_cb_bytes(s, r, kb, key_rows);
            });
        }

        std::pair<uint32_t, uint32_t> dkv_blocks(const Dims& s, bool key_rows) {
            return fit_blocks([&](uint32_t r, uint32_t qb) {
                return dkv_cb_bytes(s, r, qb, key_rows);
            });
        }

        // One dK / dV job of a core: up to R key tile rows of one K / V head (global
        // first row krow0, `start` within the head, rows `stride` apart) against query
        // tiles [q_first, Sqt) of the head's query heads (global first query head
        // head0).
        struct KvJob {
            uint32_t krow0, rows, start, stride, head0, q_first;
        };

        // A key chunk of a lane: rows [r0, r0 + n) of K / V head `head` (global).
        struct KvChunk {
            uint32_t head, r0, n;
        };

        // dK / dV jobs per core (index y * gx + x) for lanes of `length` cores down
        // each grid column, as plan() with key rows in place of query rows: a lane job
        // is up to length * r key rows of one K / V head (causal: rows `length` apart,
        // chunks largest work first onto the least loaded lane), and the lane streams
        // the query blocks of the head's group.
        std::vector<std::vector<KvJob>> plan_kv(
            const Dims& s, uint32_t r, uint32_t gx, uint32_t gy, uint32_t length
        ) {
            std::vector<std::pair<uint32_t, uint32_t>> lanes;
            for (uint32_t x = 0; x < gx; ++x) {
                for (uint32_t y0 = 0; y0 + length <= gy; y0 += length) {
                    lanes.emplace_back(x, y0);
                }
            }
            const uint32_t heads = s.b * s.hk;
            const uint32_t total = heads * s.skt;
            std::vector<std::vector<KvChunk>> lane_chunks(lanes.size());
            if (s.causal) {
                std::vector<std::pair<uint64_t, KvChunk>> pieces;
                for (uint32_t head = 0; head < heads; ++head) {
                    const uint32_t n_chunks = ceil_div(s.skt, length * r);
                    for (uint32_t c = 0; c < n_chunks; ++c) {
                        const uint32_t r0 = c * s.skt / n_chunks,
                                       r1 = (c + 1) * s.skt / n_chunks;
                        const uint64_t work = static_cast<uint64_t>(
                                                  ceil_div(r1 - r0, length)
                                              ) *
                            ((s.sqt - r0) + (s.sqt - r1 + 1)) / 2;
                        pieces.push_back({work, KvChunk{head, r0, r1 - r0}});
                    }
                }
                std::stable_sort(
                    pieces.begin(), pieces.end(), [](const auto& a, const auto& b) {
                        return a.first > b.first;
                    }
                );
                std::vector<uint64_t> load(lanes.size(), 0);
                for (const auto& [work, chunk] : pieces) {
                    size_t i = lanes.size();
                    for (size_t j = 0; j < lanes.size(); ++j) {
                        if (lane_chunks[j].size() < kMaxJobs &&
                            (i == lanes.size() || load[j] < load[i])) {
                            i = j;
                        }
                    }
                    if (i == lanes.size()) {
                        i = std::min_element(load.begin(), load.end()) - load.begin();
                    }
                    lane_chunks[i].push_back(chunk);
                    load[i] += work;
                }
            } else {
                for (size_t i = 0; i < lanes.size(); ++i) {
                    const uint32_t lo = static_cast<uint32_t>(
                        static_cast<uint64_t>(i) * total / lanes.size()
                    );
                    const uint32_t hi = static_cast<uint32_t>(
                        static_cast<uint64_t>(i + 1) * total / lanes.size()
                    );
                    for (uint32_t pos = lo; pos < hi;) {
                        const uint32_t head = pos / s.skt, r0 = pos % s.skt;
                        const uint32_t n = std::min({hi - pos, s.skt - r0, length * r});
                        lane_chunks[i].push_back(KvChunk{head, r0, n});
                        pos += n;
                    }
                }
            }
            const uint32_t group = s.h / s.hk;
            std::vector<std::vector<KvJob>> per_core(gx * gy);
            for (size_t i = 0; i < lanes.size(); ++i) {
                const auto [x, y0] = lanes[i];
                for (uint32_t y = y0; y < y0 + length; ++y) {
                    auto& jobs = per_core[y * gx + x];
                    for (const auto& c : lane_chunks[i]) {
                        uint32_t start, stride, rows;
                        if (s.causal) {
                            start = c.r0 + (y - y0);
                            stride = length;
                            rows = start < c.r0 + c.n
                                ? ceil_div(c.r0 + c.n - start, length)
                                : 0;
                            start = std::min(start, c.r0 + c.n);
                        } else {
                            const uint32_t per = ceil_div(c.n, length);
                            start = std::min(c.r0 + (y - y0) * per, c.r0 + c.n);
                            stride = 1;
                            rows = std::min(per, c.r0 + c.n - start);
                        }
                        const uint32_t bi = c.head / s.hk, g = c.head % s.hk;
                        jobs.push_back(
                            KvJob{
                                c.head * s.skt + start,
                                rows,
                                start,
                                stride,
                                bi * s.h + g * group,
                                s.causal ? c.r0 : 0
                            }
                        );
                    }
                }
            }
            return per_core;
        }

        // The busiest core's score tiles of a dK / dV plan.
        uint64_t max_kv_tiles(
            const Dims& s, const std::vector<std::vector<KvJob>>& per_core
        ) {
            const uint32_t group = s.h / s.hk;
            uint64_t best = 0;
            for (const auto& jobs : per_core) {
                uint64_t t = 0;
                for (const auto& j : jobs) {
                    for (uint32_t i = 0; i < j.rows; ++i) {
                        t += static_cast<uint64_t>(group) *
                            (s.causal ? s.sqt - (j.start + i * j.stride) : s.sqt);
                    }
                }
                best = std::max(best, t);
            }
            return best;
        }

        // Lane length and per-core jobs of the dK / dV plan, chosen as choose_lanes().
        std::pair<uint32_t, std::vector<std::vector<KvJob>>> choose_kv_lanes(
            const Dims& s, uint32_t r, uint32_t gx, uint32_t gy
        ) {
            uint32_t length = 1;
            std::vector<std::vector<KvJob>> per_core;
            uint64_t best = UINT64_MAX;
            for (uint32_t len = 1; len <= gy; ++len) {
                if (gy % len != 0) {
                    continue;
                }
                auto p = plan_kv(s, r, gx, gy, len);
                size_t most = 0;
                for (const auto& jobs : p) {
                    most = std::max(most, jobs.size());
                }
                const uint64_t t = max_kv_tiles(s, p);
                if (most <= kMaxJobs && t <= best) {
                    best = t;
                    length = len;
                    per_core = std::move(p);
                }
            }
            return {length, std::move(per_core)};
        }

        // A circular buffer and the most pages any kernel writes into it under one
        // reservation: the size of its pushes, or with `slots` the tiles written in
        // place into a buffer that is reserved once and never pushed.
        struct Cb {
            uint32_t index, pages, page;
            tt::DataFormat format;
            uint32_t writes;
            bool slots = false;
        };

        // What the kernels do with `cbs`, for the launch check.
        std::vector<program::Ring> rings_of(const std::vector<Cb>& cbs) {
            std::vector<program::Ring> rings;
            for (const auto& cb : cbs) {
                if (cb.slots) {
                    rings.push_back(
                        {cb.index, {}, static_cast<uint64_t>(cb.writes) * cb.page}
                    );
                } else {
                    rings.push_back({cb.index, {cb.writes}, 0});
                }
            }
            return rings;
        }

        // The argument counts of a program's reader, writer and compute (kernels 0, 1,
        // 2): {per core, per job} each.
        std::vector<program::ArgCount> job_args(
            std::pair<uint32_t, uint32_t> reader,
            std::pair<uint32_t, uint32_t> writer,
            std::pair<uint32_t, uint32_t> compute
        ) {
            return {
                {.kernel = 0, .per_core = reader.first, .per_item = reader.second},
                {.kernel = 1, .per_core = writer.first, .per_item = writer.second},
                {.kernel = 2, .per_core = compute.first, .per_item = compute.second},
            };
        }

        // Circular buffers `cbs` on every core of `all`; each holds whole reservations
        // of its largest write.
        void add_cbs(
            tt::tt_metal::ProgramDescriptor& desc,
            const tt::tt_metal::CoreRangeSet& all,
            const std::vector<Cb>& cbs
        ) {
            for (const auto& cb : cbs) {
                TT_FATAL(
                    cb.writes > 0 && cb.pages >= cb.writes && cb.pages % cb.writes == 0,
                    "ktt_flash: circular buffer {} of {} pages takes writes of {}",
                    cb.index,
                    cb.pages,
                    cb.writes
                );
                desc.cbs.push_back(
                    tt::tt_metal::CBDescriptor{
                        .total_size = cb.pages * cb.page,
                        .core_ranges = all,
                        .format_descriptors = {tt::tt_metal::CBFormatDescriptor{
                            .buffer_index = static_cast<uint8_t>(cb.index),
                            .data_format = cb.format,
                            .page_size = cb.page
                        }}
                    }
                );
            }
        }

        // Reader (NoC0, RISCV_1), writer (NoC1, RISCV_0) and compute kernels on every
        // core of `all`, then the lane's ready / valid semaphores (ids 0, 1).
        void add_kernels(
            tt::tt_metal::ProgramDescriptor& desc,
            const tt::tt_metal::CoreRangeSet& all,
            const std::string& name,
            std::vector<uint32_t> reader_ct,
            std::vector<uint32_t> writer_ct,
            std::vector<uint32_t> compute_ct,
            tt::tt_metal::KernelDescriptor::RuntimeArgs rd,
            tt::tt_metal::KernelDescriptor::RuntimeArgs wr,
            tt::tt_metal::KernelDescriptor::RuntimeArgs cp,
            const tt::tt_metal::KernelDescriptor::Defines& defines,
            const tt::tt_metal::ComputeConfigDescriptor& compute,
            const tt::tt_metal::KernelDescriptor::Defines& reader_defines = {}
        ) {
            using namespace tt::tt_metal;
            const NOC reader_noc = NOC::NOC_0;
            const NOC writer_noc = reader_noc == NOC::NOC_0 ? NOC::NOC_1 : NOC::NOC_0;
            TT_FATAL(
                writer_noc != reader_noc, "ktt_flash: reader and writer share a NoC"
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source(
                        "attention/ktt_flash/" + name + "_reader.cpp"
                    ),
                    .core_ranges = all,
                    .compile_time_args = std::move(reader_ct),
                    .defines = reader_defines,
                    .runtime_args = std::move(rd),
                    .config = DataMovementConfigDescriptor{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = reader_noc,
                        .noc_mode = NOC_MODE::DM_DEDICATED_NOC
                    }
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source(
                        "attention/ktt_flash/" + name + "_writer.cpp"
                    ),
                    .core_ranges = all,
                    .compile_time_args = std::move(writer_ct),
                    .runtime_args = std::move(wr),
                    .config = DataMovementConfigDescriptor{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = writer_noc,
                        .noc_mode = NOC_MODE::DM_DEDICATED_NOC
                    }
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source(
                        "attention/ktt_flash/" + name + "_compute.cpp"
                    ),
                    .core_ranges = all,
                    .compile_time_args = std::move(compute_ct),
                    .defines = defines,
                    .runtime_args = std::move(cp),
                    .config = compute
                }
            );
            for (uint32_t id = 0; id < 2; ++id) {
                desc.semaphores.push_back(
                    SemaphoreDescriptor{
                        .id = id,
                        .core_type = tt::CoreType::WORKER,
                        .core_ranges = all,
                        .initial_value = 0
                    }
                );
            }
        }

        // fp32 Dst at `fidelity`, accurate SFPU, `fp32_cbs` unpacked straight to Dst.
        tt::tt_metal::ComputeConfigDescriptor bw_compute(
            tt::tt_metal::MathFidelity fidelity,
            std::initializer_list<uint32_t> fp32_cbs
        ) {
            using namespace tt::tt_metal;
            ComputeConfigDescriptor compute{
                .math_fidelity = fidelity,
                .fp32_dest_acc_en = true,
                .dst_full_sync_en = false,
                .unpack_to_dest_mode = ComputeConfigDescriptor::UnpackToDestModes(
                    NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default
                ),
                .math_approx_mode = false,
            };
            for (const uint32_t cb : fp32_cbs) {
                compute.unpack_to_dest_mode[cb] = UnpackToDestMode::UnpackToDestFp32;
            }
            return compute;
        }

        std::vector<uint32_t> accessor_args(
            std::initializer_list<const ttnn::Tensor*> tensors
        ) {
            std::vector<uint32_t> out;
            for (const auto* t : tensors) {
                const auto args = tt::tt_metal::TensorAccessorArgs(*t->buffer())
                                      .get_compile_time_args();
                out.insert(out.end(), args.begin(), args.end());
            }
            return out;
        }

    }  // namespace

    bool supports(
        const ttnn::Tensor& q,
        const ttnn::Tensor& k,
        const ttnn::Tensor& v,
        const std::optional<ttnn::Tensor>& mask,
        bool causal
    ) {
        using tt::tt_metal::DataType;
        const auto& qs = q.logical_shape();
        const auto& ks = k.logical_shape();
        const auto& vs = v.logical_shape();
        if (qs.rank() != 4 || ks.rank() != 4 || vs.rank() != 4) {
            return false;
        }
        for (const auto* t : {&q, &k, &v}) {
            if (t->dtype() != DataType::BFLOAT16 ||
                t->layout() != tt::tt_metal::Layout::TILE) {
                return false;
            }
        }
        const bool shapes =
            (qs[0] == ks[0] && ks[0] == vs[0] && ks[1] == vs[1] && qs[1] % ks[1] == 0 &&
             ks[2] == vs[2] && qs[3] == ks[3] && qs[3] % 32 == 0 && vs[3] % 32 == 0 &&
             qs[3] <= 256 && vs[3] <= 256 && (!causal || (qs[2] == ks[2] && !mask)));
        if (!shapes) {
            return false;
        }
        if (mask) {
            const auto& ms = mask->logical_shape();
            if (!(ms.rank() == 4 && mask->dtype() == DataType::BFLOAT16 &&
                  mask->layout() == tt::tt_metal::Layout::TILE &&
                  (ms[0] == 1 || ms[0] == qs[0]) && (ms[1] == 1 || ms[1] == qs[1]) &&
                  (ms[2] == 1 || ms[2] == qs[2]) && ms[3] == ks[2])) {
                return false;
            }
        }
        // The circular buffers fit and every core's jobs fit its runtime args.
        const Dims s = dims_of(q, k, v, mask, causal);
        const uint32_t r = blocks(s).first;
        const auto grid = detail::device()->compute_with_storage_grid_size();
        return r > 0 && args_fit(s, r, grid.x * grid.y);
    }

    ttnn::Tensor mask_operand(const ttnn::Tensor& mask, float scale) {
        auto m = ttnn::multiply(ttnn::clamp(mask, -1e4f, std::nullopt), 1.0f / scale);
        if (mask.logical_shape()[-1] % 32 != 0) {
            m = ttnn::fill_implicit_tile_padding(m, -1e4f / scale);
        }
        return m;
    }

    ttnn::Tensor run(
        const ttnn::Tensor& q,
        const ttnn::Tensor& k,
        const ttnn::Tensor& v,
        const std::optional<ttnn::Tensor>& mask,
        float scale,
        bool causal,
        const ttnn::ComputeKernelConfig& config,
        bool approx,
        ttnn::Tensor* stats,
        Dump* dump
    ) {
        using namespace tt::tt_metal;
        const auto& qs = q.logical_shape();
        const bool with_stats = stats != nullptr;
        const Dims s = dims_of(q, k, v, mask, causal, with_stats);
        const bool row_mask = s.mask_rows, bcast_mask = s.mask_bcast;
        const uint32_t tail = k.logical_shape()[2] % 32;
        const auto rk = blocks(s);
        const uint32_t r = rk.first, kb = rk.second;
        TT_FATAL(
            r > 0, "ktt_flash: circular buffers do not fit (supports() checks this)"
        );

        const float a = static_cast<float>(scale * 1.4426950408889634);
        const float thr = approx ? 32500.818359375f + 256.0f * 16.0f : 639.0f + 16.0f;
        const uint32_t a_bits = std::bit_cast<uint32_t>(a),
                       thr_bits = std::bit_cast<uint32_t>(thr);

        auto out = ttnn::empty(
            ttnn::Shape({qs[0], qs[1], qs[2], v.logical_shape()[3]}),
            DataType::BFLOAT16,
            Layout::TILE,
            detail::device(),
            MemoryConfig{}
        );
        if (with_stats) {
            *stats = ttnn::empty(
                ttnn::Shape({qs[0], qs[1], qs[2], 32}),
                DataType::FLOAT32,
                Layout::TILE,
                detail::device(),
                MemoryConfig{}
            );
        }
        program::mark(program::Phase::Alloc);
        program::Key key("ktt_flash");
        key.add(q)
            .add(k)
            .add(v)
            .add(static_cast<int64_t>(std::bit_cast<uint32_t>(scale)))
            .add(causal)
            .add(approx);
        if (mask) {
            key.add(*mask);
        }
        key.add(static_cast<int64_t>(config.math_fidelity))
            .add(static_cast<int64_t>(r))
            .add(static_cast<int64_t>(kb));
        key.add(with_stats);
        // io: inputs, the statistics (when written), then the output last.
        std::vector<ttnn::Tensor> io = {q, k, v};
        if (mask) {
            io.push_back(*mask);
        }
        if (with_stats) {
            io.push_back(*stats);
        }
        io.push_back(out);
        const uint32_t out_index = static_cast<uint32_t>(io.size() - 1);
        const uint32_t stats_index = with_stats ? out_index - 1 : out_index;
        const uint32_t mask_index = mask ? 3u : 0u;
        const ttnn::Tensor& stats_t = with_stats ? *stats : out;
        // Debug dump: per step of the dumped row 8 + W fp32 tiles (compute.cpp,
        // FLASH_DUMP).
        uint32_t dump_index = 0;
        if (dump) {
            const uint32_t w = s.dvt + 1 + (with_stats ? 1 : 0);
            const uint32_t steps = ceil_div(s.last_key(dump->row % s.sqt) + 1, kSc);
            dump->pages = steps * (8 + w);
            dump->tiles = ttnn::empty(
                ttnn::Shape({dump->pages * 32, 32}),
                DataType::FLOAT32,
                Layout::TILE,
                detail::device(),
                MemoryConfig{}
            );
            key.add(static_cast<int64_t>(dump->row) + 1);
            io.insert(io.begin() + out_index, dump->tiles);
            dump_index = out_index;
        }
        const uint32_t out_slot = dump ? out_index + 1 : out_index;
        return program::launch(key, io, [&] {
            const auto grid = detail::device()->compute_with_storage_grid_size();
            const uint32_t gx = grid.x, gy = grid.y;
            const auto [length, per_core] = choose_lanes(s, r, gx, gy);
            TT_FATAL(
                !per_core.empty(),
                "ktt_flash: no lane plan fits the runtime args (supports() checks this)"
            );

            KernelDescriptor::RuntimeArgs rd, wr, cp;
            std::vector<program::Slot> dump_slots;
            for (uint32_t y = 0; y < gy; ++y) {
                for (uint32_t x = 0; x < gx; ++x) {
                    const auto& jobs = per_core[y * gx + x];
                    std::vector<uint32_t> ra = {
                        q.buffer()->address(),
                        k.buffer()->address(),
                        v.buffer()->address(),
                        (mask ? mask->buffer() : q.buffer())->address()
                    };
                    const auto lane = lane_args(x, y, length, gy, jobs.size());
                    ra.insert(ra.end(), lane.begin(), lane.end());
                    std::vector<uint32_t> wa = {
                        out.buffer()->address(),
                        stats_t.buffer()->address(),
                        1u,
                        static_cast<uint32_t>(jobs.size())
                    };
                    std::vector<uint32_t> ca = {1u, static_cast<uint32_t>(jobs.size())};
                    uint32_t dump_job = UINT32_MAX, dump_r = 0;
                    for (uint32_t ji = 0; ji < jobs.size(); ++ji) {
                        const auto& j = jobs[ji];
                        ra.insert(
                            ra.end(), {j.q0, j.k0, j.v0, j.rows, j.nk, j.stride, j.u0}
                        );
                        wa.insert(wa.end(), {j.o0, j.rows, j.stride});
                        ca.insert(ca.end(), {j.rows, j.start, j.nk, j.stride});
                        for (uint32_t i = 0; dump && i < j.rows; ++i) {
                            if (j.o0 / s.dvt + i * j.stride == dump->row) {
                                dump_job = ji;
                                dump_r = i;
                            }
                        }
                    }
                    ra.push_back(tail);
                    if (dump) {
                        if (dump_job != UINT32_MAX) {
                            dump_slots.push_back(
                                {.kernel = 1,
                                 .entry = static_cast<int32_t>(wr.size()),
                                 .position = static_cast<uint32_t>(wa.size()),
                                 .tensor = dump_index}
                            );
                        }
                        wa.insert(
                            wa.end(),
                            {dump->tiles.buffer()->address(), dump_job, dump->pages}
                        );
                        ca.insert(ca.end(), {dump_job, dump_r});
                    }
                    const CoreCoord core(x, y);
                    rd.push_back({core, ra});
                    wr.push_back({core, wa});
                    cp.push_back({core, ca});
                }
            }
            const CoreRangeSet all(
                CoreRange(CoreCoord(0, 0), CoreCoord(gx - 1, gy - 1))
            );
            program::mark(program::Phase::Args);

            ProgramDescriptor desc;
            const auto bf16 = tt::DataFormat::Float16_b, f32 = tt::DataFormat::Float32;
            // Accumulator tiles per row (compute.cpp W): O, the row sum, with
            // statistics the lane sums.
            const uint32_t w = s.dvt + 1 + (with_stats ? 1 : 0);
            std::vector<Cb> cbs = {
                {dev::cb_q, r * s.dt, kTile, bf16, r * s.dt},
                {dev::cb_k, 2 * s.dt * kb, kTile, bf16, s.dt * kb},
                {dev::cb_v, 2 * kb * s.dvt, kTile, bf16, kb * s.dvt},
                {dev::cb_ones, 1, kTile, bf16, 1},
                {dev::cb_p, 2 * kSc, kTile, bf16, kSc},
                {dev::cb_st, 2 * r, kTile32, f32, 2 * r, true},
                {dev::cb_o, r * w, kTile32, f32, r * w, true},
                {dev::cb_stok, 4, kToken, bf16, 1},
                {dev::cb_otok, r + 4, kToken, bf16, 1},
                {dev::cb_ctail, 1, kTile, bf16, 1},
                {dev::cb_causal, 1, kTile, bf16, 1},
                {dev::cb_tail, 1, kTile, bf16, 1},
                {dev::cb_out, 2 * s.dvt, kTile, bf16, s.dvt},
            };
            if (row_mask) {
                cbs.push_back({dev::cb_um, 4 * kSc, kTile, bf16, kSc});
            }
            if (bcast_mask) {
                cbs.push_back({dev::cb_mb, 2 * kb, kTile, bf16, kb});
            }
            if (with_stats) {
                cbs.push_back({dev::cb_lane, kLaneSlots, kTile32, f32, 1});
                cbs.push_back({dev::cb_stats, kStatSlots, kTile32, f32, 1});
            }
            if (dump) {
                cbs.push_back({dev::cb_dump, 8, kTile32, f32, 1});
            }
            add_cbs(desc, all, cbs);

            KernelDescriptor::Defines defines;
            if (causal) {
                defines.emplace_back("FLASH_CAUSAL", "1");
            }
            if (row_mask) {
                defines.emplace_back("FLASH_MASK_ROW", "1");
            }
            if (bcast_mask) {
                defines.emplace_back("FLASH_MASK_BCAST", "1");
            }
            std::vector<uint32_t> reader_ct = {s.dt, s.dvt, r, s.skt, kb};
            for (const auto* t : {&q, &k, &v, mask ? &*mask : &q}) {
                const auto args = TensorAccessorArgs(*t->buffer())
                                      .get_compile_time_args();
                reader_ct.insert(reader_ct.end(), args.begin(), args.end());
            }
            std::vector<uint32_t> writer_ct = {s.dvt};
            const auto out_args = accessor_args({&out, &stats_t});
            writer_ct.insert(writer_ct.end(), out_args.begin(), out_args.end());
            KernelDescriptor::Defines writer_defines;
            if (with_stats) {
                writer_defines.emplace_back("FLASH_STATS", "1");
            }
            if (dump) {
                const auto dump_args = accessor_args({&dump->tiles});
                writer_ct.insert(writer_ct.end(), dump_args.begin(), dump_args.end());
                writer_defines.emplace_back("FLASH_DUMP", "1");
            }
            // Every reader on NoC0 (senders multicast down their column, NoC0 runs +y),
            // every writer on NoC1.
            const NOC reader_noc = NOC::NOC_0;
            const NOC writer_noc = reader_noc == NOC::NOC_0 ? NOC::NOC_1 : NOC::NOC_0;
            TT_FATAL(
                writer_noc != reader_noc, "ktt_flash: reader and writer share a NoC"
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("attention/ktt_flash/reader.cpp"),
                    .core_ranges = all,
                    .compile_time_args = reader_ct,
                    .defines = defines,
                    .runtime_args = rd,
                    .config = DataMovementConfigDescriptor{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = reader_noc,
                        .noc_mode = NOC_MODE::DM_DEDICATED_NOC
                    }
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("attention/ktt_flash/writer.cpp"),
                    .core_ranges = all,
                    .compile_time_args = writer_ct,
                    .defines = writer_defines,
                    .runtime_args = wr,
                    .config = DataMovementConfigDescriptor{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = writer_noc,
                        .noc_mode = NOC_MODE::DM_DEDICATED_NOC
                    }
                }
            );
            std::vector<program::Slot> slots;
            for (uint32_t pos = 0; pos < 3; ++pos) {
                slots.push_back(
                    {.kernel = 0,
                     .entry = program::Slot::kEveryCore,
                     .position = pos,
                     .tensor = pos}
                );
            }
            slots.push_back(
                {.kernel = 0,
                 .entry = program::Slot::kEveryCore,
                 .position = 3,
                 .tensor = mask_index}
            );
            slots.push_back(
                {.kernel = 1,
                 .entry = program::Slot::kEveryCore,
                 .position = 0,
                 .tensor = out_slot}
            );
            slots.insert(slots.end(), dump_slots.begin(), dump_slots.end());
            slots.push_back(
                {.kernel = 1,
                 .entry = program::Slot::kEveryCore,
                 .position = 1,
                 .tensor = stats_index}
            );
            ComputeConfigDescriptor compute{
                .math_fidelity = config.math_fidelity,
                .fp32_dest_acc_en = true,
                .dst_full_sync_en = false,
                .unpack_to_dest_mode = ComputeConfigDescriptor::UnpackToDestModes(
                    NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default
                ),
                .math_approx_mode = approx,
            };
            compute.unpack_to_dest_mode[dev::cb_o] = UnpackToDestMode::UnpackToDestFp32;
            compute
                .unpack_to_dest_mode[dev::cb_st] = UnpackToDestMode::UnpackToDestFp32;
            if (with_stats) {
                compute.unpack_to_dest_mode[dev::cb_lane] = UnpackToDestMode::
                    UnpackToDestFp32;
            }
            KernelDescriptor::Defines compute_defines = defines;
            if (approx) {
                compute_defines.emplace_back("FLASH_APPROX", "1");
            }
            if (with_stats) {
                compute_defines.emplace_back("FLASH_STATS", "1");
            }
            if (dump) {
                compute_defines.emplace_back("FLASH_DUMP", "1");
            }
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("attention/ktt_flash/compute.cpp"),
                    .core_ranges = all,
                    .compile_time_args =
                        {s.dt, s.dvt, r, s.skt, kb, a_bits, thr_bits, 0u, tail},
                    .defines = compute_defines,
                    .runtime_args = cp,
                    .config = compute
                }
            );
            for (uint32_t id = 0; id < 2; ++id) {
                desc.semaphores.push_back(
                    SemaphoreDescriptor{
                        .id = id,
                        .core_type = tt::CoreType::WORKER,
                        .core_ranges = all,
                        .initial_value = 0
                    }
                );
            }
            return program::Built{
                std::move(desc),
                std::move(slots),
                job_args(
                    {kReaderArgs, kReaderJobArgs},
                    {kWriterArgs + (dump ? kWriterDumpArgs : 0), kWriterJobArgs},
                    {kComputeArgs + (dump ? kComputeDumpArgs : 0), kComputeJobArgs}
                ),
                rings_of(cbs)
            };
        });
    }

    bool supports_backward(
        const ttnn::Tensor& q, const ttnn::Tensor& k, const ttnn::Tensor& v, bool causal
    ) {
        if (!supports(q, k, v, std::nullopt, causal)) {
            return false;
        }
        const bool ragged = q.logical_shape()[2] % 32 != 0 ||
            k.logical_shape()[2] % 32 != 0;
        const Dims s = dims_of(q, k, v, std::nullopt, causal, true);
        const auto grid = detail::device()->compute_with_storage_grid_size();
        const uint32_t cores = grid.x * grid.y;
        const uint32_t rf = blocks(s).first;
        if (rf == 0 || !args_fit(s, rf, cores)) {
            return false;
        }
        // Both orientations (the backward picks one per call); a ragged length runs the
        // default.
        for (const bool key_rows : {false, true}) {
            if (key_rows && ragged) {
                continue;
            }
            const uint32_t r = dq_blocks(s, key_rows).first,
                           rk = dkv_blocks(s, key_rows).first;
            if (r == 0 || rk == 0 || !args_fit(s, r, cores) ||
                !args_fit(
                    static_cast<uint64_t>(s.b) * s.hk, s.skt, causal, rk, cores
                )) {
                return false;
            }
        }
        return true;
    }

    std::array<ttnn::Tensor, 3> backward(
        const ttnn::Tensor& grad,
        const ttnn::Tensor& out,
        const ttnn::Tensor& q,
        const ttnn::Tensor& k,
        const ttnn::Tensor& v,
        const ttnn::Tensor& stats,
        float scale,
        bool causal,
        const ttnn::ComputeKernelConfig& config,
        bool key_rows
    ) {
        using namespace tt::tt_metal;
        const auto& qs = q.logical_shape();
        const Dims s = dims_of(q, k, v, std::nullopt, causal);
        // Queries in the last query tile and keys in the last key tile (0: 32); the
        // key-row orientation takes whole tiles only.
        const uint32_t qtail = qs[2] % 32, tail = k.logical_shape()[2] % 32;
        key_rows = key_rows && qtail == 0 && tail == 0;
        const uint32_t nst = stat_tiles(key_rows);
        const auto q_blocks = dq_blocks(s, key_rows),
                   kv_blocks = dkv_blocks(s, key_rows);
        const uint32_t r = q_blocks.first, kb = q_blocks.second, rk = kv_blocks.first,
                       qb = kv_blocks.second;
        TT_FATAL(
            r > 0 && rk > 0,
            "ktt_flash: backward circular buffers do not fit (supports_backward() "
            "checks this)"
        );
        const uint32_t a_bits = std::bit_cast<uint32_t>(
            static_cast<float>(scale * 1.4426950408889634)
        );
        const uint32_t scale_bits = std::bit_cast<uint32_t>(scale);
        const auto fidelity = config.math_fidelity;

        auto dd = ttnn::empty(
            ttnn::Shape({qs[0], qs[1], qs[2], 32 * nst}),
            DataType::FLOAT32,
            Layout::TILE,
            detail::device(),
            MemoryConfig{}
        );
        auto dq = ttnn::empty(
            q.logical_shape(),
            DataType::BFLOAT16,
            Layout::TILE,
            detail::device(),
            MemoryConfig{}
        );
        auto dk = ttnn::empty(
            k.logical_shape(),
            DataType::BFLOAT16,
            Layout::TILE,
            detail::device(),
            MemoryConfig{}
        );
        auto dv = ttnn::empty(
            v.logical_shape(),
            DataType::BFLOAT16,
            Layout::TILE,
            detail::device(),
            MemoryConfig{}
        );
        program::mark(program::Phase::Alloc);
        const auto bf16 = tt::DataFormat::Float16_b, f32 = tt::DataFormat::Float32;
        KernelDescriptor::Defines defines;
        if (causal) {
            defines.emplace_back("FLASH_CAUSAL", "1");
        }
        if (key_rows) {
            defines.emplace_back("FLASH_BW_KQ", "1");
        }

        // dQ and the statistics tiles (w and D per query tile row). io: q, k, v, dO, O,
        // log-sum-exp, statistics, dQ.
        program::Key key_q("ktt_flash_dq");
        key_q.add(q)
            .add(k)
            .add(v)
            .add(grad)
            .add(out)
            .add(stats)
            .add(static_cast<int64_t>(scale_bits))
            .add(causal);
        key_q.add(static_cast<int64_t>(fidelity))
            .add(static_cast<int64_t>(r))
            .add(static_cast<int64_t>(kb));
        key_q.add(key_rows);
        dq = program::launch(key_q, {q, k, v, grad, out, stats, dd, dq}, [&] {
            namespace c = bw::dq;
            const auto grid = detail::device()->compute_with_storage_grid_size();
            const uint32_t gx = grid.x, gy = grid.y;
            const auto [length, per_core] = choose_lanes(s, r, gx, gy);
            TT_FATAL(
                !per_core.empty(), "ktt_flash: no dQ lane plan fits the runtime args"
            );
            KernelDescriptor::RuntimeArgs rd, wr, cp;
            for (uint32_t y = 0; y < gy; ++y) {
                for (uint32_t x = 0; x < gx; ++x) {
                    const auto& jobs = per_core[y * gx + x];
                    std::vector<uint32_t> ra;
                    for (const auto* t : {&q, &k, &v, &grad, &out, &stats}) {
                        ra.push_back(t->buffer()->address());
                    }
                    const auto lane = lane_args(x, y, length, gy, jobs.size());
                    ra.insert(ra.end(), lane.begin(), lane.end());
                    std::vector<uint32_t> wa = {
                        dq.buffer()->address(),
                        dd.buffer()->address(),
                        1u,
                        static_cast<uint32_t>(jobs.size())
                    };
                    std::vector<uint32_t> ca = {1u, static_cast<uint32_t>(jobs.size())};
                    for (const auto& j : jobs) {
                        const uint32_t row = j.q0 / s.dt, kvh = j.k0 / (s.skt * s.dt);
                        ra.insert(ra.end(), {row, kvh, j.rows, j.nk, j.stride});
                        wa.insert(wa.end(), {row, j.rows, j.stride});
                        ca.insert(ca.end(), {j.rows, j.start, j.nk, j.stride});
                    }
                    ra.push_back(tail);
                    const CoreCoord core(x, y);
                    rd.push_back({core, ra});
                    wr.push_back({core, wa});
                    cp.push_back({core, ca});
                }
            }
            const CoreRangeSet all(
                CoreRange(CoreCoord(0, 0), CoreCoord(gx - 1, gy - 1))
            );
            program::mark(program::Phase::Args);
            ProgramDescriptor desc;
            const std::vector<Cb> cbs = {
                {c::cb_q, r * s.dt, kTile, bf16, r * s.dt},
                {c::cb_k, 2 * s.dt * kb, kTile, bf16, s.dt * kb},
                {c::cb_v, 2 * kb * s.dvt, kTile, bf16, kb * s.dvt},
                {c::cb_do, r * s.dvt, kTile, bf16, r * s.dvt},
                {c::cb_oin, r * s.dvt, kTile, bf16, r * s.dvt},
                {c::cb_lse, r, kTile32, f32, r},
                {c::cb_ones, 1, kTile, bf16, 1},
                {c::cb_prod, s.dvt, kTile, bf16, s.dvt},
                {c::cb_ds, 2 * kChunk, kTile, bf16, kChunk},
                {c::cb_acc, r * s.dt, kTile32, f32, r * s.dt, true},
                {c::cb_causal, 1, kTile, bf16, 1},
                {c::cb_out, 2 * s.dt, kTile, bf16, s.dt},
                {c::cb_dout, 2 * nst, kTile32, f32, nst},
                {c::cb_tok, 2, kToken, bf16, 1},
                {c::cb_w, r, kTile32, f32, r, true},
                {c::cb_tail, 1, kTile, bf16, 1},
                {c::cb_ctail, 1, kTile, bf16, 1},
                {c::cb_rtail, 1, kTile, bf16, 1},
            };
            add_cbs(desc, all, cbs);
            std::vector<uint32_t> reader_ct = {s.dt, s.dvt, r, s.skt, kb};
            const auto rargs = accessor_args({&q, &k, &v, &grad, &out, &stats});
            reader_ct.insert(reader_ct.end(), rargs.begin(), rargs.end());
            std::vector<uint32_t> writer_ct = {s.dt, nst};
            const auto wargs = accessor_args({&dq, &dd});
            writer_ct.insert(writer_ct.end(), wargs.begin(), wargs.end());
            add_kernels(
                desc,
                all,
                "bw_dq",
                reader_ct,
                writer_ct,
                {s.dt, s.dvt, r, kb, a_bits, scale_bits, s.sqt, s.skt, qtail, tail},
                rd,
                wr,
                cp,
                defines,
                bw_compute(fidelity, {c::cb_lse, c::cb_acc, c::cb_w})
            );
            std::vector<program::Slot> slots;
            for (uint32_t pos = 0; pos < 6; ++pos) {
                slots.push_back(
                    {.kernel = 0,
                     .entry = program::Slot::kEveryCore,
                     .position = pos,
                     .tensor = pos}
                );
            }
            slots.push_back(
                {.kernel = 1,
                 .entry = program::Slot::kEveryCore,
                 .position = 0,
                 .tensor = 7}
            );
            slots.push_back(
                {.kernel = 1,
                 .entry = program::Slot::kEveryCore,
                 .position = 1,
                 .tensor = 6}
            );
            return program::Built{
                std::move(desc),
                std::move(slots),
                job_args(
                    {kDqReaderArgs, kDqReaderJobArgs},
                    {kDqWriterArgs, kDqWriterJobArgs},
                    {kDqComputeArgs, kDqComputeJobArgs}
                ),
                rings_of(cbs)
            };
        });

        // dK and dV. io: q, k, v, dO, the statistics tiles, dK, dV.
        program::Key key_kv("ktt_flash_dkv");
        key_kv.add(q)
            .add(k)
            .add(v)
            .add(grad)
            .add(dd)
            .add(static_cast<int64_t>(scale_bits))
            .add(causal);
        key_kv.add(static_cast<int64_t>(fidelity))
            .add(static_cast<int64_t>(rk))
            .add(static_cast<int64_t>(qb));
        key_kv.add(key_rows);
        dv = program::launch(key_kv, {q, k, v, grad, dd, dk, dv}, [&] {
            namespace c = bw::dkv;
            const auto grid = detail::device()->compute_with_storage_grid_size();
            const uint32_t gx = grid.x, gy = grid.y;
            const auto [length, per_core] = choose_kv_lanes(s, rk, gx, gy);
            TT_FATAL(
                !per_core.empty(),
                "ktt_flash: no dK / dV lane plan fits the runtime args"
            );
            KernelDescriptor::RuntimeArgs rd, wr, cp;
            for (uint32_t y = 0; y < gy; ++y) {
                for (uint32_t x = 0; x < gx; ++x) {
                    const auto& jobs = per_core[y * gx + x];
                    std::vector<uint32_t> ra;
                    for (
                        const ttnn::Tensor* t :
                        std::initializer_list<const ttnn::Tensor*>{
                            &q, &k, &v, &grad, &dd
                        }
                    ) {
                        ra.push_back(t->buffer()->address());
                    }
                    const auto lane = lane_args(x, y, length, gy, jobs.size());
                    ra.insert(ra.end(), lane.begin(), lane.end());
                    std::vector<uint32_t> wa = {
                        dk.buffer()->address(),
                        dv.buffer()->address(),
                        1u,
                        static_cast<uint32_t>(jobs.size())
                    };
                    std::vector<uint32_t> ca = {1u, static_cast<uint32_t>(jobs.size())};
                    for (const auto& j : jobs) {
                        ra.insert(
                            ra.end(),
                            {j.krow0,
                             j.rows,
                             j.stride,
                             j.head0,
                             j.q_first,
                             s.sqt - j.q_first}
                        );
                        wa.insert(wa.end(), {j.krow0, j.rows, j.stride});
                        ca.insert(
                            ca.end(),
                            {j.rows, j.start, s.sqt - j.q_first, j.stride, j.q_first}
                        );
                    }
                    ra.push_back(s.skt * 64 + tail);
                    const CoreCoord core(x, y);
                    rd.push_back({core, ra});
                    wr.push_back({core, wa});
                    cp.push_back({core, ca});
                }
            }
            const CoreRangeSet all(
                CoreRange(CoreCoord(0, 0), CoreCoord(gx - 1, gy - 1))
            );
            program::mark(program::Phase::Args);
            ProgramDescriptor desc;
            const uint32_t w = s.dt + s.dvt;
            std::vector<Cb> cbs = {
                {c::cb_k, rk * s.dt, kTile, bf16, rk * s.dt},
                {c::cb_v, rk * s.dvt, kTile, bf16, rk * s.dvt},
                {c::cb_q, 2 * qb * s.dt, kTile, bf16, qb * s.dt},
                {c::cb_do, 2 * qb * s.dvt, kTile, bf16, qb * s.dvt},
                {c::cb_wd, 2 * nst * qb, kTile32, f32, nst * qb},
                {c::cb_p, 2 * kChunk, kTile, bf16, kChunk},
                {c::cb_ds, 2 * kChunk, kTile, bf16, kChunk},
                {c::cb_dk, rk * s.dt, kTile32, f32, rk * s.dt, true},
                {c::cb_dv, rk * s.dvt, kTile32, f32, rk * s.dvt, true},
                {c::cb_causal, 1, kTile, bf16, 1},
                {c::cb_out, 2 * w, kTile, bf16, w},
                {c::cb_tok, 2, kToken, bf16, 1},
                {c::cb_tail, 1, kTile, bf16, 1},
                {c::cb_ctail, 1, kTile, bf16, 1},
            };
            if (!key_rows) {
                cbs.push_back({c::cb_qt, kChunk * s.dt, kTile, bf16, kChunk * s.dt});
                cbs.push_back({c::cb_dot, kChunk * s.dvt, kTile, bf16, kChunk * s.dvt});
                cbs.push_back({c::cb_fin, w, kTile, bf16, w});
            }
            add_cbs(desc, all, cbs);
            const uint32_t group = s.h / s.hk;
            std::vector<uint32_t> reader_ct = {s.dt, s.dvt, rk, qb, s.sqt, group, nst};
            const auto rargs = accessor_args({&q, &k, &v, &grad, &dd});
            reader_ct.insert(reader_ct.end(), rargs.begin(), rargs.end());
            std::vector<uint32_t> writer_ct = {s.dt, s.dvt};
            const auto wargs = accessor_args({&dk, &dv});
            writer_ct.insert(writer_ct.end(), wargs.begin(), wargs.end());
            add_kernels(
                desc,
                all,
                "bw_dkv",
                reader_ct,
                writer_ct,
                {s.dt, s.dvt, rk, qb, group, a_bits, scale_bits, s.skt, tail},
                rd,
                wr,
                cp,
                defines,
                bw_compute(fidelity, {c::cb_wd, c::cb_dk, c::cb_dv})
            );
            std::vector<program::Slot> slots;
            for (uint32_t pos = 0; pos < 5; ++pos) {
                slots.push_back(
                    {.kernel = 0,
                     .entry = program::Slot::kEveryCore,
                     .position = pos,
                     .tensor = pos}
                );
            }
            slots.push_back(
                {.kernel = 1,
                 .entry = program::Slot::kEveryCore,
                 .position = 0,
                 .tensor = 5}
            );
            slots.push_back(
                {.kernel = 1,
                 .entry = program::Slot::kEveryCore,
                 .position = 1,
                 .tensor = 6}
            );
            return program::Built{
                std::move(desc),
                std::move(slots),
                job_args(
                    {kDkvReaderArgs, kDkvReaderJobArgs},
                    {kDkvWriterArgs, kDkvWriterJobArgs},
                    {kDkvComputeArgs, kDkvComputeJobArgs}
                ),
                rings_of(cbs)
            };
        });
        return {dq, dk, dv};
    }

    namespace {

        // Jobs per core of the packed variable-length programs: every core's job list
        // is padded to this length (empty jobs), so the argument counts do not depend
        // on the lengths. The reader takes 16 + 8 per job + 6 runtime args.
        constexpr uint32_t kVarJobs = 39;
        constexpr uint32_t kVarReaderArgs = 16 + 7,
                           kVarReaderJobArgs = 8;   // reader.cpp, FLASH_VARLEN
        constexpr uint32_t kVarComputeJobArgs = 5;  // compute.cpp, FLASH_VARLEN
        static_assert(
            kVarReaderArgs + kVarReaderJobArgs * kVarJobs <=
            tt::tt_metal::max_runtime_args
        );

        // One job of a core: `rows` query tile rows from `start` of query head `head`
        // against the key tiles [key0, key0 + nk) of K / V head `kv`.
        struct VarJob {
            uint32_t head, kv, start, rows, key0, nk;
        };

        // Dims of the packed q [Tq, H D], k [Tk, Hk D], v [Tk, Hk Dv] of `h` / `hk`
        // heads for the forward's buffer sizes (per-row mask tiles; `key_mask`: also
        // mask rows).
        Dims varlen_dims(
            const ttnn::Tensor& q,
            const ttnn::Tensor& k,
            const ttnn::Tensor& v,
            uint32_t h,
            uint32_t hk,
            bool stats,
            bool key_mask = false
        ) {
            Dims s{};
            s.b = 1;
            s.h = h;
            s.hk = hk;
            s.sqt = ceil_div(q.logical_shape()[0], 32);
            s.skt = ceil_div(k.logical_shape()[0], 32);
            s.dt = q.logical_shape()[1] / h / 32;
            s.dvt = v.logical_shape()[1] / hk / 32;
            s.mb = s.mh = 1;
            s.mask_rows = true;
            s.mask_bcast = key_mask;
            s.stats = stats;
            return s;
        }

        uint32_t varlen_cb_bytes(const Dims& s, uint32_t r, uint32_t kb) {
            return cb_bytes(s, r, kb) + 2 * dev::kPlanPages * kTile32;
        }

        // Jobs per core (index y * gx + x) for lanes of `length` cores down each grid
        // column: the lanes take even ranges of the (head, query tile row) order; a
        // lane job is up to length * r consecutive rows of one head, split over the
        // lane's cores, and streams the key tiles any of its rows sees.
        std::vector<std::vector<VarJob>> plan_varlen(
            const Dims& s,
            const std::vector<varlen::Row>& rows,
            uint32_t r,
            uint32_t gx,
            uint32_t gy,
            uint32_t length
        ) {
            std::vector<std::vector<VarJob>> per_core(gx * gy);
            const uint32_t lanes = gx * (gy / length);
            const uint32_t total = s.h * s.sqt, group = s.h / s.hk;
            for (uint32_t i = 0; i < lanes; ++i) {
                const uint32_t x = i / (gy / length), y0 = i % (gy / length) * length;
                const uint32_t lo = static_cast<uint32_t>(
                    static_cast<uint64_t>(i) * total / lanes
                );
                const uint32_t hi = static_cast<uint32_t>(
                    static_cast<uint64_t>(i + 1) * total / lanes
                );
                for (uint32_t pos = lo; pos < hi;) {
                    const uint32_t head = pos / s.sqt, r0 = pos % s.sqt;
                    const uint32_t n = std::min({hi - pos, s.sqt - r0, length * r});
                    uint32_t key0 = rows[r0].kfirst, key1 = rows[r0].klast;
                    for (uint32_t t = r0; t < r0 + n; ++t) {
                        key0 = std::min(key0, rows[t].kfirst);
                        key1 = std::max(key1, rows[t].klast);
                    }
                    const uint32_t per = ceil_div(n, length);
                    for (uint32_t y = y0; y < y0 + length; ++y) {
                        const uint32_t start = std::min(r0 + (y - y0) * per, r0 + n);
                        per_core[y * gx + x].push_back(
                            VarJob{
                                head,
                                head / group,
                                start,
                                std::min(per, r0 + n - start),
                                key0,
                                key1 - key0 + 1
                            }
                        );
                    }
                    pos += n;
                }
            }
            return per_core;
        }

        // Lane length and per-core jobs: the fewest score tiles on the busiest core,
        // then the longest lane; empty when no plan keeps kVarJobs jobs per core.
        std::pair<uint32_t, std::vector<std::vector<VarJob>>> choose_varlen_lanes(
            const Dims& s,
            const std::vector<varlen::Row>& rows,
            uint32_t r,
            uint32_t gx,
            uint32_t gy
        ) {
            uint32_t length = 1;
            std::vector<std::vector<VarJob>> per_core;
            uint64_t best = UINT64_MAX;
            for (uint32_t len = 1; len <= gy; ++len) {
                if (gy % len != 0) {
                    continue;
                }
                auto p = plan_varlen(s, rows, r, gx, gy, len);
                uint64_t most = 0;
                size_t jobs = 0;
                for (const auto& core : p) {
                    uint64_t tiles = 0;
                    for (const auto& j : core) {
                        for (uint32_t t = j.start; t < j.start + j.rows; ++t) {
                            tiles += rows[t].klast - rows[t].kfirst + 1;
                        }
                    }
                    most = std::max(most, tiles);
                    jobs = std::max(jobs, core.size());
                }
                if (jobs <= kVarJobs && most <= best) {
                    best = most;
                    length = len;
                    per_core = std::move(p);
                }
            }
            return {length, std::move(per_core)};
        }

    }  // namespace

    namespace {

        // A key mask is TILE bf16 [1, Tk] (additive, over the scale): one row tile per
        // key tile.
        void check_key_mask(const ttnn::Tensor* key_mask, const ttnn::Tensor& k) {
            if (!key_mask) {
                return;
            }
            const auto& ms = key_mask->logical_shape();
            if (!(ms.rank() == 2 && ms[0] == 1 && ms[1] == k.logical_shape()[0] &&
                  key_mask->dtype() == tt::tt_metal::DataType::BFLOAT16 &&
                  key_mask->layout() == tt::tt_metal::Layout::TILE)) {
                throw std::runtime_error(
                    "kohakutt_native: sdpa_varlen: the key mask must be bf16 [1, key "
                    "tokens]"
                );
            }
        }

    }  // namespace

    bool supports_varlen(
        const ttnn::Tensor& q,
        const ttnn::Tensor& k,
        const ttnn::Tensor& v,
        uint32_t h,
        uint32_t hk
    ) {
        using tt::tt_metal::DataType;
        const auto& qs = q.logical_shape();
        const auto& ks = k.logical_shape();
        const auto& vs = v.logical_shape();
        if (qs.rank() != 2 || ks.rank() != 2 || vs.rank() != 2 || h == 0 || hk == 0 ||
            h % hk != 0) {
            return false;
        }
        for (const auto* t : {&q, &k, &v}) {
            if (t->dtype() != DataType::BFLOAT16 ||
                t->layout() != tt::tt_metal::Layout::TILE) {
                return false;
            }
        }
        if (!(ks[0] == vs[0] && qs[1] % (32 * h) == 0 && vs[1] % (32 * hk) == 0 &&
              ks[1] * h == qs[1] * hk && qs[1] / h <= 256 && vs[1] / hk <= 256)) {
            return false;
        }
        const Dims s = varlen_dims(q, k, v, h, hk, true, true);
        return fit_blocks([&](uint32_t r, uint32_t kb) {
                   return varlen_cb_bytes(s, r, kb);
               }).first > 0;
    }

    ttnn::Tensor run_varlen(
        const ttnn::Tensor& q,
        const ttnn::Tensor& k,
        const ttnn::Tensor& v,
        uint32_t h,
        uint32_t hk,
        const ttnn::Tensor& plan,
        const ttnn::Tensor& masks,
        const std::vector<varlen::Row>& rows,
        float scale,
        const ttnn::ComputeKernelConfig& config,
        bool approx,
        ttnn::Tensor* stats,
        const ttnn::Tensor* key_mask
    ) {
        using namespace tt::tt_metal;
        const auto& qs = q.logical_shape();
        const bool with_stats = stats != nullptr;
        const Dims s = varlen_dims(q, k, v, h, hk, with_stats, key_mask != nullptr);
        check_key_mask(key_mask, k);
        TT_FATAL(
            rows.size() == s.sqt,
            "ktt_flash: the varlen plan has {} rows for {} query tile rows",
            rows.size(),
            s.sqt
        );
        const auto rk = fit_blocks([&](uint32_t r, uint32_t kb) {
            return varlen_cb_bytes(s, r, kb);
        });
        const uint32_t r = rk.first, kb = rk.second;
        TT_FATAL(
            r > 0,
            "ktt_flash: varlen circular buffers do not fit (supports_varlen() checks "
            "this)"
        );

        const float a = static_cast<float>(scale * 1.4426950408889634);
        const float thr = approx ? 32500.818359375f + 256.0f * 16.0f : 639.0f + 16.0f;
        const uint32_t a_bits = std::bit_cast<uint32_t>(a),
                       thr_bits = std::bit_cast<uint32_t>(thr);

        // Heads side by side, as the inputs: [Tq, H Dv] and the statistics [Tq, H 32].
        auto out = ttnn::empty(
            ttnn::Shape({qs[0], h * s.dvt * 32}),
            DataType::BFLOAT16,
            Layout::TILE,
            detail::device(),
            MemoryConfig{}
        );
        if (with_stats) {
            *stats = ttnn::empty(
                ttnn::Shape({qs[0], h * 32}),
                DataType::FLOAT32,
                Layout::TILE,
                detail::device(),
                MemoryConfig{}
            );
        }
        program::mark(program::Phase::Alloc);
        // The key holds what the kernels and buffers depend on, not the lengths.
        program::Key key("ktt_flash_varlen");
        key.add(static_cast<int64_t>(s.dt)).add(static_cast<int64_t>(s.dvt));
        key.add(static_cast<int64_t>(std::bit_cast<uint32_t>(scale))).add(approx);
        key.add(static_cast<int64_t>(config.math_fidelity))
            .add(static_cast<int64_t>(r))
            .add(static_cast<int64_t>(kb));
        key.add(with_stats).add(key_mask != nullptr);
        for (
            const auto* t : {&q, &k, &v, &plan, &masks, key_mask ? key_mask : &masks}
        ) {
            key.add(static_cast<int64_t>(t->memory_config().buffer_type()));
        }
        std::vector<ttnn::Tensor> io = {q, k, v, masks, plan};
        if (key_mask) {
            io.push_back(*key_mask);
        }
        if (with_stats) {
            io.push_back(*stats);
        }
        io.push_back(out);
        const ttnn::Tensor& stats_t = with_stats ? *stats : out;
        return program::launch_fresh(key, io, [&] {
            const auto grid = detail::device()->compute_with_storage_grid_size();
            const uint32_t gx = grid.x, gy = grid.y;
            const auto [length, per_core] = choose_varlen_lanes(s, rows, r, gx, gy);
            if (per_core.empty()) {
                throw std::runtime_error(
                    "kohakutt_native: sdpa_varlen: too many query tile rows per core "
                    "for one launch"
                );
            }
            const uint32_t plan_pages = static_cast<uint32_t>(
                plan.buffer()->num_pages()
            );

            KernelDescriptor::RuntimeArgs rd, wr, cp;
            for (uint32_t y = 0; y < gy; ++y) {
                for (uint32_t x = 0; x < gx; ++x) {
                    const auto& jobs = per_core[y * gx + x];
                    std::vector<uint32_t> ra = {
                        q.buffer()->address(),
                        k.buffer()->address(),
                        v.buffer()->address(),
                        masks.buffer()->address()
                    };
                    const auto lane = lane_args(x, y, length, gy, kVarJobs);
                    ra.insert(ra.end(), lane.begin(), lane.end());
                    std::vector<uint32_t> wa = {
                        out.buffer()->address(),
                        stats_t.buffer()->address(),
                        1u,
                        kVarJobs
                    };
                    std::vector<uint32_t> ca = {1u, kVarJobs};
                    for (uint32_t ji = 0; ji < kVarJobs; ++ji) {
                        const VarJob j = ji < jobs.size() ? jobs[ji] : VarJob{};
                        // A tile row holds its heads side by side: the row step is H
                        // tiles of D (the readers and the writer multiply `stride` by
                        // DT / DV).
                        ra.insert(
                            ra.end(),
                            {(j.start * s.h + j.head) * s.dt,
                             (j.key0 * s.hk + j.kv) * s.dt,
                             (j.key0 * s.hk + j.kv) * s.dvt,
                             j.rows,
                             j.nk,
                             s.h,
                             j.key0,
                             j.start}
                        );
                        wa.insert(
                            wa.end(), {(j.start * s.h + j.head) * s.dvt, j.rows, s.h}
                        );
                        ca.insert(ca.end(), {j.rows, j.start, j.nk, 1u, j.key0});
                    }
                    ra.insert(
                        ra.end(),
                        {plan.buffer()->address(),
                         plan_pages,
                         s.sqt * 64 + qs[0] % 32,
                         s.skt * 64 + k.logical_shape()[0] % 32,
                         s.hk * s.dt,
                         s.hk * s.dvt,
                         key_mask ? key_mask->buffer()->address() : 0u}
                    );
                    const CoreCoord core(x, y);
                    rd.push_back({core, ra});
                    wr.push_back({core, wa});
                    cp.push_back({core, ca});
                }
            }
            const CoreRangeSet all(
                CoreRange(CoreCoord(0, 0), CoreCoord(gx - 1, gy - 1))
            );
            program::mark(program::Phase::Args);

            ProgramDescriptor desc;
            const auto bf16 = tt::DataFormat::Float16_b, f32 = tt::DataFormat::Float32;
            const uint32_t w = s.dvt + 1 + (with_stats ? 1 : 0);
            std::vector<Cb> cbs = {
                {dev::cb_q, r * s.dt, kTile, bf16, r * s.dt},
                {dev::cb_k, 2 * s.dt * kb, kTile, bf16, s.dt * kb},
                {dev::cb_v, 2 * kb * s.dvt, kTile, bf16, kb * s.dvt},
                {dev::cb_ones, 1, kTile, bf16, 1},
                {dev::cb_p, 2 * kSc, kTile, bf16, kSc},
                {dev::cb_st, 2 * r, kTile32, f32, 2 * r, true},
                {dev::cb_o, r * w, kTile32, f32, r * w, true},
                {dev::cb_stok, 4, kToken, bf16, 1},
                {dev::cb_otok, r + 4, kToken, bf16, 1},
                {dev::cb_ctail, 1, kTile, bf16, 1},
                {dev::cb_causal, 1, kTile, bf16, 1},
                {dev::cb_tail, 1, kTile, bf16, 1},
                {dev::cb_out, 2 * s.dvt, kTile, bf16, s.dvt},
                {dev::cb_um, 4 * kSc, kTile, bf16, kSc},
                {dev::cb_plan,
                 2 * dev::kPlanPages,
                 kTile32,
                 tt::DataFormat::UInt32,
                 dev::kPlanPages},
            };
            if (with_stats) {
                cbs.push_back({dev::cb_lane, kLaneSlots, kTile32, f32, 1});
                cbs.push_back({dev::cb_stats, kStatSlots, kTile32, f32, 1});
            }
            if (key_mask) {
                cbs.push_back({dev::cb_mb, 2 * kb, kTile, bf16, kb});
            }
            add_cbs(desc, all, cbs);

            KernelDescriptor::Defines defines = {{"FLASH_VARLEN", "1"}};
            if (key_mask) {
                defines.emplace_back("FLASH_MASK_BCAST", "1");
            }
            std::vector<uint32_t> reader_ct = {s.dt, s.dvt, r, 0u, kb};
            const auto rargs = accessor_args({&q, &k, &v, &masks, &plan});
            reader_ct.insert(reader_ct.end(), rargs.begin(), rargs.end());
            if (key_mask) {
                const auto margs = accessor_args({key_mask});
                reader_ct.insert(reader_ct.end(), margs.begin(), margs.end());
            }
            std::vector<uint32_t> writer_ct = {s.dvt};
            const auto out_args = accessor_args({&out, &stats_t});
            writer_ct.insert(writer_ct.end(), out_args.begin(), out_args.end());
            KernelDescriptor::Defines writer_defines;
            if (with_stats) {
                writer_defines.emplace_back("FLASH_STATS", "1");
            }
            const NOC reader_noc = NOC::NOC_0;
            const NOC writer_noc = reader_noc == NOC::NOC_0 ? NOC::NOC_1 : NOC::NOC_0;
            TT_FATAL(
                writer_noc != reader_noc, "ktt_flash: reader and writer share a NoC"
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("attention/ktt_flash/reader.cpp"),
                    .core_ranges = all,
                    .compile_time_args = reader_ct,
                    .defines = defines,
                    .runtime_args = rd,
                    .config = DataMovementConfigDescriptor{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = reader_noc,
                        .noc_mode = NOC_MODE::DM_DEDICATED_NOC
                    }
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("attention/ktt_flash/writer.cpp"),
                    .core_ranges = all,
                    .compile_time_args = writer_ct,
                    .defines = writer_defines,
                    .runtime_args = wr,
                    .config = DataMovementConfigDescriptor{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = writer_noc,
                        .noc_mode = NOC_MODE::DM_DEDICATED_NOC
                    }
                }
            );
            ComputeConfigDescriptor compute{
                .math_fidelity = config.math_fidelity,
                .fp32_dest_acc_en = true,
                .dst_full_sync_en = false,
                .unpack_to_dest_mode = ComputeConfigDescriptor::UnpackToDestModes(
                    NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default
                ),
                .math_approx_mode = approx,
            };
            compute.unpack_to_dest_mode[dev::cb_o] = UnpackToDestMode::UnpackToDestFp32;
            compute
                .unpack_to_dest_mode[dev::cb_st] = UnpackToDestMode::UnpackToDestFp32;
            if (with_stats) {
                compute.unpack_to_dest_mode[dev::cb_lane] = UnpackToDestMode::
                    UnpackToDestFp32;
            }
            KernelDescriptor::Defines compute_defines = defines;
            if (approx) {
                compute_defines.emplace_back("FLASH_APPROX", "1");
            }
            if (with_stats) {
                compute_defines.emplace_back("FLASH_STATS", "1");
            }
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("attention/ktt_flash/compute.cpp"),
                    .core_ranges = all,
                    .compile_time_args =
                        {s.dt, s.dvt, r, 0u, kb, a_bits, thr_bits, 0u, 0u},
                    .defines = compute_defines,
                    .runtime_args = cp,
                    .config = compute
                }
            );
            for (uint32_t id = 0; id < 2; ++id) {
                desc.semaphores.push_back(
                    SemaphoreDescriptor{
                        .id = id,
                        .core_type = tt::CoreType::WORKER,
                        .core_ranges = all,
                        .initial_value = 0
                    }
                );
            }
            return program::Built{
                std::move(desc),
                {},
                {
                    {.kernel = 0,
                     .per_core = kVarReaderArgs + kVarReaderJobArgs * kVarJobs},
                    {.kernel = 1, .per_core = kWriterArgs + kWriterJobArgs * kVarJobs},
                    {.kernel = 2,
                     .per_core = kComputeArgs + kVarComputeJobArgs * kVarJobs},
                },
                rings_of(cbs)
            };
        });
    }

    namespace {

        // Runtime args of the packed backward programs (bw_dq_reader.cpp,
        // bw_dkv_reader.cpp, FLASH_VARLEN): {per core, per job}.
        constexpr uint32_t kVarDqReaderArgs = 18 + 8, kVarDqReaderJobArgs = 8;
        constexpr uint32_t kVarDkvReaderArgs = 17 + 7, kVarDkvReaderJobArgs = 7;
        static_assert(
            kVarDqReaderArgs + kVarDqReaderJobArgs * kVarJobs <=
            tt::tt_metal::max_runtime_args
        );
        static_assert(
            kVarDkvReaderArgs + kVarDkvReaderJobArgs * kVarJobs <=
            tt::tt_metal::max_runtime_args
        );

        // The plan's buffers of a packed backward program: a chunk's mask tiles twice,
        // the plan pages twice.
        constexpr uint32_t kVarBwBytes = 2 * kChunk * kTile +
            2 * dev::kPlanPages * kTile32;

        std::pair<uint32_t, uint32_t> varlen_dq_blocks(const Dims& s) {
            // With the key mask's rows of two key blocks.
            return fit_blocks([&](uint32_t r, uint32_t kb) {
                return dq_cb_bytes(s, r, kb, false) + kVarBwBytes + 2 * kb * kTile;
            });
        }

        std::pair<uint32_t, uint32_t> varlen_dkv_blocks(const Dims& s) {
            // With the key mask's row of each key row.
            return fit_blocks([&](uint32_t r, uint32_t qb) {
                return dkv_cb_bytes(s, r, qb, false) + kVarBwBytes + r * kTile;
            });
        }

        // One dK / dV job of a core: `rows` key tile rows from `start` of K / V head
        // `kv` against the query tiles [q_first, q_first + nq) of the head's query
        // heads.
        struct KvVarJob {
            uint32_t kv, start, rows, q_first, nq;
        };

        // dK / dV jobs per core (index y * gx + x), as plan_varlen() with key tile rows
        // and the key-side records `rows`: a lane job streams the query tiles any of
        // its rows sees (none: nq 0).
        std::vector<std::vector<KvVarJob>> plan_kv_varlen(
            const Dims& s,
            const std::vector<varlen::Row>& rows,
            uint32_t r,
            uint32_t gx,
            uint32_t gy,
            uint32_t length
        ) {
            std::vector<std::vector<KvVarJob>> per_core(gx * gy);
            const uint32_t lanes = gx * (gy / length);
            const uint32_t total = s.hk * s.skt;
            for (uint32_t i = 0; i < lanes; ++i) {
                const uint32_t x = i / (gy / length), y0 = i % (gy / length) * length;
                const uint32_t lo = static_cast<uint32_t>(
                    static_cast<uint64_t>(i) * total / lanes
                );
                const uint32_t hi = static_cast<uint32_t>(
                    static_cast<uint64_t>(i + 1) * total / lanes
                );
                for (uint32_t pos = lo; pos < hi;) {
                    const uint32_t kv = pos / s.skt, r0 = pos % s.skt;
                    const uint32_t n = std::min({hi - pos, s.skt - r0, length * r});
                    uint32_t first = UINT32_MAX, last = 0;
                    for (uint32_t t = r0; t < r0 + n; ++t) {
                        if (rows[t].kfirst <= rows[t].klast) {
                            first = std::min(first, rows[t].kfirst);
                            last = std::max(last, rows[t].klast);
                        }
                    }
                    const uint32_t q_first = first == UINT32_MAX ? 0 : first;
                    const uint32_t nq = first == UINT32_MAX ? 0 : last - first + 1;
                    const uint32_t per = ceil_div(n, length);
                    for (uint32_t y = y0; y < y0 + length; ++y) {
                        const uint32_t start = std::min(r0 + (y - y0) * per, r0 + n);
                        per_core[y * gx + x].push_back(
                            KvVarJob{
                                kv, start, std::min(per, r0 + n - start), q_first, nq
                            }
                        );
                    }
                    pos += n;
                }
            }
            return per_core;
        }

        // Lane length and per-core jobs of the packed dK / dV plan, chosen as
        // choose_varlen_lanes().
        std::pair<uint32_t, std::vector<std::vector<KvVarJob>>> choose_kv_varlen_lanes(
            const Dims& s,
            const std::vector<varlen::Row>& rows,
            uint32_t r,
            uint32_t gx,
            uint32_t gy
        ) {
            uint32_t length = 1;
            std::vector<std::vector<KvVarJob>> per_core;
            uint64_t best = UINT64_MAX;
            for (uint32_t len = 1; len <= gy; ++len) {
                if (gy % len != 0) {
                    continue;
                }
                auto p = plan_kv_varlen(s, rows, r, gx, gy, len);
                uint64_t most = 0;
                size_t jobs = 0;
                for (const auto& core : p) {
                    uint64_t tiles = 0;
                    for (const auto& j : core) {
                        for (uint32_t t = j.start; t < j.start + j.rows; ++t) {
                            tiles += rows[t].kfirst <= rows[t].klast
                                ? rows[t].klast - rows[t].kfirst + 1
                                : 0;
                        }
                    }
                    most = std::max(most, tiles);
                    jobs = std::max(jobs, core.size());
                }
                if (jobs <= kVarJobs && most <= best) {
                    best = most;
                    length = len;
                    per_core = std::move(p);
                }
            }
            return {length, std::move(per_core)};
        }

    }  // namespace

    bool supports_varlen_backward(
        const ttnn::Tensor& q,
        const ttnn::Tensor& k,
        const ttnn::Tensor& v,
        uint32_t h,
        uint32_t hk
    ) {
        if (!supports_varlen(q, k, v, h, hk)) {
            return false;
        }
        const Dims s = varlen_dims(q, k, v, h, hk, true);
        return varlen_dq_blocks(s).first > 0 && varlen_dkv_blocks(s).first > 0;
    }

    std::array<ttnn::Tensor, 3> backward_varlen(
        const ttnn::Tensor& grad,
        const ttnn::Tensor& out,
        const ttnn::Tensor& q,
        const ttnn::Tensor& k,
        const ttnn::Tensor& v,
        const ttnn::Tensor& stats,
        uint32_t h,
        uint32_t hk,
        const ttnn::Tensor& plan,
        const ttnn::Tensor& masks,
        const std::vector<varlen::Row>& rows,
        const ttnn::Tensor& key_plan,
        const ttnn::Tensor& key_masks,
        const std::vector<varlen::Row>& key_rows,
        float scale,
        const ttnn::ComputeKernelConfig& config,
        const ttnn::Tensor* key_mask
    ) {
        using namespace tt::tt_metal;
        const auto& qs = q.logical_shape();
        const auto& ks = k.logical_shape();
        check_key_mask(key_mask, k);
        const Dims s = varlen_dims(q, k, v, h, hk, true);
        TT_FATAL(
            rows.size() == s.sqt && key_rows.size() == s.skt,
            "ktt_flash: the varlen plans have {} / {} rows for {} query and {} key "
            "tile rows",
            rows.size(),
            key_rows.size(),
            s.sqt,
            s.skt
        );
        const auto q_blocks = varlen_dq_blocks(s), kv_blocks = varlen_dkv_blocks(s);
        const uint32_t r = q_blocks.first, kb = q_blocks.second, rk = kv_blocks.first,
                       qb = kv_blocks.second;
        TT_FATAL(
            r > 0 && rk > 0, "ktt_flash: varlen backward circular buffers do not fit"
        );
        const uint32_t a_bits = std::bit_cast<uint32_t>(
            static_cast<float>(scale * 1.4426950408889634)
        );
        const uint32_t scale_bits = std::bit_cast<uint32_t>(scale);
        const auto fidelity = config.math_fidelity;
        const uint32_t tails_q = s.sqt * 64 + qs[0] % 32,
                       tails_k = s.skt * 64 + ks[0] % 32;

        auto dd = ttnn::empty(
            ttnn::Shape({qs[0], h * 32}),
            DataType::FLOAT32,
            Layout::TILE,
            detail::device(),
            MemoryConfig{}
        );
        auto dq = ttnn::empty(
            q.logical_shape(),
            DataType::BFLOAT16,
            Layout::TILE,
            detail::device(),
            MemoryConfig{}
        );
        auto dk = ttnn::empty(
            k.logical_shape(),
            DataType::BFLOAT16,
            Layout::TILE,
            detail::device(),
            MemoryConfig{}
        );
        auto dv = ttnn::empty(
            v.logical_shape(),
            DataType::BFLOAT16,
            Layout::TILE,
            detail::device(),
            MemoryConfig{}
        );
        program::mark(program::Phase::Alloc);
        const auto bf16 = tt::DataFormat::Float16_b, f32 = tt::DataFormat::Float32;
        KernelDescriptor::Defines defines = {{"FLASH_VARLEN", "1"}};
        if (key_mask) {
            defines.emplace_back("FLASH_MASK_BCAST", "1");
        }
        const uint32_t key_mask_address = key_mask ? key_mask->buffer()->address() : 0u;
        auto base_key = [&](const char* name,
                            std::initializer_list<const ttnn::Tensor*> tensors) {
            program::Key key(name);
            key.add(static_cast<int64_t>(s.dt)).add(static_cast<int64_t>(s.dvt));
            key.add(static_cast<int64_t>(scale_bits))
                .add(static_cast<int64_t>(fidelity));
            key.add(key_mask != nullptr);
            if (key_mask) {
                key.add(static_cast<int64_t>(key_mask->memory_config().buffer_type()));
            }
            for (const auto* t : tensors) {
                key.add(static_cast<int64_t>(t->memory_config().buffer_type()));
            }
            return key;
        };

        // dQ and the statistics tiles. io: q, k, v, dO, O, log-sum-exp, masks, plan,
        // statistics, dQ.
        auto key_q = base_key(
            "ktt_flash_varlen_dq", {&q, &k, &v, &grad, &out, &stats, &masks, &plan}
        );
        key_q.add(static_cast<int64_t>(r)).add(static_cast<int64_t>(kb));
        std::vector<ttnn::Tensor> io_q = {q, k, v, grad, out, stats, masks, plan};
        if (key_mask) {
            io_q.push_back(*key_mask);
        }
        io_q.push_back(dd);
        io_q.push_back(dq);
        dq = program::launch_fresh(key_q, io_q, [&] {
            namespace c = bw::dq;
            const auto grid = detail::device()->compute_with_storage_grid_size();
            const uint32_t gx = grid.x, gy = grid.y;
            const auto [length, per_core] = choose_varlen_lanes(s, rows, r, gx, gy);
            if (per_core.empty()) {
                throw std::runtime_error(
                    "kohakutt_native: sdpa_varlen backward: too many query tile rows "
                    "per core for one launch"
                );
            }
            const uint32_t plan_pages = static_cast<uint32_t>(
                plan.buffer()->num_pages()
            );
            KernelDescriptor::RuntimeArgs rd, wr, cp;
            for (uint32_t y = 0; y < gy; ++y) {
                for (uint32_t x = 0; x < gx; ++x) {
                    const auto& jobs = per_core[y * gx + x];
                    std::vector<uint32_t> ra;
                    for (const auto* t : {&q, &k, &v, &grad, &out, &stats}) {
                        ra.push_back(t->buffer()->address());
                    }
                    const auto lane = lane_args(x, y, length, gy, kVarJobs);
                    ra.insert(ra.end(), lane.begin(), lane.end());
                    std::vector<uint32_t> wa = {
                        dq.buffer()->address(), dd.buffer()->address(), 1u, kVarJobs
                    };
                    std::vector<uint32_t> ca = {1u, kVarJobs};
                    for (uint32_t ji = 0; ji < kVarJobs; ++ji) {
                        const VarJob j = ji < jobs.size() ? jobs[ji] : VarJob{};
                        const uint32_t row = j.start * s.h + j.head,
                                       key = j.key0 * s.hk + j.kv;
                        ra.insert(
                            ra.end(),
                            {row,
                             key * s.dt,
                             key * s.dvt,
                             j.rows,
                             j.nk,
                             s.h,
                             j.key0,
                             j.start}
                        );
                        wa.insert(wa.end(), {row, j.rows, s.h});
                        ca.insert(ca.end(), {j.rows, j.start, j.nk, 1u, j.key0});
                    }
                    ra.insert(
                        ra.end(),
                        {masks.buffer()->address(),
                         plan.buffer()->address(),
                         plan_pages,
                         tails_q,
                         tails_k,
                         s.hk * s.dt,
                         s.hk * s.dvt,
                         key_mask_address}
                    );
                    const CoreCoord core(x, y);
                    rd.push_back({core, ra});
                    wr.push_back({core, wa});
                    cp.push_back({core, ca});
                }
            }
            const CoreRangeSet all(
                CoreRange(CoreCoord(0, 0), CoreCoord(gx - 1, gy - 1))
            );
            program::mark(program::Phase::Args);
            ProgramDescriptor desc;
            std::vector<Cb> cbs = {
                {c::cb_q, r * s.dt, kTile, bf16, r * s.dt},
                {c::cb_k, 2 * s.dt * kb, kTile, bf16, s.dt * kb},
                {c::cb_v, 2 * kb * s.dvt, kTile, bf16, kb * s.dvt},
                {c::cb_do, r * s.dvt, kTile, bf16, r * s.dvt},
                {c::cb_oin, r * s.dvt, kTile, bf16, r * s.dvt},
                {c::cb_lse, r, kTile32, f32, r},
                {c::cb_ones, 1, kTile, bf16, 1},
                {c::cb_prod, s.dvt, kTile, bf16, s.dvt},
                {c::cb_ds, 2 * kChunk, kTile, bf16, kChunk},
                {c::cb_acc, r * s.dt, kTile32, f32, r * s.dt, true},
                {c::cb_out, 2 * s.dt, kTile, bf16, s.dt},
                {c::cb_dout, 2, kTile32, f32, 1},
                {c::cb_tok, 2, kToken, bf16, 1},
                {c::cb_w, r, kTile32, f32, r, true},
                {c::cb_um, 2 * kChunk, kTile, bf16, kChunk},
                {c::cb_plan,
                 2 * dev::kPlanPages,
                 kTile32,
                 tt::DataFormat::UInt32,
                 dev::kPlanPages},
            };
            if (key_mask) {
                cbs.push_back({c::cb_mb, 2 * kb, kTile, bf16, kb});
            }
            add_cbs(desc, all, cbs);
            std::vector<uint32_t> reader_ct = {s.dt, s.dvt, r, 0u, kb};
            const auto rargs = accessor_args(
                {&q, &k, &v, &grad, &out, &stats, &masks, &plan}
            );
            reader_ct.insert(reader_ct.end(), rargs.begin(), rargs.end());
            if (key_mask) {
                const auto margs = accessor_args({key_mask});
                reader_ct.insert(reader_ct.end(), margs.begin(), margs.end());
            }
            std::vector<uint32_t> writer_ct = {s.dt, 1u};
            const auto wargs = accessor_args({&dq, &dd});
            writer_ct.insert(writer_ct.end(), wargs.begin(), wargs.end());
            add_kernels(
                desc,
                all,
                "bw_dq",
                reader_ct,
                writer_ct,
                {s.dt, s.dvt, r, kb, a_bits, scale_bits, 0u, 0u, 0u, 0u},
                rd,
                wr,
                cp,
                defines,
                bw_compute(fidelity, {c::cb_lse, c::cb_acc, c::cb_w}),
                defines
            );
            return program::Built{
                std::move(desc),
                {},
                {
                    {.kernel = 0,
                     .per_core = kVarDqReaderArgs + kVarDqReaderJobArgs * kVarJobs},
                    {.kernel = 1,
                     .per_core = kDqWriterArgs + kDqWriterJobArgs * kVarJobs},
                    {.kernel = 2, .per_core = kDqComputeArgs + 5 * kVarJobs},
                },
                rings_of(cbs)
            };
        });

        // dK and dV. io: q, k, v, dO, the statistics tiles, key-side masks and plan,
        // dK, dV.
        const uint32_t group = s.h / s.hk;
        auto key_kv = base_key(
            "ktt_flash_varlen_dkv", {&q, &k, &v, &grad, &dd, &key_masks, &key_plan}
        );
        key_kv.add(static_cast<int64_t>(rk))
            .add(static_cast<int64_t>(qb))
            .add(static_cast<int64_t>(group));
        std::vector<ttnn::Tensor> io_kv = {q, k, v, grad, dd, key_masks, key_plan};
        if (key_mask) {
            io_kv.push_back(*key_mask);
        }
        io_kv.push_back(dk);
        io_kv.push_back(dv);
        dv = program::launch_fresh(key_kv, io_kv, [&] {
            namespace c = bw::dkv;
            const auto grid = detail::device()->compute_with_storage_grid_size();
            const uint32_t gx = grid.x, gy = grid.y;
            const auto [length, per_core] = choose_kv_varlen_lanes(
                s, key_rows, rk, gx, gy
            );
            if (per_core.empty()) {
                throw std::runtime_error(
                    "kohakutt_native: sdpa_varlen backward: too many key tile rows per "
                    "core for one launch"
                );
            }
            const uint32_t plan_pages = static_cast<uint32_t>(
                key_plan.buffer()->num_pages()
            );
            KernelDescriptor::RuntimeArgs rd, wr, cp;
            for (uint32_t y = 0; y < gy; ++y) {
                for (uint32_t x = 0; x < gx; ++x) {
                    const auto& jobs = per_core[y * gx + x];
                    std::vector<uint32_t> ra;
                    for (
                        const ttnn::Tensor* t :
                        std::initializer_list<const ttnn::Tensor*>{
                            &q, &k, &v, &grad, &dd
                        }
                    ) {
                        ra.push_back(t->buffer()->address());
                    }
                    const auto lane = lane_args(x, y, length, gy, kVarJobs);
                    ra.insert(ra.end(), lane.begin(), lane.end());
                    std::vector<uint32_t> wa = {
                        dk.buffer()->address(), dv.buffer()->address(), 1u, kVarJobs
                    };
                    std::vector<uint32_t> ca = {1u, kVarJobs};
                    for (uint32_t ji = 0; ji < kVarJobs; ++ji) {
                        const KvVarJob j = ji < jobs.size() ? jobs[ji] : KvVarJob{};
                        const uint32_t row = j.start * s.hk + j.kv;
                        ra.insert(
                            ra.end(),
                            {row, j.rows, s.hk, j.kv * group, j.q_first, j.nq, j.start}
                        );
                        wa.insert(wa.end(), {row, j.rows, s.hk});
                        ca.insert(ca.end(), {j.rows, j.start, j.nq, 1u, j.q_first});
                    }
                    ra.insert(
                        ra.end(),
                        {key_masks.buffer()->address(),
                         key_plan.buffer()->address(),
                         plan_pages,
                         tails_q,
                         tails_k,
                         s.h,
                         key_mask_address}
                    );
                    const CoreCoord core(x, y);
                    rd.push_back({core, ra});
                    wr.push_back({core, wa});
                    cp.push_back({core, ca});
                }
            }
            const CoreRangeSet all(
                CoreRange(CoreCoord(0, 0), CoreCoord(gx - 1, gy - 1))
            );
            program::mark(program::Phase::Args);
            ProgramDescriptor desc;
            const uint32_t w = s.dt + s.dvt;
            std::vector<Cb> cbs = {
                {c::cb_k, rk * s.dt, kTile, bf16, rk * s.dt},
                {c::cb_v, rk * s.dvt, kTile, bf16, rk * s.dvt},
                {c::cb_q, 2 * qb * s.dt, kTile, bf16, qb * s.dt},
                {c::cb_do, 2 * qb * s.dvt, kTile, bf16, qb * s.dvt},
                {c::cb_wd, 2 * qb, kTile32, f32, qb},
                {c::cb_p, 2 * kChunk, kTile, bf16, kChunk},
                {c::cb_ds, 2 * kChunk, kTile, bf16, kChunk},
                {c::cb_dk, rk * s.dt, kTile32, f32, rk * s.dt, true},
                {c::cb_dv, rk * s.dvt, kTile32, f32, rk * s.dvt, true},
                {c::cb_out, 2 * w, kTile, bf16, w},
                {c::cb_tok, 2, kToken, bf16, 1},
                {c::cb_qt, kChunk * s.dt, kTile, bf16, kChunk * s.dt},
                {c::cb_dot, kChunk * s.dvt, kTile, bf16, kChunk * s.dvt},
                {c::cb_fin, w, kTile, bf16, w},
                {c::cb_um, 2 * kChunk, kTile, bf16, kChunk},
                {c::cb_plan,
                 2 * dev::kPlanPages,
                 kTile32,
                 tt::DataFormat::UInt32,
                 dev::kPlanPages},
            };
            if (key_mask) {
                cbs.push_back({c::cb_mb, rk, kTile, bf16, rk});
            }
            add_cbs(desc, all, cbs);
            std::vector<uint32_t> reader_ct = {s.dt, s.dvt, rk, qb, 0u, group, 1u};
            const auto rargs = accessor_args(
                {&q, &k, &v, &grad, &dd, &key_masks, &key_plan}
            );
            reader_ct.insert(reader_ct.end(), rargs.begin(), rargs.end());
            if (key_mask) {
                const auto margs = accessor_args({key_mask});
                reader_ct.insert(reader_ct.end(), margs.begin(), margs.end());
            }
            std::vector<uint32_t> writer_ct = {s.dt, s.dvt};
            const auto wargs = accessor_args({&dk, &dv});
            writer_ct.insert(writer_ct.end(), wargs.begin(), wargs.end());
            add_kernels(
                desc,
                all,
                "bw_dkv",
                reader_ct,
                writer_ct,
                {s.dt, s.dvt, rk, qb, group, a_bits, scale_bits, 0u, 0u},
                rd,
                wr,
                cp,
                defines,
                bw_compute(fidelity, {c::cb_wd, c::cb_dk, c::cb_dv}),
                defines
            );
            return program::Built{
                std::move(desc),
                {},
                {
                    {.kernel = 0,
                     .per_core = kVarDkvReaderArgs + kVarDkvReaderJobArgs * kVarJobs},
                    {.kernel = 1,
                     .per_core = kDkvWriterArgs + kDkvWriterJobArgs * kVarJobs},
                    {.kernel = 2,
                     .per_core = kDkvComputeArgs + kDkvComputeJobArgs * kVarJobs},
                },
                rings_of(cbs)
            };
        });
        return {dq, dk, dv};
    }

}  // namespace ktt::bridge::ktt_flash
