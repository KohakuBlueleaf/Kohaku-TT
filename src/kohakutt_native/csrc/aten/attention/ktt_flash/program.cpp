// Host side of ktt_flash: block sizes under the program L1 budget, the lane plan (each core's
// jobs), circular buffers, kernels and runtime args.
#include "aten/attention/ktt_flash/program.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <numeric>
#include <tuple>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "aten/attention/ktt_flash/flash.hpp"
#include "core/device/handle.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"

namespace ktt::bridge::ktt_flash {
    namespace {

        namespace dev = ::ktt_flash;

        constexpr uint32_t kTile = 2048;    // bf16 tile
        constexpr uint32_t kTile32 = 4096;  // fp32 tile
        constexpr uint32_t kToken = 64;     // token page
        constexpr uint32_t kSc = dev::kSubChunk;
        // Jobs per core: the reader takes 16 + 7 per job runtime args, at most
        // tt::tt_metal::max_runtime_args (341).
        constexpr uint32_t kMaxJobs = 46;
        static_assert(16 + 7 * kMaxJobs <= tt::tt_metal::max_runtime_args);

        uint32_t ceil_div(uint32_t a, uint32_t b) {
            return (a + b - 1) / b;
        }

        // Problem in tiles. Mask: batches / heads of the mask (1 or the full count), `mask_rows`
        // for a per-query-row mask, `mask_bcast` for a mask row broadcast over the queries.
        struct Dims {
            uint32_t b, h, hk, sqt, skt, dt, dvt;
            bool causal;
            uint32_t mb, mh;
            bool mask_rows, mask_bcast;

            // Last key tile query tile row i sees.
            uint32_t last_key(uint32_t i) const {
                return causal ? std::min(i, skt - 1) : skt - 1;
            }
        };

        Dims dims_of(
            const ttnn::Tensor& q, const ttnn::Tensor& k, const ttnn::Tensor& v,
            const std::optional<ttnn::Tensor>& mask, bool causal
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
            return s;
        }

        // One job of a core: up to R query tile rows of one head (first row `start`, rows
        // `stride` apart) against key tiles [0, nk); first pages of Q, K, V, O and the mask.
        struct Job {
            uint32_t q0, k0, v0, o0, rows, start, nk, stride, u0;
        };

        // A query chunk of a lane: rows [r0, r0 + n) of head (bi, hi), K / V head g.
        struct Chunk {
            uint32_t bi, hi, g, r0, n;
        };

        // Jobs per core (index y * gx + x) for lanes of `length` cores down each grid column (the
        // sender at the top). A lane job is up to length * r query rows of one head, split over
        // the lane: contiguous rows, or (causal) rows `length` apart. Non-causal lanes take even
        // ranges of the (batch, head, row) order; causal lanes take chunks largest work first
        // onto the least loaded lane.
        std::vector<std::vector<Job>> plan(const Dims& s, uint32_t r, uint32_t gx, uint32_t gy, uint32_t length) {
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
                        const uint32_t r0 = c * s.sqt / n_chunks, r1 = (c + 1) * s.sqt / n_chunks;
                        const uint64_t work = static_cast<uint64_t>(ceil_div(r1 - r0, length)) *
                                              (s.last_key(r0) + s.last_key(r1 - 1) + 2) / 2;
                        pieces.push_back({work, Chunk{bi, hi, hi / group, r0, r1 - r0}});
                    }
                }
                std::stable_sort(pieces.begin(), pieces.end(), [](const auto& a, const auto& b) {
                    return a.first > b.first;
                });
                // Least loaded among the lanes below kMaxJobs chunks (any lane once all are full).
                std::vector<uint64_t> load(lanes.size(), 0);
                for (const auto& [work, chunk] : pieces) {
                    size_t i = lanes.size();
                    for (size_t j = 0; j < lanes.size(); ++j) {
                        if (lane_chunks[j].size() < kMaxJobs && (i == lanes.size() || load[j] < load[i])) {
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
                    const uint32_t lo = static_cast<uint32_t>(static_cast<uint64_t>(i) * total / lanes.size());
                    const uint32_t hi_ = static_cast<uint32_t>(static_cast<uint64_t>(i + 1) * total / lanes.size());
                    for (uint32_t pos = lo; pos < hi_;) {
                        const uint32_t head = pos / s.sqt, r0 = pos % s.sqt;
                        const uint32_t n = std::min({hi_ - pos, s.sqt - r0, length * r});
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
                            rows = start < c.r0 + c.n ? ceil_div(c.r0 + c.n - start, length) : 0;
                            start = std::min(start, c.r0 + c.n);
                        } else {
                            const uint32_t per = ceil_div(c.n, length);
                            start = std::min(c.r0 + (y - y0) * per, c.r0 + c.n);
                            stride = 1;
                            rows = std::min(per, c.r0 + c.n - start);
                        }
                        const uint32_t nk = s.last_key(c.r0 + c.n - 1) + 1;
                        const uint32_t mbi = s.mb > 1 ? c.bi : 0, mhi = s.mh > 1 ? c.hi : 0;
                        const uint32_t u0 = s.mask_rows ? ((mbi * s.mh + mhi) * s.sqt + start) * s.skt
                                                        : (mbi * s.mh + mhi) * s.skt;
                        const uint32_t qrow = (c.bi * s.h + c.hi) * s.sqt + start;
                        jobs.push_back(Job{
                            qrow * s.dt, (c.bi * s.hk + c.g) * s.skt * s.dt, (c.bi * s.hk + c.g) * s.skt * s.dvt,
                            qrow * s.dvt, rows, start, nk, stride, u0
                        });
                    }
                }
            }
            return per_core;
        }

        // The busiest core's score tiles.
        uint64_t max_tiles(const Dims& s, const std::vector<std::vector<Job>>& per_core) {
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
            const uint32_t w = s.dvt + 1;
            uint32_t bf16 = r * s.dt + 2 * s.dt * kb + 2 * kb * s.dvt + 1 + 2 * kSc + 2 * s.dvt + 4;
            if (s.mask_rows) {
                bf16 += 4 * kSc;
            }
            if (s.mask_bcast) {
                bf16 += 2 * kb;
            }
            // fp32: the accumulators and the state tiles.
            return bf16 * kTile + (r * w + 2 * r) * kTile32 + (r + 8) * kToken;
        }

        // Query rows per job (R) and key tiles per block (KB, whole sub-chunks) under the circular
        // buffer budget (allocator read: on the stream worker); r = 0 when nothing fits.
        std::pair<uint32_t, uint32_t> blocks(const Dims& s) {
            const uint32_t budget = program::cb_budget();
            uint32_t r = 13, kb = 12;
            while (cb_bytes(s, r, kb) > budget) {
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

        // True when lanes of one core (the most lanes) keep every core at kMaxJobs jobs or fewer.
        // Non-causal: a lane's m = ceil(T / cores) consecutive rows give at most m / r full
        // chunks, one per head boundary and the last. Causal: ceil(Sqt / r) chunks per head,
        // capped at kMaxJobs per lane.
        bool args_fit(const Dims& s, uint32_t r, uint32_t cores) {
            if (s.causal) {
                return static_cast<uint64_t>(s.b) * s.h * ceil_div(s.sqt, r) <= static_cast<uint64_t>(kMaxJobs) * cores;
            }
            const uint64_t m = (static_cast<uint64_t>(s.b) * s.h * s.sqt + cores - 1) / cores;
            return m / r + (m + s.sqt - 1) / s.sqt + 1 <= kMaxJobs;
        }

    }  // namespace

    bool supports(
        const ttnn::Tensor& q, const ttnn::Tensor& k, const ttnn::Tensor& v,
        const std::optional<ttnn::Tensor>& mask, bool causal
    ) {
        using tt::tt_metal::DataType;
        const auto& qs = q.logical_shape();
        const auto& ks = k.logical_shape();
        const auto& vs = v.logical_shape();
        if (qs.rank() != 4 || ks.rank() != 4 || vs.rank() != 4) {
            return false;
        }
        for (const auto* t : {&q, &k, &v}) {
            if (t->dtype() != DataType::BFLOAT16 || t->layout() != tt::tt_metal::Layout::TILE) {
                return false;
            }
        }
        const bool shapes = (
            qs[0] == ks[0] && ks[0] == vs[0] && ks[1] == vs[1] && qs[1] % ks[1] == 0 && ks[2] == vs[2] &&
            qs[3] == ks[3] && qs[3] % 32 == 0 && vs[3] % 32 == 0 && qs[3] <= 256 && vs[3] <= 256 &&
            (!causal || (qs[2] == ks[2] && !mask))
        );
        if (!shapes) {
            return false;
        }
        if (mask) {
            const auto& ms = mask->logical_shape();
            if (!(ms.rank() == 4 && mask->dtype() == DataType::BFLOAT16 &&
                  mask->layout() == tt::tt_metal::Layout::TILE && (ms[0] == 1 || ms[0] == qs[0]) &&
                  (ms[1] == 1 || ms[1] == qs[1]) && (ms[2] == 1 || ms[2] == qs[2]) && ms[3] == ks[2])) {
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
        return ttnn::multiply(ttnn::clamp(mask, -1e4f, std::nullopt), 1.0f / scale);
    }

    ttnn::Tensor run(
        const ttnn::Tensor& q, const ttnn::Tensor& k, const ttnn::Tensor& v,
        const std::optional<ttnn::Tensor>& mask, float scale, bool causal,
        const ttnn::ComputeKernelConfig& config, bool approx
    ) {
        using namespace tt::tt_metal;
        const auto& qs = q.logical_shape();
        const Dims s = dims_of(q, k, v, mask, causal);
        const bool row_mask = s.mask_rows, bcast_mask = s.mask_bcast;
        const uint32_t tail = k.logical_shape()[2] % 32;
        const auto rk = blocks(s);
        const uint32_t r = rk.first, kb = rk.second;
        TT_FATAL(r > 0, "ktt_flash: circular buffers do not fit (supports() checks this)");

        const float a = static_cast<float>(scale * 1.4426950408889634);
        const float thr = approx ? 32500.818359375f + 256.0f * 16.0f : 639.0f + 16.0f;
        const uint32_t a_bits = std::bit_cast<uint32_t>(a), thr_bits = std::bit_cast<uint32_t>(thr);

        auto out = ttnn::empty(
            ttnn::Shape({qs[0], qs[1], qs[2], v.logical_shape()[3]}), DataType::BFLOAT16, Layout::TILE,
            detail::device(), MemoryConfig{}
        );
        program::mark(program::Phase::Alloc);
        program::Key key("ktt_flash");
        key.add(q).add(k).add(v).add(static_cast<int64_t>(std::bit_cast<uint32_t>(scale))).add(causal).add(approx);
        if (mask) {
            key.add(*mask);
        }
        key.add(static_cast<int64_t>(config.math_fidelity)).add(static_cast<int64_t>(r)).add(static_cast<int64_t>(kb));
        std::vector<ttnn::Tensor> io = {q, k, v};
        if (mask) {
            io.push_back(*mask);
        }
        io.push_back(out);
        const uint32_t out_index = static_cast<uint32_t>(io.size() - 1);
        const uint32_t mask_index = mask ? 3u : 0u;
        return program::launch(key, io, [&] {
            const auto grid = detail::device()->compute_with_storage_grid_size();
            const uint32_t gx = grid.x, gy = grid.y;
            // Lane length: the fewest score tiles on the busiest core, then the longest lane
            // (fewest K / V reads).
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
            TT_FATAL(!per_core.empty(), "ktt_flash: no lane plan fits the runtime args (supports() checks this)");
            auto phys = [&](uint32_t x, uint32_t y) {
                const auto c = detail::device()->worker_core_from_logical_core(CoreCoord(x, y));
                return std::pair<uint32_t, uint32_t>(c.x, c.y);
            };

            KernelDescriptor::RuntimeArgs rd, wr, cp;
            for (uint32_t y = 0; y < gy; ++y) {
                const bool sender = y % length == 0;
                for (uint32_t x = 0; x < gx; ++x) {
                    const auto& jobs = per_core[y * gx + x];
                    const uint32_t y0 = y - y % length;
                    const auto top = phys(x, y0), first = phys(x, std::min(y0 + 1, gy - 1));
                    const auto last = phys(x, y0 + length - 1);
                    std::vector<uint32_t> ra = {
                        q.buffer()->address(), k.buffer()->address(), v.buffer()->address(),
                        (mask ? mask->buffer() : q.buffer())->address(), 1u, first.first, first.second,
                        last.first, last.second, length - 1, top.first, top.second, 0u, 1u, sender ? 1u : 0u,
                        static_cast<uint32_t>(jobs.size())
                    };
                    std::vector<uint32_t> wa = {out.buffer()->address(), 1u, static_cast<uint32_t>(jobs.size())};
                    std::vector<uint32_t> ca = {1u, static_cast<uint32_t>(jobs.size())};
                    for (const auto& j : jobs) {
                        ra.insert(ra.end(), {j.q0, j.k0, j.v0, j.rows, j.nk, j.stride, j.u0});
                        wa.insert(wa.end(), {j.o0, j.rows, j.stride});
                        ca.insert(ca.end(), {j.rows, j.start, j.nk, j.stride});
                    }
                    const CoreCoord core(x, y);
                    rd.push_back({core, ra});
                    wr.push_back({core, wa});
                    cp.push_back({core, ca});
                }
            }
            const CoreRangeSet all(CoreRange(CoreCoord(0, 0), CoreCoord(gx - 1, gy - 1)));
            program::mark(program::Phase::Args);

            ProgramDescriptor desc;
            const auto bf16 = tt::DataFormat::Float16_b;
            struct Cb {
                uint32_t index, pages, page;
                tt::DataFormat format;
            };
            std::vector<Cb> cbs = {
                {dev::cb_q, r * s.dt, kTile, bf16}, {dev::cb_k, 2 * s.dt * kb, kTile, bf16},
                {dev::cb_v, 2 * kb * s.dvt, kTile, bf16}, {dev::cb_ones, 1, kTile, bf16},
                {dev::cb_p, 2 * kSc, kTile, bf16}, {dev::cb_st, 2 * r, kTile32, tt::DataFormat::Float32},
                {dev::cb_o, r * (s.dvt + 1), kTile32, tt::DataFormat::Float32}, {dev::cb_stok, 4, kToken, bf16},
                {dev::cb_otok, r + 4, kToken, bf16}, {dev::cb_ident, 1, kTile, bf16},
                {dev::cb_causal, 1, kTile, bf16}, {dev::cb_col0, 1, kTile, bf16}, {dev::cb_tail, 1, kTile, bf16},
                {dev::cb_out, 2 * s.dvt, kTile, bf16},
            };
            if (row_mask) {
                cbs.push_back({dev::cb_um, 4 * kSc, kTile, bf16});
            }
            if (bcast_mask) {
                cbs.push_back({dev::cb_mb, 2 * kb, kTile, bf16});
            }
            for (const auto& cb : cbs) {
                desc.cbs.push_back(CBDescriptor{
                    .total_size = cb.pages * cb.page,
                    .core_ranges = all,
                    .format_descriptors = {CBFormatDescriptor{
                        .buffer_index = static_cast<uint8_t>(cb.index), .data_format = cb.format, .page_size = cb.page
                    }}
                });
            }

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
                const auto args = TensorAccessorArgs(*t->buffer()).get_compile_time_args();
                reader_ct.insert(reader_ct.end(), args.begin(), args.end());
            }
            std::vector<uint32_t> writer_ct = {s.dvt};
            const auto out_ct = TensorAccessorArgs(*out.buffer()).get_compile_time_args();
            writer_ct.insert(writer_ct.end(), out_ct.begin(), out_ct.end());
            // Every reader on NoC0 (senders multicast down their column, NoC0 runs +y), every
            // writer on NoC1.
            const NOC reader_noc = NOC::NOC_0;
            const NOC writer_noc = reader_noc == NOC::NOC_0 ? NOC::NOC_1 : NOC::NOC_0;
            TT_FATAL(writer_noc != reader_noc, "ktt_flash: reader and writer share a NoC");
            desc.kernels.push_back(KernelDescriptor{
                .kernel_source = program::source("attention/ktt_flash/reader.cpp"),
                .core_ranges = all,
                .compile_time_args = reader_ct,
                .defines = defines,
                .runtime_args = rd,
                .config = DataMovementConfigDescriptor{
                    .processor = DataMovementProcessor::RISCV_1, .noc = reader_noc,
                    .noc_mode = NOC_MODE::DM_DEDICATED_NOC
                }
            });
            desc.kernels.push_back(KernelDescriptor{
                .kernel_source = program::source("attention/ktt_flash/writer.cpp"),
                .core_ranges = all,
                .compile_time_args = writer_ct,
                .runtime_args = wr,
                .config = DataMovementConfigDescriptor{
                    .processor = DataMovementProcessor::RISCV_0, .noc = writer_noc,
                    .noc_mode = NOC_MODE::DM_DEDICATED_NOC
                }
            });
            std::vector<program::Slot> slots;
            for (uint32_t pos = 0; pos < 3; ++pos) {
                slots.push_back({.kernel = 0, .entry = program::Slot::kEveryCore, .position = pos, .tensor = pos});
            }
            slots.push_back({.kernel = 0, .entry = program::Slot::kEveryCore, .position = 3, .tensor = mask_index});
            slots.push_back({.kernel = 1, .entry = program::Slot::kEveryCore, .position = 0, .tensor = out_index});
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
            compute.unpack_to_dest_mode[dev::cb_st] = UnpackToDestMode::UnpackToDestFp32;
            KernelDescriptor::Defines compute_defines = defines;
            if (approx) {
                compute_defines.emplace_back("FLASH_APPROX", "1");
            }
            desc.kernels.push_back(KernelDescriptor{
                .kernel_source = program::source("attention/ktt_flash/compute.cpp"),
                .core_ranges = all,
                .compile_time_args = {s.dt, s.dvt, r, s.skt, kb, a_bits, thr_bits, 0u, tail},
                .defines = compute_defines,
                .runtime_args = cp,
                .config = compute
            });
            for (uint32_t id = 0; id < 2; ++id) {
                desc.semaphores.push_back(SemaphoreDescriptor{
                    .id = id, .core_type = tt::CoreType::WORKER, .core_ranges = all, .initial_value = 0
                });
            }
            return program::Built{std::move(desc), std::move(slots)};
        });
    }

}  // namespace ktt::bridge::ktt_flash
