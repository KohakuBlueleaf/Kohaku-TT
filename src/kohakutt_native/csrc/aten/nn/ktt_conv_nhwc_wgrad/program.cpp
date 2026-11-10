// Host side of ktt_conv_nhwc_wgrad: plans (slots of go Cin-group columns inside one
// panel, a planes feeder column right of each slot when go > 1, one Cout group per
// (slot, row); each core walks Cin blocks of bc tiles over pixel blocks of kb tiles),
// the copy tables of the shifted dY tiles per pixel block, circular buffers, kernels
// and runtime args; and the X^T program.
#include "aten/nn/ktt_conv_nhwc_wgrad/program.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "core/device/handle.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"

namespace ktt::bridge::ktt_conv_nhwc_wgrad {
    namespace {

        constexpr uint32_t kTile = 2048;     // bf16 tile
        constexpr uint32_t kPart = 4096;     // fp32 tile
        constexpr uint32_t kPageWords = 16;  // table pages: multiples of 64 B
        constexpr uint32_t kSlots = 4;       // X^T and planes ring slots
        // Grid cost model (cycles): per tile product, per built tile, per fp32 tile
        // packed with L1 accumulation.
        constexpr double kProduct = 32, kBuild = 110, kPack = 60;
        // Column-top reader's X^T send per tile, and one built tile per RISC (cycles):
        // they size the top reader's share of its even builds (k0).
        constexpr uint32_t kSendPerTile = 72, kBuildRisc = 206;
        // Tiles the X^T program gives each core at least, and its tiles per DST
        // acquire.
        constexpr uint32_t kRotateMinTiles = 4, kRotateBlock = 4;
        // Runtime args the kernel sources read, per core and common: reader, writer,
        // feeder, compute.
        constexpr uint32_t kReaderArgs = 14, kWriterArgs = 12, kFeederArgs = 8,
                           kComputeArgs = 1;
        constexpr uint32_t kReaderCommon = 18, kWriterCommon = 24, kFeederCommon = 13,
                           kComputeCommon = 9;
        // The X^T program's: the blocked tile reader, rotate_writer, rotate_compute.
        constexpr uint32_t kRotateReaderArgs = 2, kRotateWriterArgs = 2,
                           kRotateComputeArgs = 1;
        constexpr uint32_t kRotateReaderCommon = 2, kRotateWriterCommon = 4,
                           kRotateComputeCommon = 1;
        static_assert(
            kReaderArgs <= tt::tt_metal::max_runtime_args &&
            kWriterArgs <= tt::tt_metal::max_runtime_args
        );

        uint32_t cdiv(uint32_t a, uint32_t b) {
            return (a + b - 1) / b;
        }

        // The device grid and its panels.
        struct Grid {
            uint32_t x = 0, y = 0;
            std::vector<std::pair<uint32_t, uint32_t>> panels;
        };

        // Slots (first column, feeder column or -1) of go columns over the panels.
        std::vector<std::pair<uint32_t, int64_t>> slots_of(
            const Grid& grid, uint32_t go
        ) {
            const bool feed = go > 1;
            std::vector<std::pair<uint32_t, int64_t>> slots;
            for (const auto& [a, b] : grid.panels) {
                const uint32_t width = b - a + 1;
                const uint32_t k = feed ? width / (go + 1) : width / go;
                for (uint32_t i = 0; i < k; ++i) {
                    slots.emplace_back(
                        a + i * go, feed ? static_cast<int64_t>(a + k * go + i) : -1
                    );
                }
            }
            return slots;
        }

        // Column j of a slot owns Cin tiles [j share, (j + 1) share), walked as nq
        // blocks of bc; group (slot, row) owns bo Cout tiles; every core streams all Mt
        // pixel tiles as npb blocks of kb. Tiles past Ct or Nt compute nothing that is
        // written.
        struct Plan {
            Geometry g;
            uint32_t ct = 0, nt = 0, mt = 0, taps = 0;
            uint32_t go = 1, bc = 1, kb = 1, bo = 1, share = 1, nq = 1, npb = 1,
                     rows = 1;
            // Largest tap offset in pixels; input tiles a planes slot holds; even / odd
            // shifted tiles per step.
            uint32_t max_off = 0, t_in_max = 1, n_even = 1, n_odd = 0;
            bool feed = false;
            // (first column, feeder column or -1) per slot.
            std::vector<std::pair<uint32_t, int64_t>> slots;

            // dY tiles [lo, lo + n) pixel block pb reads (halo included); the kernels
            // compute the same range.
            std::pair<uint32_t, uint32_t> block_range(uint32_t pb) const {
                const int64_t mp = static_cast<int64_t>(pb) * kb;
                if (mp >= mt) {
                    return {0, 1};
                }
                const int64_t lo = std::max<int64_t>(0, 32 * mp - max_off);
                const int64_t hi = std::min<int64_t>(
                    32 * static_cast<int64_t>(mt) - 1, 32 * (mp + kb) - 1 + max_off
                );
                return {
                    static_cast<uint32_t>(lo / 32),
                    static_cast<uint32_t>(hi / 32 + 1 - lo / 32)
                };
            }

            std::string text() const {
                return "go " + std::to_string(go) + " bc " + std::to_string(bc) +
                    " x " + std::to_string(nq) + " kb " + std::to_string(kb) + " x " +
                    std::to_string(npb) + " bo " + std::to_string(bo) + " slots " +
                    std::to_string(slots.size()) + " x " + std::to_string(rows);
            }
        };

        // The plan (go, bc, kb, bo) of g; nullopt when a column would hold no Cin tile,
        // no slot fits a panel, or the groups do not cover the Cout tiles.
        std::optional<Plan> make_plan(
            const Geometry& g,
            const Grid& grid,
            uint32_t go,
            uint32_t bc,
            uint32_t kb,
            uint32_t bo
        ) {
            Plan pl;
            pl.g = g;
            pl.ct = cdiv(g.c, 32);
            pl.nt = cdiv(g.o, 32);
            pl.mt = cdiv(g.n * g.h * g.w, 32);
            pl.taps = g.k * g.k;
            if (go == 0 || bc == 0 || bc > 4 || kb == 0 || bo == 0 || kb > pl.mt) {
                return std::nullopt;
            }
            pl.go = go;
            pl.bc = bc;
            pl.kb = kb;
            pl.bo = bo;
            pl.share = cdiv(cdiv(pl.ct, go), bc) * bc;
            if ((go - 1) * pl.share >= pl.ct) {
                return std::nullopt;
            }
            pl.nq = pl.share / bc;
            pl.npb = cdiv(pl.mt, kb);
            pl.feed = go > 1;
            const auto slots = slots_of(grid, go);
            if (slots.empty()) {
                return std::nullopt;
            }
            const uint32_t count = static_cast<uint32_t>(slots.size());
            const uint32_t groups = cdiv(pl.nt, bo);
            pl.rows = std::min(grid.y, cdiv(groups, count));
            const uint32_t used = std::min(count, cdiv(groups, pl.rows));
            if (static_cast<uint64_t>(used) * pl.rows < groups) {
                return std::nullopt;
            }
            pl.slots.assign(slots.begin(), slots.begin() + used);
            const uint32_t p = (g.k - 1) / 2;
            pl.max_off = p * g.w + p;
            pl.t_in_max = kb + cdiv(2 * pl.max_off, 32) + 1;
            pl.n_even = cdiv(kb * pl.taps, 2);
            pl.n_odd = kb * pl.taps / 2;
            return pl;
        }

        // Circular-buffer bytes per core of pl with table pages of page_bytes: 4 X^T
        // slots, 2 blocks each of even and odd shifted tiles, 4 planes slots, 2 output
        // blocks, the fp32 partials of one output block, 2 table pages per mover.
        uint64_t cb_bytes(const Plan& pl, uint32_t page_bytes) {
            const uint64_t tiles = static_cast<uint64_t>(kSlots) * pl.bc * pl.kb +
                2ull * pl.n_even + 2ull * std::max(1u, pl.n_odd) +
                static_cast<uint64_t>(kSlots) * pl.t_in_max + 2ull * pl.bc;
            return tiles * kTile + static_cast<uint64_t>(pl.bc) * pl.taps * kPart +
                4ull * page_bytes;
        }

        // A candidate plan: its blocking, modeled cost, circular-buffer bytes without
        // the tables.
        struct Candidate {
            uint32_t go, bc, kb, bo;
            double cost;
            uint64_t cb_lower;
        };

        using PlanKey = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>;

        // Every plan of g with the fewest Cout tiles per group that cover Nt, cheapest
        // first: modeled cycles per core, max(FPU, builds) plus one pack per output
        // tile and pixel block; ties in enumeration order (go ascending, bc and kb
        // descending).
        std::vector<Candidate> candidates(const Geometry& g, const Grid& grid) {
            const uint32_t ct = cdiv(g.c, 32), nt = cdiv(g.o, 32),
                           mt = cdiv(g.n * g.h * g.w, 32);
            const uint32_t taps = g.k * g.k;
            std::vector<Candidate> out;
            for (uint32_t go = 1; go <= std::min(ct, grid.x); ++go) {
                const uint32_t lanes = static_cast<uint32_t>(slots_of(grid, go).size());
                if (lanes == 0) {
                    continue;
                }
                const uint32_t bo = std::max(1u, cdiv(nt, lanes * grid.y));
                for (const uint32_t bc : {4u, 3u, 2u, 1u}) {
                    uint32_t seen = 0;
                    for (const uint32_t size : {16u, 12u, 8u, 6u, 4u, 3u, 2u, 1u}) {
                        const uint32_t kb = std::min(size, mt);
                        if (kb == seen) {
                            continue;
                        }
                        seen = kb;
                        const auto pl = make_plan(g, grid, go, bc, kb, bo);
                        if (!pl) {
                            continue;
                        }
                        // Per output block: products over the Mt pixel tiles, builds
                        // over every tile of its pixel blocks (the last block's tiles
                        // past Mt included).
                        const double blocks = static_cast<double>(bo) * pl->nq;
                        const double built = static_cast<double>(pl->npb) * kb;
                        const double cost = blocks * taps *
                                std::max(bc * kProduct * mt, kBuild * built) +
                            blocks * pl->npb * bc * taps * kPack;
                        out.push_back(
                            {go, bc, kb, bo, cost, cb_bytes(*pl, kPageWords * 4)}
                        );
                    }
                }
            }
            std::stable_sort(
                out.begin(), out.end(), [](const Candidate& a, const Candidate& b) {
                    return a.cost < b.cost;
                }
            );
            return out;
        }

        // A plan with its copy table: page 2 (kind npb + pb) holds the reader's words
        // of pixel block pb, the next page the writer's; kind 0 for cores whose reader
        // builds every even tile, kind 1 for column tops that send X^T (the reader
        // builds even tiles [0, k0_top)).
        struct PlanTables {
            Plan plan;
            uint32_t k0_top = 0;
            uint32_t page_words = 0;
            uint64_t cb_bytes = 0;
            std::vector<uint32_t> table;
            std::optional<ttnn::Tensor> device;
        };

        std::unique_ptr<PlanTables> make_layout(const Plan& pl) {
            auto out = std::make_unique<PlanTables>();
            out->plan = pl;
            // The column top's reader also sends X^T: its share of the even builds
            // shrinks by the send's time in built tiles.
            const uint32_t send = cdiv(kSendPerTile * pl.bc * pl.kb, kBuildRisc);
            const int64_t share = (static_cast<int64_t>(pl.n_even) + pl.n_odd - send) /
                2;
            out->k0_top = pl.feed && pl.rows > 1
                ? static_cast<uint32_t>(std::clamp<int64_t>(share, 0, pl.n_even))
                : pl.n_even;
            std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> pages;
            for (const uint32_t k0 : {pl.n_even, out->k0_top}) {
                for (uint32_t pb = 0; pb < pl.npb; ++pb) {
                    const auto [tile_lo, t_in] = pl.block_range(pb);
                    if (t_in > pl.t_in_max) {
                        throw std::logic_error(
                            "ktt_conv_nhwc_wgrad: a pixel block reads more tiles than "
                            "a planes slot"
                        );
                    }
                    // Per shifted tile (m - first) taps + t: its words, even and odd
                    // tiles apart.
                    std::vector<std::vector<uint32_t>> tiles[2];
                    uint32_t i = 0;
                    for (uint32_t m = pb * pl.kb; m < (pb + 1) * pl.kb; ++m) {
                        for (uint32_t t = 0; t < pl.taps; ++t) {
                            tiles[i & 1].emplace_back();
                            ktt_conv_nhwc::copy_runs(
                                pl.g,
                                m,
                                t,
                                m < pl.mt,
                                tile_lo,
                                t_in,
                                pl.t_in_max,
                                tiles[i & 1].back()
                            );
                            ++i;
                        }
                    }
                    std::vector<uint32_t> reader, writer;
                    for (uint32_t k = 0; k < k0; ++k) {
                        reader.insert(
                            reader.end(), tiles[0][k].begin(), tiles[0][k].end()
                        );
                    }
                    for (const auto& w : tiles[1]) {
                        writer.insert(writer.end(), w.begin(), w.end());
                    }
                    for (uint32_t k = k0; k < pl.n_even; ++k) {
                        writer.insert(
                            writer.end(), tiles[0][k].begin(), tiles[0][k].end()
                        );
                    }
                    out->page_words = std::max<uint32_t>(
                        out->page_words,
                        static_cast<uint32_t>(std::max(reader.size(), writer.size()))
                    );
                    pages.emplace_back(std::move(reader), std::move(writer));
                }
            }
            out->page_words = cdiv(std::max(out->page_words, kPageWords), kPageWords) *
                kPageWords;
            out->table.assign(2 * pages.size() * out->page_words, 0);
            for (size_t i = 0; i < pages.size(); ++i) {
                std::copy(
                    pages[i].first.begin(),
                    pages[i].first.end(),
                    out->table.begin() + 2 * i * out->page_words
                );
                std::copy(
                    pages[i].second.begin(),
                    pages[i].second.end(),
                    out->table.begin() + (2 * i + 1) * out->page_words
                );
            }
            out->cb_bytes = cb_bytes(pl, out->page_words * 4);
            return out;
        }

        // The copy table of `l` on the device: uint32 ROW_MAJOR [4 npb, page_words] in
        // DRAM.
        ttnn::Tensor upload_table(const PlanTables& l) {
            using namespace tt::tt_metal;
            const uint32_t pages = static_cast<uint32_t>(l.table.size() / l.page_words);
            const TensorSpec spec(
                ttnn::Shape({pages, l.page_words}),
                TensorLayout(
                    DataType::UINT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}
                )
            );
            return ttnn::Tensor::from_span(
                ttsl::Span<const uint32_t>(l.table.data(), l.table.size()),
                spec,
                detail::device()
            );
        }

        // A geometry's plans, cheapest first, and the layouts built for them.
        struct Entry {
            std::vector<Candidate> order;
            std::map<PlanKey, std::unique_ptr<PlanTables>> layouts;
        };

        // Guards cache(), forced() and last().
        std::mutex& cache_mutex() {
            static std::mutex m;
            return m;
        }

        // Never destroyed: device tables must not be freed after the device closes at
        // exit.
        std::map<std::vector<uint32_t>, Entry>& cache() {
            static auto* map = new std::map<std::vector<uint32_t>, Entry>();
            return *map;
        }

        // debug_plan()'s plan (go 0: none) and in-kernel passes.
        struct Forced {
            uint32_t go = 0, bc = 0, kb = 0, bo = 0, iters = 1;
        };

        Forced& forced() {
            static Forced f;
            return f;
        }

        std::string& last() {
            static auto* text = new std::string();
            return *text;
        }

        Grid grid_of() {
            const auto size = detail::device()->compute_with_storage_grid_size();
            return Grid{
                static_cast<uint32_t>(size.x),
                static_cast<uint32_t>(size.y),
                program::panels()
            };
        }

        program::Built describe(
            const PlanTables& l,
            const std::vector<ttnn::Tensor>& io,
            const ttnn::ComputeKernelConfig& config,
            uint32_t iters
        ) {
            using namespace tt::tt_metal;
            const Plan& pl = l.plan;
            const auto& xt = io[0];
            const auto& dy = io[1];
            const auto& tab = io[2];
            const auto& out = io[3];
            auto phys = [](uint32_t cx, uint32_t cy) {
                const auto c = detail::device()->worker_core_from_logical_core(
                    CoreCoord(cx, cy)
                );
                return std::pair<uint32_t, uint32_t>(c.x, c.y);
            };
            std::vector<CoreRange> comp_ranges, feed_ranges;
            for (const auto& [x0, fx] : pl.slots) {
                comp_ranges.emplace_back(
                    CoreCoord(x0, 0), CoreCoord(x0 + pl.go - 1, pl.rows - 1)
                );
                if (fx >= 0) {
                    feed_ranges.emplace_back(
                        CoreCoord(fx, 0), CoreCoord(fx, pl.rows - 1)
                    );
                }
            }
            std::vector<CoreRange> all_ranges = comp_ranges;
            all_ranges.insert(all_ranges.end(), feed_ranges.begin(), feed_ranges.end());
            const CoreRangeSet comp = CoreRangeSet(comp_ranges).merge_ranges();
            const CoreRangeSet all = CoreRangeSet(all_ranges).merge_ranges();

            const uint32_t m_lo = 0, m_cnt = pl.mt, row0 = 0;
            KernelDescriptor::RuntimeArgs rd, wr, fd, cp;
            for (size_t s = 0; s < pl.slots.size(); ++s) {
                const auto [x0, fx] = pl.slots[s];
                for (uint32_t y = 0; y < pl.rows; ++y) {
                    // Group (slot, row): Cout tiles [n0, n0 + bo).
                    const uint32_t n0 = (static_cast<uint32_t>(s) * pl.rows + y) *
                        pl.bo;
                    if (fx >= 0) {
                        // Planes from the slot's feeder, right of it in its panel (NoC1
                        // flows from high x); it multicasts to the go cores of the
                        // group, rectangle given end-first.
                        const auto b = phys(x0 + pl.go - 1, y), e = phys(x0, y);
                        fd.push_back(
                            {CoreCoord(fx, y),
                             {n0,
                              m_lo,
                              m_cnt,
                              b.first,
                              b.second,
                              e.first,
                              e.second,
                              pl.go}}
                        );
                    }
                    for (uint32_t j = 0; j < pl.go; ++j) {
                        const uint32_t x = x0 + j;
                        // X^T: the column top (low y, where NoC0 flows from) multicasts
                        // to rows 1 .. rows - 1.
                        const bool top = y == 0;
                        std::array<uint32_t, 5> mc{};
                        if (top && pl.rows > 1) {
                            const auto b = phys(x, 1), e = phys(x, pl.rows - 1);
                            mc = {b.first, b.second, e.first, e.second, pl.rows - 1};
                        }
                        const auto snd = phys(x, 0);
                        const auto ld = fx >= 0 ? phys(static_cast<uint32_t>(fx), y)
                                                : phys(x, y);
                        const uint32_t kind = top && l.k0_top < pl.n_even ? 1 : 0;
                        const uint32_t k0 = kind ? l.k0_top : pl.n_even;
                        const uint32_t tab_page0 = kind * pl.npb * 2;
                        const CoreCoord at(x, y);
                        rd.push_back(
                            {at,
                             {j * pl.share,
                              m_lo,
                              m_cnt,
                              tab_page0,
                              pl.n_even,
                              top ? 1u : 0u,
                              mc[0],
                              mc[1],
                              mc[2],
                              mc[3],
                              mc[4],
                              snd.first,
                              snd.second,
                              k0}}
                        );
                        wr.push_back(
                            {at,
                             {j * pl.share,
                              n0,
                              m_lo,
                              m_cnt,
                              tab_page0,
                              pl.n_odd,
                              pl.n_even,
                              k0,
                              fx >= 0 ? 0u : 1u,
                              ld.first,
                              ld.second,
                              row0}}
                        );
                        cp.push_back({at, {m_cnt}});
                    }
                }
            }
            program::mark(program::Phase::Args);

            ProgramDescriptor desc;
            const auto bf16 = tt::DataFormat::Float16_b;
            const uint32_t x_block = pl.bc * pl.kb;
            const uint32_t page_bytes = l.page_words * 4;
            const uint32_t slot_bytes = pl.t_in_max * kTile;
            const uint32_t part_pages = pl.bc * pl.taps;

            // X^T and planes rings are 4 slots deep (step s: compute on s - 1, builds
            // s, multicast s + 1, reads s + 2); shifted-tile rings 2 blocks; table
            // rings 2 steps.
            struct Cb {
                uint32_t index, pages, page;
                tt::DataFormat format;
            };

            const std::vector<Cb> cbs = {
                {1, kSlots * x_block, kTile, bf16},
                {2, 2 * pl.n_even, kTile, bf16},
                {3, 2 * std::max(1u, pl.n_odd), kTile, bf16},
                {5, kSlots * pl.t_in_max, kTile, bf16},
                {8, 2, page_bytes, tt::DataFormat::Float32},
                {9, 2, page_bytes, tt::DataFormat::Float32},
                {16, 2 * pl.bc, kTile, bf16},
                {24, part_pages, kPart, tt::DataFormat::Float32},
            };
            auto pages_of = [&](uint32_t index) {
                for (const auto& cb : cbs) {
                    if (cb.index == index) {
                        return cb.pages;
                    }
                }
                throw std::logic_error("ktt_conv_nhwc_wgrad: unknown circular buffer");
            };
            // Pushes per transaction; bytes written from the base of the rings the
            // movers address by slot (X^T, even blocks, planes, tables).
            std::vector<program::Ring> rings = {
                {.cb = 1,
                 .pushes = {x_block},
                 .written = static_cast<uint64_t>(kSlots) * x_block * kTile},
                {.cb = 2, .pushes = {pl.n_even}, .written = 2ull * pl.n_even * kTile},
                {.cb = 5, .written = static_cast<uint64_t>(kSlots) * slot_bytes},
                {.cb = 8, .written = 2ull * page_bytes},
                {.cb = 9, .written = 2ull * page_bytes},
                {.cb = 16, .pushes = {pl.bc}},
                {.cb = 24, .pushes = {pl.bc}},
            };
            if (pl.n_odd > 0) {
                rings.push_back({.cb = 3, .pushes = {pl.n_odd}});
            }
            // The writer addresses even block s at ring position s & 1.
            if (pages_of(2) != 2 * pl.n_even) {
                throw std::logic_error(
                    "ktt_conv_nhwc_wgrad: the even-tile ring must hold exactly two "
                    "blocks"
                );
            }
            for (uint32_t pb = 0; pb < pl.npb; ++pb) {
                if (pl.block_range(pb).second * kTile > slot_bytes) {
                    throw std::logic_error(
                        "ktt_conv_nhwc_wgrad: a pixel block's input exceeds its planes "
                        "slot"
                    );
                }
            }
            for (const auto& cb : cbs) {
                desc.cbs.push_back(
                    CBDescriptor{
                        .total_size = cb.pages * cb.page,
                        .core_ranges = all,
                        .format_descriptors = {CBFormatDescriptor{
                            .buffer_index = static_cast<uint8_t>(cb.index),
                            .data_format = cb.format,
                            .page_size = cb.page
                        }}
                    }
                );
            }
            // 0 X count, 1 X flag, 2 planes ready, 3 planes done, 4 planes count, 5
            // planes flag, 6 a0 go, 7 a0 done.
            for (uint32_t id = 0; id < 8; ++id) {
                desc.semaphores.push_back(
                    SemaphoreDescriptor{
                        .id = id,
                        .core_type = tt::CoreType::WORKER,
                        .core_ranges = all,
                        .initial_value = 0
                    }
                );
            }

            auto accessor = [](const ttnn::Tensor& t) {
                return TensorAccessorArgs(*t.buffer()).get_compile_time_args();
            };
            auto concat = [](std::vector<uint32_t> a, const std::vector<uint32_t>& b) {
                a.insert(a.end(), b.begin(), b.end());
                return a;
            };
            const uint32_t min_off = static_cast<uint32_t>(
                -static_cast<int32_t>(pl.max_off)
            );
            // One NoC per mover on every core: reader NoC0 (X^T multicast down a
            // column), writer and feeder NoC1 (planes multicast toward low x).
            const NOC reader_noc = NOC::NOC_0;
            const NOC writer_noc = reader_noc == NOC::NOC_0 ? NOC::NOC_1 : NOC::NOC_0;
            if (writer_noc == reader_noc) {
                throw std::logic_error(
                    "ktt_conv_nhwc_wgrad: reader and writer share a NoC"
                );
            }
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source(
                        "nn/ktt_conv_nhwc_wgrad/reader.cpp"
                    ),
                    .core_ranges = comp,
                    .compile_time_args = concat(accessor(xt), accessor(tab)),
                    .runtime_args = rd,
                    .common_runtime_args =
                        {xt.buffer()->address(),
                         tab.buffer()->address(),
                         pl.ct,
                         pl.mt,
                         pl.bc,
                         pl.kb,
                         pl.nq,
                         pl.npb,
                         pl.bo,
                         iters,
                         0,
                         1,
                         2,
                         3,
                         6,
                         7,
                         page_bytes,
                         slot_bytes},
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
                        "nn/ktt_conv_nhwc_wgrad/writer.cpp"
                    ),
                    .core_ranges = comp,
                    .compile_time_args = concat(
                        concat(accessor(dy), accessor(tab)), accessor(out)
                    ),
                    .runtime_args = wr,
                    .common_runtime_args =
                        {dy.buffer()->address(),
                         tab.buffer()->address(),
                         out.buffer()->address(),
                         pl.nt,
                         pl.taps,
                         pl.bc,
                         pl.kb,
                         pl.nq,
                         pl.npb,
                         pl.bo,
                         iters,
                         4,
                         5,
                         2,
                         3,
                         6,
                         7,
                         page_bytes,
                         slot_bytes,
                         pl.mt,
                         pl.ct,
                         min_off,
                         pl.max_off,
                         kTile},
                    .config = DataMovementConfigDescriptor{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = writer_noc,
                        .noc_mode = NOC_MODE::DM_DEDICATED_NOC
                    }
                }
            );
            std::vector<program::Slot> slots = {
                {.kernel = 0,
                 .entry = program::Slot::kCommon,
                 .position = 0,
                 .tensor = 0},
                {.kernel = 0,
                 .entry = program::Slot::kCommon,
                 .position = 1,
                 .tensor = 2},
                {.kernel = 1,
                 .entry = program::Slot::kCommon,
                 .position = 0,
                 .tensor = 1},
                {.kernel = 1,
                 .entry = program::Slot::kCommon,
                 .position = 1,
                 .tensor = 2},
                {.kernel = 1,
                 .entry = program::Slot::kCommon,
                 .position = 2,
                 .tensor = 3},
            };
            std::vector<program::ArgCount> args = {
                {.kernel = 0, .per_core = kReaderArgs, .common = kReaderCommon},
                {.kernel = 1, .per_core = kWriterArgs, .common = kWriterCommon},
            };
            if (!feed_ranges.empty()) {
                desc.kernels.push_back(
                    KernelDescriptor{
                        .kernel_source = program::source(
                            "nn/ktt_conv_nhwc_wgrad/feeder.cpp"
                        ),
                        .core_ranges = CoreRangeSet(feed_ranges).merge_ranges(),
                        .compile_time_args = accessor(dy),
                        .runtime_args = fd,
                        .common_runtime_args =
                            {dy.buffer()->address(),
                             pl.nt,
                             pl.kb,
                             pl.nq,
                             pl.npb,
                             pl.bo,
                             iters,
                             4,
                             5,
                             slot_bytes,
                             pl.mt,
                             min_off,
                             pl.max_off},
                        .config = DataMovementConfigDescriptor{
                            .processor = DataMovementProcessor::RISCV_0,
                            .noc = writer_noc,
                            .noc_mode = NOC_MODE::DM_DEDICATED_NOC
                        }
                    }
                );
                slots.push_back(
                    {.kernel = 2,
                     .entry = program::Slot::kCommon,
                     .position = 0,
                     .tensor = 1}
                );
                args.push_back(
                    {.kernel = 2, .per_core = kFeederArgs, .common = kFeederCommon}
                );
            }
            ComputeConfigDescriptor compute{
                .math_fidelity = config.math_fidelity,
                .fp32_dest_acc_en = true,
                .dst_full_sync_en = false,
                .unpack_to_dest_mode = ComputeConfigDescriptor::UnpackToDestModes(
                    NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default
                ),
                .math_approx_mode = config.math_approx_mode,
            };
            compute.unpack_to_dest_mode[24] = UnpackToDestMode::UnpackToDestFp32;
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source(
                        "nn/ktt_conv_nhwc_wgrad/compute.cpp"
                    ),
                    .core_ranges = comp,
                    .runtime_args = cp,
                    .common_runtime_args =
                        {pl.taps,
                         pl.bc,
                         pl.kb,
                         pl.nq,
                         pl.npb,
                         pl.bo,
                         iters,
                         pl.n_even,
                         pl.n_odd},
                    .config = compute
                }
            );
            args.push_back({
                .kernel = static_cast<uint32_t>(desc.kernels.size() - 1),
                .per_core = kComputeArgs,
                .common = kComputeCommon,
            });
            return program::Built{
                std::move(desc), std::move(slots), std::move(args), std::move(rings)
            };
        }

    }  // namespace

    ttnn::Tensor rotate(
        const ttnn::Tensor& x, const Geometry& g, const ttnn::Tensor* into
    ) {
        using namespace tt::tt_metal;
        const uint32_t ct = cdiv(g.c, 32), mt = cdiv(g.n * g.h * g.w, 32);
        auto out = into ? *into
                        : ttnn::empty(
                              ttnn::Shape({ct * 32, mt * 32}),
                              DataType::BFLOAT16,
                              Layout::TILE,
                              detail::device(),
                              MemoryConfig{}
                          );
        program::mark(program::Phase::Alloc);
        program::Key key("ktt_conv_nhwc_rotate");
        key.add(x).add(out);
        return program::launch(key, {x, out}, [&] {
            const auto s = program::split_tiles(ct * mt, kRotateMinTiles);
            ProgramDescriptor desc;
            for (const uint8_t index : {uint8_t{0}, uint8_t{16}}) {
                desc.cbs.push_back(
                    CBDescriptor{
                        .total_size = 4 * kRotateBlock * kTile,
                        .core_ranges = s.range,
                        .format_descriptors = {CBFormatDescriptor{
                            .buffer_index = index,
                            .data_format = tt::DataFormat::Float16_b,
                            .page_size = kTile
                        }}
                    }
                );
            }
            KernelDescriptor::RuntimeArgs ranges, counts;
            uint32_t start = 0;
            for (size_t i = 0; i < s.cores.size(); ++i) {
                ranges.push_back({s.cores[i], {start, s.count(i)}});
                counts.push_back({s.cores[i], {s.count(i)}});
                start += s.count(i);
            }
            program::mark(program::Phase::Args);
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("eltwise/ktt_unary/reader.cpp"),
                    .core_ranges = s.range,
                    .compile_time_args = TensorAccessorArgs(*x.buffer())
                                             .get_compile_time_args(),
                    .runtime_args = ranges,
                    .common_runtime_args = {x.buffer()->address(), kRotateBlock},
                    .config = ReaderConfigDescriptor{}
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source(
                        "nn/ktt_conv_nhwc_wgrad/rotate_writer.cpp"
                    ),
                    .core_ranges = s.range,
                    .compile_time_args = TensorAccessorArgs(*out.buffer())
                                             .get_compile_time_args(),
                    .runtime_args = ranges,
                    .common_runtime_args =
                        {out.buffer()->address(), kRotateBlock, ct, mt},
                    .config = WriterConfigDescriptor{}
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source(
                        "nn/ktt_conv_nhwc_wgrad/rotate_compute.cpp"
                    ),
                    .core_ranges = s.range,
                    .runtime_args = counts,
                    .common_runtime_args = {kRotateBlock},
                    .config = ComputeConfigDescriptor{
                        // A copy: no FPU multiply.
                        .math_fidelity = MathFidelity::LoFi,
                        .fp32_dest_acc_en = false,
                        .math_approx_mode = false
                    }
                }
            );
            const std::vector<program::Slot> slots = {
                {.kernel = 0, .entry = program::Slot::kCommon},
                {.kernel = 1, .entry = program::Slot::kCommon, .tensor = 1}
            };
            // A core's last transaction on either ring may be shorter than a block.
            return program::Built{
                std::move(desc),
                slots,
                {
                    {.kernel = 0,
                     .per_core = kRotateReaderArgs,
                     .common = kRotateReaderCommon},
                    {.kernel = 1,
                     .per_core = kRotateWriterArgs,
                     .common = kRotateWriterCommon},
                    {.kernel = 2,
                     .per_core = kRotateComputeArgs,
                     .common = kRotateComputeCommon},
                },
                {{.cb = 0, .pushes = {kRotateBlock}},
                 {.cb = 16, .pushes = {kRotateBlock}}}
            };
        });
    }

    ttnn::Tensor run(
        const ttnn::Tensor& xt,
        const ttnn::Tensor& dy,
        const Geometry& g,
        const ttnn::ComputeKernelConfig& config,
        Tables tables
    ) {
        using namespace tt::tt_metal;
        const uint32_t ct = cdiv(g.c, 32), nt = cdiv(g.o, 32), taps = g.k * g.k;
        auto out = ttnn::empty(
            ttnn::Shape({ct * taps * 32, nt * 32}),
            DataType::BFLOAT16,
            Layout::TILE,
            detail::device(),
            MemoryConfig{}
        );
        program::mark(program::Phase::Alloc);
        // The cheapest plan whose circular buffers fit the budget (read after the
        // output is allocated) and whose copy table `tables` allows; debug_plan()'s
        // plan alone when set.
        const uint64_t budget = program::cb_budget();
        PlanTables* chosen = nullptr;
        std::optional<ttnn::Tensor> once;
        uint32_t iters = 1;
        {
            const std::lock_guard lock(cache_mutex());
            const Grid grid = grid_of();
            auto& entry = cache()[{g.n, g.c, g.h, g.w, g.o, g.k}];
            if (entry.order.empty()) {
                entry.order = candidates(g, grid);
            }
            const Forced& f = forced();
            iters = f.iters;
            const std::vector<Candidate> only = {{f.go, f.bc, f.kb, f.bo, 0, 0}};
            for (const auto& cand : f.go > 0 ? only : entry.order) {
                if (cand.cb_lower > budget) {
                    continue;
                }
                const PlanKey key{cand.go, cand.bc, cand.kb, cand.bo};
                const auto it = entry.layouts.find(key);
                PlanTables* l = it == entry.layouts.end() ? nullptr : it->second.get();
                if (!l) {
                    if (tables == Tables::CachedOnly) {
                        continue;
                    }
                    const auto pl = make_plan(
                        g, grid, cand.go, cand.bc, cand.kb, cand.bo
                    );
                    if (!pl) {
                        continue;
                    }
                    l = (entry.layouts[key] = make_layout(*pl)).get();
                }
                if (l->cb_bytes > budget ||
                    (tables == Tables::CachedOnly && !l->device)) {
                    continue;
                }
                if (!l->device) {
                    if (tables == Tables::Cache) {
                        l->device = upload_table(*l);
                    } else {
                        once = upload_table(*l);
                    }
                }
                chosen = l;
                break;
            }
            if (chosen) {
                last() = chosen->plan.text();
            }
        }
        if (!chosen) {
            throw std::runtime_error(
                "ktt_conv_nhwc_wgrad: no plan fits the circular-buffer budget"
            );
        }
        const ttnn::Tensor table = once ? *once : *chosen->device;
        const Plan& pl = chosen->plan;
        program::Key key("ktt_conv_nhwc_wgrad");
        key.add(xt).add(dy).add(table).add(out);
        key.add(static_cast<int64_t>(config.math_fidelity))
            .add(static_cast<int64_t>(config.math_approx_mode));
        for (
            const int64_t v :
            {g.n, g.c, g.h, g.w, g.o, g.k, pl.go, pl.bc, pl.kb, pl.bo, iters}
        ) {
            key.add(v);
        }
        const std::vector<ttnn::Tensor> io = {xt, dy, table, out};
        const PlanTables& layout = *chosen;
        return program::launch(key, io, [&] {
            return describe(layout, io, config, iters);
        });
    }

    void debug_plan(
        uint32_t go, uint32_t bc, uint32_t kb, uint32_t bo, uint32_t iters
    ) {
        const std::lock_guard lock(cache_mutex());
        forced() = Forced{go, bc, kb, bo, std::max(1u, iters)};
    }

    std::string last_plan() {
        const std::lock_guard lock(cache_mutex());
        return last();
    }

    void clear() {
        const std::lock_guard lock(cache_mutex());
        cache().clear();
    }

}  // namespace ktt::bridge::ktt_conv_nhwc_wgrad
