// Host side of ktt_conv_nhwc: plans (pixel blocks of go Cout-group columns inside one
// panel, a planes feeder column right of each block when go > 1, rows of pixel groups;
// each core walks its share in blocks of bm_b output tiles x bn_b Cout tiles), their
// per-(core, pixel block) A-tile copy tables, the W' layout, circular buffers, kernels
// and runtime args.
#include "aten/nn/ktt_conv_nhwc/program.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <mutex>
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

namespace ktt::bridge::ktt_conv_nhwc {
    namespace {

        constexpr uint32_t kTile = 2048;         // bf16 tile
        constexpr uint32_t kPart = 4096;         // fp32 tile
        constexpr uint32_t kFace = 512;          // bf16 face (16 x 16)
        constexpr uint32_t kZero = 0xFFFFFFFFu;  // copy source: MEM_ZEROS
        constexpr uint32_t kPageWords = 16;      // table pages: multiples of 64 B
        constexpr uint32_t kSlots = 4;           // W' and planes ring slots
        // Output tiles per pixel block the planner tries.
        constexpr uint32_t kMaxBlockM = 64;
        // Grid cost model (cycles at kClock): per tile product, per built A tile, DRAM
        // bytes.
        constexpr double kClock = 1.35e9;
        constexpr double kProduct = 32, kBuild = 110, kDram = 400e9;
        // Column-top reader's W' send per W' tile, and one built A tile per RISC
        // (cycles): they size the top reader's share of its A0 builds (k0).
        constexpr uint32_t kSendPerTile = 72, kBuildRisc = 206;
        // Runtime args the kernel sources read, per core and common: reader, writer,
        // feeder, compute.
        constexpr uint32_t kReaderArgs = 13, kWriterArgs = 11, kFeederArgs = 7,
                           kComputeArgs = 1;
        constexpr uint32_t kReaderCommon = 17, kWriterCommon = 23, kFeederCommon = 11,
                           kComputeCommon = 10;
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

        // Pixel blocks of go adjacent columns inside one panel; with go > 1 each block
        // has a planes feeder column right of it in the same panel. Column j of a block
        // is Cout group j (bn tiles, walked as nnb sub-blocks of bn_b); pixel group
        // (block b, row y) owns bm output tiles, walked as npb pixel blocks of bm_b.
        struct Plan {
            Geometry g;
            uint32_t p = 0, ct = 0, nt = 0, npix = 0, mt = 0, taps = 0;
            uint32_t go = 1, bn = 1, bn_b = 1, nnb = 1, bm = 1, bm_b = 1, npb = 1,
                     sub = 1, rows = 1;
            // Largest tap offset in pixels (the smallest is its negative); input tiles
            // a planes slot holds; even / odd A tiles per step.
            uint32_t max_off = 0, t_in_max = 1, n_even = 1, n_odd = 0;
            bool feed = false;
            // (first column, feeder column or -1) per block.
            std::vector<std::pair<uint32_t, int64_t>> blocks;
            // Compute columns, and per grid column its (block or -1, j).
            std::vector<uint32_t> xs;
            std::vector<std::pair<int64_t, uint32_t>> col;

            // First output tile of core (x, y); its share is [m0, m0 + bm), tiles >= Mt
            // are zero (not written).
            uint32_t first_tile(uint32_t x, uint32_t y) const {
                return (static_cast<uint32_t>(col[x].first) * rows + y) * bm;
            }

            // Input tiles [lo, lo + n) that pixel block pb of the share starting at m0
            // reads (halo included); the kernels compute the same range.
            std::pair<uint32_t, uint32_t> block_range(uint32_t m0, uint32_t pb) const {
                const int64_t mp = static_cast<int64_t>(m0) +
                    static_cast<int64_t>(pb) * bm_b;
                if (mp >= mt) {
                    return {0, 1};
                }
                const int64_t lo = std::max<int64_t>(0, 32 * mp - max_off);
                const int64_t hi = std::min<int64_t>(
                    32 * static_cast<int64_t>(mt) - 1, 32 * (mp + bm_b) - 1 + max_off
                );
                return {
                    static_cast<uint32_t>(lo / 32),
                    static_cast<uint32_t>(hi / 32 + 1 - lo / 32)
                };
            }

            // Input pixel offset of tap t (row-major over the K x K window).
            int64_t offset(uint32_t t) const {
                const int64_t dy = t / g.k, dx = t % g.k;
                return (dy - p) * static_cast<int64_t>(g.w) + (dx - p);
            }

            std::string text() const {
                return "go " + std::to_string(go) + " bn_b " + std::to_string(bn_b) +
                    " x " + std::to_string(nnb) + " bm_b " + std::to_string(bm_b) +
                    " x " + std::to_string(npb) + " sub " + std::to_string(sub) +
                    " blocks " + std::to_string(blocks.size()) + " x " +
                    std::to_string(rows);
            }
        };

        // Block slots (first column, feeder column or -1) of go columns over the
        // panels.
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

        // The plan (go, bm_b, npb, bn_b) of g; nullopt when Nt or bn do not divide, no
        // block fits a panel, or the grid does not cover the output.
        std::optional<Plan> make_plan(
            const Geometry& g,
            const Grid& grid,
            uint32_t go,
            uint32_t bm_b,
            uint32_t npb,
            uint32_t bn_b
        ) {
            Plan pl;
            pl.g = g;
            pl.p = (g.k - 1) / 2;
            pl.ct = cdiv(g.c, 32);
            pl.nt = cdiv(g.o, 32);
            pl.npix = g.n * g.h * g.w;
            pl.mt = cdiv(pl.npix, 32);
            pl.taps = g.k * g.k;
            if (go == 0 || bm_b == 0 || npb == 0 || bn_b == 0 || pl.nt % go != 0 ||
                (pl.nt / go) % bn_b != 0) {
                return std::nullopt;
            }
            pl.go = go;
            pl.bn = pl.nt / go;
            pl.bn_b = bn_b;
            pl.nnb = pl.bn / bn_b;
            pl.bm_b = bm_b;
            pl.npb = npb;
            pl.bm = bm_b * npb;
            for (const uint32_t s : {4u, 3u, 2u, 1u}) {
                if (bn_b % s == 0) {
                    pl.sub = s;
                    break;
                }
            }
            pl.feed = go > 1;
            const auto slots = slots_of(grid, go);
            if (slots.empty()) {
                return std::nullopt;
            }
            const uint32_t count = static_cast<uint32_t>(slots.size());
            const uint32_t groups = cdiv(pl.mt, pl.bm);
            pl.rows = std::min(grid.y, cdiv(groups, count));
            const uint32_t blocks = std::min(count, cdiv(groups, pl.rows));
            if (static_cast<uint64_t>(blocks) * pl.rows * pl.bm < pl.mt) {
                return std::nullopt;
            }
            pl.blocks.assign(slots.begin(), slots.begin() + blocks);
            pl.col.assign(grid.x, {-1, 0});
            for (size_t b = 0; b < pl.blocks.size(); ++b) {
                for (uint32_t j = 0; j < go; ++j) {
                    pl.xs.push_back(pl.blocks[b].first + j);
                    pl.col[pl.blocks[b].first + j] = {static_cast<int64_t>(b), j};
                }
            }
            pl.max_off = pl.p * g.w + pl.p;
            pl.t_in_max = bm_b + cdiv(2 * pl.max_off, 32) + 1;
            pl.n_even = cdiv(bm_b * pl.taps, 2);
            pl.n_odd = bm_b * pl.taps / 2;
            return pl;
        }

        // Circular-buffer bytes per core of pl with table pages of page_bytes: 4 W'
        // slots, 2 blocks each of even and odd A tiles, 4 planes slots, 2 output
        // subblocks, the fp32 partials of one pixel block, 2 table pages per mover.
        uint64_t cb_bytes(const Plan& pl, uint32_t page_bytes) {
            const uint64_t tiles = static_cast<uint64_t>(kSlots) * pl.taps * pl.bn_b +
                2ull * pl.n_even + 2ull * std::max(1u, pl.n_odd) +
                static_cast<uint64_t>(kSlots) * pl.t_in_max + 2ull * pl.sub;
            return tiles * kTile + static_cast<uint64_t>(pl.bm_b) * pl.bn * kPart +
                4ull * page_bytes;
        }

        // A candidate plan: its blocking, blocks per core (npb x nnb), the modeled cost
        // of its grid, circular-buffer bytes without the tables.
        struct Candidate {
            uint32_t go, bm_b, npb, bn_b, blocks;
            double cost;
            uint64_t cb_lower;
        };

        using PlanKey = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>;

        // Every plan of g: per go and bm_b, the fewest pixel blocks npb that cover the
        // output, with every split of the Cout group. Order: fewest blocks per core
        // first, then the fewest pixel blocks (each one streams W' again), then the
        // grid's modeled cost (max of FPU, A builds and DRAM bytes per core); ties in
        // enumeration order (go, then bm_b ascending).
        std::vector<Candidate> candidates(const Geometry& g, const Grid& grid) {
            const uint32_t ct = cdiv(g.c, 32), nt = cdiv(g.o, 32),
                           mt = cdiv(g.n * g.h * g.w, 32);
            const uint32_t taps = g.k * g.k, p = (g.k - 1) / 2;
            const double halo = 2.0 * (static_cast<double>(p) * g.w + p);
            std::vector<Candidate> out;
            for (const uint32_t go : {1u, 2u, 4u, 8u, 16u}) {
                if (nt % go != 0 || go > grid.x) {
                    continue;
                }
                const uint32_t lanes = static_cast<uint32_t>(slots_of(grid, go).size());
                if (lanes == 0) {
                    continue;
                }
                const uint32_t bn = nt / go;
                for (uint32_t bm_b = 1; bm_b <= std::min(mt, kMaxBlockM); ++bm_b) {
                    const uint32_t npb = std::max(
                        1u, cdiv(cdiv(mt, bm_b), lanes * grid.y)
                    );
                    for (uint32_t nnb = 1; nnb <= bn; ++nnb) {
                        if (bn % nnb != 0) {
                            continue;
                        }
                        const auto pl = make_plan(g, grid, go, bm_b, npb, bn / nnb);
                        if (!pl) {
                            continue;
                        }
                        const double cols = static_cast<double>(pl->blocks.size()) * go;
                        const double steps = static_cast<double>(ct) * npb;
                        const double fpu = steps * taps * bm_b * bn * kProduct;
                        const double dm = steps * taps * bm_b * kBuild;
                        const double reads = cols * pl->rows * steps *
                                (bm_b * 32.0 + halo) * 64 +
                            cols * steps * taps * bn * 2048.0;
                        const double cost = std::max({fpu, dm, reads / kDram * kClock});
                        out.push_back(
                            {go,
                             bm_b,
                             npb,
                             bn / nnb,
                             npb * nnb,
                             cost,
                             cb_bytes(*pl, kPageWords * 4)}
                        );
                    }
                }
            }
            std::stable_sort(
                out.begin(), out.end(), [](const Candidate& a, const Candidate& b) {
                    return std::make_tuple(a.blocks, a.npb, a.cost) <
                        std::make_tuple(b.blocks, b.npb, b.cost);
                }
            );
            return out;
        }

        // bool [32]: output pixel 32 m + r reads an in-image input pixel for tap (dy,
        // dx).
        std::array<bool, 32> valid_rows(
            const Geometry& g, uint32_t m, uint32_t dy, uint32_t dx
        ) {
            std::array<bool, 32> ok{};
            const int64_t npix = static_cast<int64_t>(g.n) * g.h * g.w,
                          p = (g.k - 1) / 2;
            const int64_t hw = static_cast<int64_t>(g.h) * g.w;
            for (uint32_t r = 0; r < 32; ++r) {
                const int64_t px = 32 * static_cast<int64_t>(m) + r;
                const int64_t pc = std::min<int64_t>(px, npix - 1);
                const int64_t rem = pc % hw;
                const int64_t y = rem / g.w + dy - p, x = rem % g.w + dx - p;
                ok[r] = px < npix && y >= 0 && y < g.h && x >= 0 && x < g.w;
            }
            return ok;
        }

        // One compute core of a plan: its first output tile, first Cout tile, and k0
        // (even A tiles [0, k0) are built by the reader, odd tiles and even [k0,
        // n_even) by the writer).
        struct Core {
            uint32_t x, y, m0, n0, k0;
        };

        // A plan with its cores (index c in `cores`) and copy table: page 2 (c npb +
        // pb) holds the reader's words of core c's pixel block pb, the next page the
        // writer's.
        struct CoreTables {
            Plan plan;
            std::vector<Core> cores;
            uint32_t page_words = 0;
            uint64_t cb_bytes = 0;
            std::vector<uint32_t> table;
            std::optional<ttnn::Tensor> device;
        };

        std::unique_ptr<CoreTables> make_layout(const Plan& pl) {
            auto out = std::make_unique<CoreTables>();
            out->plan = pl;
            std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> pages;
            // The column top's reader also sends W': its share of the A0 builds shrinks
            // by the send's time in built tiles.
            const uint32_t send = cdiv(
                static_cast<uint32_t>(kSendPerTile) * pl.taps * pl.bn, kBuildRisc
            );
            for (const uint32_t x : pl.xs) {
                for (uint32_t y = 0; y < pl.rows; ++y) {
                    const uint32_t m0 = pl.first_tile(x, y);
                    uint32_t k0 = pl.n_even;
                    if (pl.feed && y == 0 && pl.rows > 1) {
                        const int64_t share = (static_cast<int64_t>(pl.n_even) +
                                               pl.n_odd - send) /
                            2;
                        k0 = static_cast<uint32_t>(
                            std::clamp<int64_t>(share, 0, pl.n_even)
                        );
                    }
                    out->cores.push_back({x, y, m0, pl.col[x].second * pl.bn, k0});
                    for (uint32_t pb = 0; pb < pl.npb; ++pb) {
                        const auto [tile_lo, t_in] = pl.block_range(m0, pb);
                        if (t_in > pl.t_in_max) {
                            throw std::logic_error(
                                "ktt_conv_nhwc: a pixel block reads more tiles than a "
                                "planes slot"
                            );
                        }
                        // Per A tile (m - first) taps + t: its words, even and odd
                        // tiles apart.
                        std::vector<std::vector<uint32_t>> tiles[2];
                        uint32_t i = 0;
                        for (
                            uint32_t m = m0 + pb * pl.bm_b; m < m0 + (pb + 1) * pl.bm_b;
                            ++m
                        ) {
                            for (uint32_t t = 0; t < pl.taps; ++t) {
                                tiles[i & 1].emplace_back();
                                copy_runs(
                                    pl.g,
                                    m,
                                    t,
                                    true,
                                    tile_lo,
                                    t_in,
                                    pl.t_in_max,
                                    tiles[i & 1].back()
                                );
                                ++i;
                            }
                        }
                        if (tiles[0].size() != pl.n_even ||
                            tiles[1].size() != pl.n_odd) {
                            throw std::logic_error(
                                "ktt_conv_nhwc: A tile counts do not match the plan"
                            );
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
                            static_cast<uint32_t>(
                                std::max(reader.size(), writer.size())
                            )
                        );
                        pages.emplace_back(std::move(reader), std::move(writer));
                    }
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

        // The copy table of `l` on the device: uint32 ROW_MAJOR [2 cores npb,
        // page_words] in DRAM.
        ttnn::Tensor upload_table(const CoreTables& l) {
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
            std::map<PlanKey, std::unique_ptr<CoreTables>> layouts;
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
            uint32_t go = 0, bm_b = 0, npb = 0, bn_b = 0, iters = 1;
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

        // io: x, W', table, X^T (the side output) when given, then out.
        program::Built describe(
            const CoreTables& l,
            const std::vector<ttnn::Tensor>& io,
            const ttnn::ComputeKernelConfig& config,
            uint32_t iters
        ) {
            using namespace tt::tt_metal;
            const Plan& pl = l.plan;
            const auto& x = io[0];
            const auto& wp = io[1];
            const auto& tab = io[2];
            const bool side = io.size() > 4;
            const uint32_t out_index = side ? 4 : 3;
            const auto& out = io[out_index];
            // Without a side output the writer's X^T accessor is the output's, unused.
            const auto& xt = io[3];
            auto phys = [](uint32_t cx, uint32_t cy) {
                const auto c = detail::device()->worker_core_from_logical_core(
                    CoreCoord(cx, cy)
                );
                return std::pair<uint32_t, uint32_t>(c.x, c.y);
            };
            std::vector<CoreRange> comp_ranges, feed_ranges;
            for (const auto& [x0, fx] : pl.blocks) {
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

            const uint32_t blocks = static_cast<uint32_t>(pl.blocks.size());
            KernelDescriptor::RuntimeArgs rd, wr, fd, cp;
            for (size_t c = 0; c < l.cores.size(); ++c) {
                const auto& core = l.cores[c];
                const uint32_t tab_page0 = static_cast<uint32_t>(c) * pl.npb * 2;
                // W': the column top (low y, where NoC0 flows from) multicasts to rows
                // 1 .. rows - 1.
                const bool top = core.y == 0;
                std::array<uint32_t, 5> mc_w{};
                if (top && pl.rows > 1) {
                    const auto s = phys(core.x, 1), e = phys(core.x, pl.rows - 1);
                    mc_w = {s.first, s.second, e.first, e.second, pl.rows - 1};
                }
                const auto snd = phys(core.x, 0);
                const auto [b, j] = pl.col[core.x];
                const auto [x0, fx] = pl.blocks[b];
                const uint32_t rot = (static_cast<uint32_t>(b) * 8 / blocks) % pl.ct;
                bool lead = true;
                auto ld = phys(core.x, core.y);
                if (fx >= 0) {
                    // Planes from the block's feeder, right of the block in its panel
                    // (NoC1 flows from high x); it multicasts to the go cores of the
                    // group, rectangle given end-first.
                    lead = false;
                    ld = phys(fx, core.y);
                    if (j == 0) {
                        const auto s = phys(x0 + pl.go - 1, core.y),
                                   e = phys(x0, core.y);
                        std::vector<uint32_t> fa{
                            core.m0, s.first, s.second, e.first, e.second, pl.go, rot
                        };
                        fd.push_back({CoreCoord(fx, core.y), fa});
                    }
                }
                const CoreCoord at(core.x, core.y);
                rd.push_back(
                    {at,
                     {core.n0,
                      tab_page0,
                      pl.n_even,
                      top ? 1u : 0u,
                      mc_w[0],
                      mc_w[1],
                      mc_w[2],
                      mc_w[3],
                      mc_w[4],
                      snd.first,
                      snd.second,
                      rot,
                      core.k0}}
                );
                // X^T comes from one column per pixel group.
                const uint32_t turns = side && j == 0 ? 1 : 0;
                wr.push_back(
                    {at,
                     {core.m0,
                      core.n0,
                      tab_page0,
                      pl.n_odd,
                      pl.n_even,
                      core.k0,
                      lead ? 1u : 0u,
                      ld.first,
                      ld.second,
                      rot,
                      turns}}
                );
                cp.push_back({at, {turns}});
            }
            program::mark(program::Phase::Args);

            ProgramDescriptor desc;
            const auto bf16 = tt::DataFormat::Float16_b;
            const uint32_t w_block = pl.taps * pl.bn_b;
            const uint32_t page_bytes = l.page_words * 4;
            const uint32_t slot_bytes = pl.t_in_max * kTile;
            const uint32_t part_pages = pl.bm_b * pl.bn;

            // W' and planes rings are 4 slots deep (step g: compute on g - 1, builds g,
            // multicast g + 1, reads g + 2); A rings 2 blocks; table rings 2 pixel
            // blocks.
            struct Cb {
                uint32_t index, pages, page;
                tt::DataFormat format;
            };

            std::vector<Cb> cbs = {
                {1, kSlots * w_block, kTile, bf16},
                {2, 2 * pl.n_even, kTile, bf16},
                {3, 2 * std::max(1u, pl.n_odd), kTile, bf16},
                {5, kSlots * pl.t_in_max, kTile, bf16},
                {8, 2, page_bytes, tt::DataFormat::Float32},
                {9, 2, page_bytes, tt::DataFormat::Float32},
                {16, 2 * pl.sub, kTile, bf16},
                {24, part_pages, kPart, tt::DataFormat::Float32},
            };
            if (side) {
                // Transposed input tiles: compute is at most two steps ahead of the
                // writer.
                cbs.push_back({17, 2 * pl.bm_b, kTile, bf16});
            }
            auto pages_of = [&](uint32_t index) {
                for (const auto& cb : cbs) {
                    if (cb.index == index) {
                        return cb.pages;
                    }
                }
                throw std::logic_error("ktt_conv_nhwc: unknown circular buffer");
            };
            // Pushes per transaction; bytes written from the base of the rings the
            // movers address by slot (W', A0 blocks, planes, tables).
            std::vector<program::Ring> rings = {
                {.cb = 1,
                 .pushes = {w_block},
                 .written = static_cast<uint64_t>(kSlots) * w_block * kTile},
                {.cb = 2, .pushes = {pl.n_even}, .written = 2ull * pl.n_even * kTile},
                {.cb = 5, .written = static_cast<uint64_t>(kSlots) * slot_bytes},
                {.cb = 8, .written = 2ull * page_bytes},
                {.cb = 9, .written = 2ull * page_bytes},
                {.cb = 16, .pushes = {pl.sub}},
                {.cb = 24, .pushes = {pl.sub}},
            };
            if (pl.n_odd > 0) {
                rings.push_back({.cb = 3, .pushes = {pl.n_odd}});
            }
            if (side) {
                rings.push_back({.cb = 17, .pushes = {1}});
                if (pl.n_odd == 0) {
                    throw std::logic_error(
                        "ktt_conv_nhwc: the side output is drained with the odd A tiles"
                    );
                }
            }
            // The writer addresses A0 block s at ring position s & 1.
            if (pages_of(2) != 2 * pl.n_even) {
                throw std::logic_error(
                    "ktt_conv_nhwc: the A0 ring must hold exactly two blocks"
                );
            }
            for (const auto& core : l.cores) {
                for (uint32_t pb = 0; pb < pl.npb; ++pb) {
                    // A block's input tiles land in one planes slot (both planes).
                    if (pl.block_range(core.m0, pb).second * kTile > slot_bytes) {
                        throw std::logic_error(
                            "ktt_conv_nhwc: a pixel block's input exceeds its planes "
                            "slot"
                        );
                    }
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
            // 0 W count, 1 W flag, 2 planes ready, 3 planes done, 4 planes count, 5
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
            // One NoC per mover on every core: reader NoC0 (W' multicast down a
            // column), writer and feeder NoC1 (planes multicast toward low x).
            const NOC reader_noc = NOC::NOC_0;
            const NOC writer_noc = reader_noc == NOC::NOC_0 ? NOC::NOC_1 : NOC::NOC_0;
            if (writer_noc == reader_noc) {
                throw std::logic_error("ktt_conv_nhwc: reader and writer share a NoC");
            }
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("nn/ktt_conv_nhwc/reader.cpp"),
                    .core_ranges = comp,
                    .compile_time_args = concat(accessor(wp), accessor(tab)),
                    .runtime_args = rd,
                    .common_runtime_args =
                        {wp.buffer()->address(),
                         tab.buffer()->address(),
                         pl.ct,
                         pl.nt,
                         pl.taps,
                         pl.bn_b,
                         pl.nnb,
                         pl.npb,
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
                    .kernel_source = program::source("nn/ktt_conv_nhwc/writer.cpp"),
                    .core_ranges = comp,
                    .compile_time_args = concat(
                        concat(concat(accessor(x), accessor(tab)), accessor(out)),
                        accessor(xt)
                    ),
                    .runtime_args = wr,
                    .common_runtime_args =
                        {x.buffer()->address(),
                         tab.buffer()->address(),
                         out.buffer()->address(),
                         pl.ct,
                         pl.nt,
                         pl.bn_b,
                         pl.nnb,
                         pl.npb,
                         iters,
                         4,
                         5,
                         2,
                         3,
                         6,
                         7,
                         page_bytes,
                         slot_bytes,
                         pl.sub,
                         pl.mt,
                         pl.bm_b,
                         min_off,
                         pl.max_off,
                         xt.buffer()->address()},
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
                 .tensor = 1},
                {.kernel = 0,
                 .entry = program::Slot::kCommon,
                 .position = 1,
                 .tensor = 2},
                {.kernel = 1,
                 .entry = program::Slot::kCommon,
                 .position = 0,
                 .tensor = 0},
                {.kernel = 1,
                 .entry = program::Slot::kCommon,
                 .position = 1,
                 .tensor = 2},
                {.kernel = 1,
                 .entry = program::Slot::kCommon,
                 .position = 2,
                 .tensor = out_index},
                {.kernel = 1,
                 .entry = program::Slot::kCommon,
                 .position = 22,
                 .tensor = 3},
            };
            std::vector<program::ArgCount> args = {
                {.kernel = 0, .per_core = kReaderArgs, .common = kReaderCommon},
                {.kernel = 1, .per_core = kWriterArgs, .common = kWriterCommon},
            };
            if (!feed_ranges.empty()) {
                args.push_back(
                    {.kernel = 2, .per_core = kFeederArgs, .common = kFeederCommon}
                );
                desc.kernels.push_back(
                    KernelDescriptor{
                        .kernel_source = program::source("nn/ktt_conv_nhwc/feeder.cpp"),
                        .core_ranges = CoreRangeSet(feed_ranges).merge_ranges(),
                        .compile_time_args = accessor(x),
                        .runtime_args = fd,
                        .common_runtime_args =
                            {x.buffer()->address(),
                             pl.ct,
                             pl.npb,
                             iters,
                             4,
                             5,
                             slot_bytes,
                             pl.mt,
                             pl.bm_b,
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
                     .tensor = 0}
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
                    .kernel_source = program::source("nn/ktt_conv_nhwc/compute.cpp"),
                    .core_ranges = comp,
                    .runtime_args = cp,
                    .common_runtime_args =
                        {pl.ct,
                         pl.taps,
                         pl.bn_b,
                         pl.nnb,
                         pl.npb,
                         pl.sub,
                         iters,
                         pl.bm_b,
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

    void copy_runs(
        const Geometry& g,
        uint32_t m,
        uint32_t t,
        bool live,
        uint32_t tile_lo,
        uint32_t t_in,
        uint32_t slot_tiles,
        std::vector<uint32_t>& words
    ) {
        const uint32_t dy = t / g.k, dx = t % g.k;
        const int64_t p = (g.k - 1) / 2;
        const int64_t off = (static_cast<int64_t>(dy) - p) * g.w +
            (static_cast<int64_t>(dx) - p);
        auto ok = valid_rows(g, m, dy, dx);
        if (!live) {
            ok.fill(false);
        }
        const uint32_t slot_bytes = slot_tiles * kTile;
        const size_t head = words.size();
        words.push_back(0);
        uint32_t runs = 0;
        for (uint32_t f = 0; f < 4; ++f) {
            const uint32_t half = f >> 1, plane = f & 1;
            uint32_t r = 0;
            while (r < 16) {
                const bool v = ok[16 * half + r];
                uint32_t e = r + 1;
                while (e < 16 && ok[16 * half + e] == v) {
                    ++e;
                }
                const uint32_t bytes = (e - r) * 32;
                uint32_t src = kZero;
                if (v) {
                    const int64_t gpx = 32 * static_cast<int64_t>(m) + 16 * half + r +
                        off;
                    const int64_t rel = gpx - 32 * static_cast<int64_t>(tile_lo);
                    if (rel < 0 ||
                        gpx + (e - r) > 32 * static_cast<int64_t>(tile_lo + t_in)) {
                        throw std::logic_error(
                            "ktt_conv_nhwc: a tap reads outside the block's input range"
                        );
                    }
                    src = plane * (slot_bytes / 2) + static_cast<uint32_t>(rel) * 32;
                    if (static_cast<uint64_t>(src) + bytes > slot_bytes) {
                        throw std::logic_error(
                            "ktt_conv_nhwc: a copy run reads past its planes slot"
                        );
                    }
                }
                words.insert(words.end(), {f * kFace + r * 32, src, bytes});
                ++runs;
                r = e;
            }
        }
        words[head] = runs;
    }

    std::vector<uint16_t> prepare(const std::vector<uint16_t>& w, const Geometry& g) {
        const uint32_t ct = cdiv(g.c, 32), nt = cdiv(g.o, 32), taps = g.k * g.k,
                       np = nt * 32;
        if (w.size() != static_cast<size_t>(g.o) * g.c * taps) {
            throw std::invalid_argument(
                "ktt_conv_nhwc: weight values do not match the geometry"
            );
        }
        std::vector<uint16_t> out(static_cast<size_t>(ct) * taps * 32 * np, 0);
        for (uint32_t o = 0; o < g.o; ++o) {
            for (uint32_t c = 0; c < g.c; ++c) {
                const uint32_t cb = c / 32, ci = c % 32;
                for (uint32_t t = 0; t < taps; ++t) {
                    const size_t row = (static_cast<size_t>(cb) * taps + t) * 32 + ci;
                    out[row * np +
                        o] = w[(static_cast<size_t>(o) * g.c + c) * taps + t];
                }
            }
        }
        return out;
    }

    std::vector<uint16_t> matmul_weight(
        const std::vector<uint16_t>& w, const Geometry& g
    ) {
        if (w.size() != static_cast<size_t>(g.o) * g.c) {
            throw std::invalid_argument(
                "ktt_conv_nhwc: weight values do not match the geometry"
            );
        }
        std::vector<uint16_t> out(static_cast<size_t>(g.c) * g.o);
        for (uint32_t o = 0; o < g.o; ++o) {
            for (uint32_t c = 0; c < g.c; ++c) {
                out[static_cast<size_t>(c) * g.o +
                    o] = w[static_cast<size_t>(o) * g.c + c];
            }
        }
        return out;
    }

    std::optional<ttnn::Tensor> run(
        const ttnn::Tensor& x,
        const ttnn::Tensor& wp,
        const Geometry& g,
        const ttnn::ComputeKernelConfig& config,
        const tt::tt_metal::MemoryConfig& memory,
        Tables tables,
        const ttnn::Tensor* xt
    ) {
        using namespace tt::tt_metal;
        const uint32_t npix = g.n * g.h * g.w;
        auto out = ttnn::empty(
            ttnn::Shape({1, 1, npix, g.o}),
            DataType::BFLOAT16,
            Layout::TILE,
            detail::device(),
            memory
        );
        program::mark(program::Phase::Alloc);
        // The cheapest plan whose circular buffers fit the budget (read after the
        // output is allocated) and whose copy table `tables` allows; debug_plan()'s
        // plan alone when set.
        const uint64_t budget = program::cb_budget();
        CoreTables* chosen = nullptr;
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
            const std::vector<Candidate> only = {
                {f.go, f.bm_b, f.npb, f.bn_b, 1, 0, 0}
            };
            for (const auto& cand : f.go > 0 ? only : entry.order) {
                if (cand.cb_lower > budget) {
                    continue;
                }
                const PlanKey key{cand.go, cand.bm_b, cand.npb, cand.bn_b};
                const auto it = entry.layouts.find(key);
                CoreTables* l = it == entry.layouts.end() ? nullptr : it->second.get();
                if (!l) {
                    if (tables == Tables::CachedOnly) {
                        continue;
                    }
                    const auto pl = make_plan(
                        g, grid, cand.go, cand.bm_b, cand.npb, cand.bn_b
                    );
                    if (!pl) {
                        continue;
                    }
                    l = (entry.layouts[key] = make_layout(*pl)).get();
                }
                // The side output's ring: two steps of transposed input tiles.
                const uint64_t side_bytes = xt ? 2ull * l->plan.bm_b * kTile : 0;
                if (l->cb_bytes + side_bytes > budget ||
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
            return std::nullopt;
        }
        const ttnn::Tensor table = once ? *once : *chosen->device;
        const Plan& pl = chosen->plan;
        program::Key key("ktt_conv_nhwc");
        key.add(x).add(wp).add(table).add(out);
        key.add(static_cast<int64_t>(config.math_fidelity))
            .add(static_cast<int64_t>(config.math_approx_mode));
        for (
            const int64_t v :
            {g.n, g.c, g.h, g.w, g.o, g.k, pl.go, pl.bm_b, pl.npb, pl.bn_b, iters}
        ) {
            key.add(v);
        }
        // The output last.
        std::vector<ttnn::Tensor> io = {x, wp, table};
        if (xt) {
            key.add(*xt);
            io.push_back(*xt);
        }
        io.push_back(out);
        const CoreTables& tables_of = *chosen;
        return program::launch(key, io, [&] {
            return describe(tables_of, io, config, iters);
        });
    }

    void debug_plan(
        uint32_t go, uint32_t bm_b, uint32_t npb, uint32_t bn_b, uint32_t iters
    ) {
        const std::lock_guard lock(cache_mutex());
        forced() = Forced{go, bm_b, npb, bn_b, std::max(1u, iters)};
    }

    std::string last_plan() {
        const std::lock_guard lock(cache_mutex());
        return last();
    }

    void clear() {
        const std::lock_guard lock(cache_mutex());
        cache().clear();
    }

}  // namespace ktt::bridge::ktt_conv_nhwc
