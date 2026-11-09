// Host side of ktt_conv_nhwc: plans (pixel blocks of go Cout-group columns inside one panel,
// a planes feeder column right of each block when go > 1, rows of pixel groups), their per-core
// A-tile copy tables, the W' layout, circular buffers, kernels and runtime args.
#include "aten/nn/ktt_conv_nhwc/program.hpp"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "core/device/handle.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"

namespace ktt::bridge::ktt_conv_nhwc {
    namespace {

        constexpr uint32_t kTile = 2048;          // bf16 tile
        constexpr uint32_t kPart = 4096;          // fp32 tile
        constexpr uint32_t kFace = 512;           // bf16 face (16 x 16)
        constexpr uint32_t kZero = 0xFFFFFFFFu;   // copy source: MEM_ZEROS
        constexpr uint32_t kPageWords = 16;       // table pages: multiples of 64 B
        constexpr uint32_t kSlots = 4;            // W' and planes ring slots
        // Plan cost model (cycles at kClock): per tile product, per built A tile, DRAM bytes.
        constexpr double kClock = 1.35e9;
        constexpr double kProduct = 32, kBuild = 110, kDram = 400e9;
        // Column-top reader's W' send per W' tile, and one built A tile per RISC (cycles):
        // they size the top reader's share of its A0 builds (k0).
        constexpr uint32_t kSendPerTile = 72, kBuildRisc = 206;
        // Per-core runtime args: reader, writer, feeder, compute (at most max_runtime_args).
        constexpr uint32_t kReaderArgs = 14, kWriterArgs = 18, kFeederArgs = 8, kComputeArgs = 2;
        static_assert(kReaderArgs <= tt::tt_metal::max_runtime_args && kWriterArgs <= tt::tt_metal::max_runtime_args);

        uint32_t cdiv(uint32_t a, uint32_t b) {
            return (a + b - 1) / b;
        }

        // The device grid and its panels.
        struct Grid {
            uint32_t x = 0, y = 0;
            std::vector<std::pair<uint32_t, uint32_t>> panels;
        };

        // Pixel blocks of go adjacent columns inside one panel; with go > 1 each block has a
        // planes feeder column right of it in the same panel. Column j of a block is Cout group
        // j; pixel group (block b, row y) owns bm output tiles.
        struct Plan {
            Geometry g;
            uint32_t p = 0, ct = 0, nt = 0, npix = 0, mt = 0, taps = 0;
            uint32_t go = 1, bn = 1, bm = 1, sub = 1, rows = 1;
            // Output block (pixel tiles x Cout tiles) a core holds in L1 at once; the
            // circular buffers are sized from it. The kernels take the whole share as one block.
            uint32_t block_m = 1, block_n = 1;
            bool feed = false;
            // (first column, feeder column or -1) per block.
            std::vector<std::pair<uint32_t, int64_t>> blocks;
            // Compute columns, and per grid column its (block or -1, j).
            std::vector<uint32_t> xs;
            std::vector<std::pair<int64_t, uint32_t>> col;
            uint32_t ncols = 0;

            // Output tiles [m0, m0 + bm) of core (x, y); tiles >= Mt are zero (not written).
            std::pair<uint32_t, uint32_t> tiles(uint32_t x, uint32_t y) const {
                const uint32_t gi = static_cast<uint32_t>(col[x].first) * rows + y;
                return {gi * bm, (gi + 1) * bm};
            }

            // Input pixel offset of tap t (row-major over the K x K window).
            int64_t offset(uint32_t t) const {
                const int64_t dy = t / g.k, dx = t % g.k;
                return (dy - p) * static_cast<int64_t>(g.w) + (dx - p);
            }
        };

        // Block slots (first column, feeder column or -1) of go columns over the panels.
        std::vector<std::pair<uint32_t, int64_t>> slots_of(const Grid& grid, uint32_t go) {
            const bool feed = go > 1;
            std::vector<std::pair<uint32_t, int64_t>> slots;
            for (const auto& [a, b] : grid.panels) {
                const uint32_t width = b - a + 1;
                const uint32_t k = feed ? width / (go + 1) : width / go;
                for (uint32_t i = 0; i < k; ++i) {
                    slots.emplace_back(a + i * go, feed ? static_cast<int64_t>(a + k * go + i) : -1);
                }
            }
            return slots;
        }

        // Rows and blocks of go-column blocks with bm tiles per pixel group; nullopt when they
        // do not cover the output.
        std::optional<std::pair<uint32_t, uint32_t>> extent(
            const Grid& grid, size_t slots, uint32_t mt, uint32_t bm
        ) {
            if (slots == 0) {
                return std::nullopt;
            }
            const uint32_t s = static_cast<uint32_t>(slots);
            const uint32_t groups = cdiv(mt, bm);
            const uint32_t rows = std::min(grid.y, cdiv(groups, s));
            const uint32_t blocks = std::min(s, cdiv(groups, rows));
            if (static_cast<uint64_t>(blocks) * rows * bm < mt) {
                return std::nullopt;
            }
            return std::make_pair(rows, blocks);
        }

        std::optional<Plan> make_plan(const Geometry& g, const Grid& grid, uint32_t go, uint32_t bm) {
            Plan pl;
            pl.g = g;
            pl.p = (g.k - 1) / 2;
            pl.ct = cdiv(g.c, 32);
            pl.nt = cdiv(g.o, 32);
            pl.npix = g.n * g.h * g.w;
            pl.mt = cdiv(pl.npix, 32);
            pl.taps = g.k * g.k;
            if (pl.nt % go != 0) {
                return std::nullopt;
            }
            pl.go = go;
            pl.bn = pl.nt / go;
            pl.bm = bm;
            pl.block_m = pl.bm;
            pl.block_n = pl.bn;
            for (const uint32_t s : {4u, 3u, 2u, 1u}) {
                if (pl.bn % s == 0) {
                    pl.sub = s;
                    break;
                }
            }
            pl.feed = go > 1;
            const auto slots = slots_of(grid, go);
            const auto ext = extent(grid, slots.size(), pl.mt, bm);
            if (!ext) {
                return std::nullopt;
            }
            pl.rows = ext->first;
            pl.blocks.assign(slots.begin(), slots.begin() + ext->second);
            pl.col.assign(grid.x, {-1, 0});
            uint32_t last = 0;
            for (size_t b = 0; b < pl.blocks.size(); ++b) {
                const auto [x0, fx] = pl.blocks[b];
                for (uint32_t j = 0; j < go; ++j) {
                    pl.xs.push_back(x0 + j);
                    pl.col[x0 + j] = {static_cast<int64_t>(b), j};
                    last = std::max(last, x0 + j);
                }
                if (fx >= 0) {
                    last = std::max(last, static_cast<uint32_t>(fx));
                }
            }
            pl.ncols = last + 1;
            return pl;
        }

        // Input tiles [tile_lo, tile_lo + t_in) the output tiles [m0, m1) read (halo included).
        std::pair<uint32_t, uint32_t> input_range(const Plan& pl, uint32_t m0, uint32_t m1) {
            if (m0 >= pl.mt) {
                return {0, 1};
            }
            const int64_t reach = static_cast<int64_t>(pl.p) * (pl.g.w + 1);
            const int64_t lo = std::max<int64_t>(0, 32 * static_cast<int64_t>(m0) - reach);
            const int64_t hi = std::min<int64_t>(
                32 * static_cast<int64_t>(pl.mt) - 1, 32 * static_cast<int64_t>(m1) - 1 + reach
            );
            const uint32_t tile_lo = static_cast<uint32_t>(lo / 32);
            return {tile_lo, static_cast<uint32_t>(hi / 32 + 1) - tile_lo};
        }

        // A candidate plan: go, bm, modeled cost, circular-buffer bytes without the tables.
        struct Candidate {
            uint32_t go, bm;
            double cost;
            uint64_t cb_lower;
        };

        // Circular-buffer bytes per core of a plan with the widest input range t_in_max and
        // table pages of page_words.
        uint64_t cb_bytes(uint32_t taps, uint32_t bn, uint32_t bm, uint32_t sub, uint32_t t_in_max, uint32_t page_words) {
            const uint64_t n_even = cdiv(bm * taps, 2), n_odd = std::max(1u, bm * taps / 2);
            return static_cast<uint64_t>(kSlots) * taps * bn * kTile + 2 * n_even * kTile + 2 * n_odd * kTile +
                static_cast<uint64_t>(kSlots) * t_in_max * kTile + 2ull * page_words * 4 + 2ull * sub * kTile +
                std::max(1u, bm * bn) * static_cast<uint64_t>(kPart);
        }

        // Every plan of g, cheapest first (modeled max(FPU, A builds, DRAM) per core; ties in
        // enumeration order: go ascending, then bm).
        std::vector<Candidate> candidates(const Geometry& g, const Grid& grid) {
            const uint32_t ct = cdiv(g.c, 32), nt = cdiv(g.o, 32), mt = cdiv(g.n * g.h * g.w, 32);
            const uint32_t taps = g.k * g.k, p = (g.k - 1) / 2;
            const double halo = 2.0 * (static_cast<double>(p) * g.w + p);
            std::vector<Candidate> out;
            for (const uint32_t go : {1u, 2u, 4u, 8u, 16u}) {
                if (nt % go != 0 || go > grid.x) {
                    continue;
                }
                const uint32_t bn = nt / go;
                uint32_t sub = 1;
                for (const uint32_t s : {4u, 3u, 2u, 1u}) {
                    if (bn % s == 0) {
                        sub = s;
                        break;
                    }
                }
                const size_t slots = slots_of(grid, go).size();
                for (uint32_t bm = 1; bm <= mt; ++bm) {
                    const auto ext = extent(grid, slots, mt, bm);
                    if (!ext) {
                        continue;
                    }
                    const auto [rows, blocks] = *ext;
                    const double cols = static_cast<double>(blocks) * go;
                    const double fpu = static_cast<double>(ct) * taps * bm * bn * kProduct;
                    const double dm = static_cast<double>(ct) * taps * bm * kBuild;
                    const double reads = cols * rows * ct * (bm * 32.0 + halo) * 64 + cols * ct * taps * bn * 2048.0;
                    const double cost = std::max({fpu, dm, reads / kDram * kClock});
                    // The widest input range over the pixel groups.
                    Plan shape;
                    shape.g = g;
                    shape.p = p;
                    shape.mt = mt;
                    uint32_t t_in_max = 1;
                    for (uint32_t gi = 0; gi < blocks * rows; ++gi) {
                        t_in_max = std::max(t_in_max, input_range(shape, gi * bm, (gi + 1) * bm).second);
                    }
                    out.push_back({go, bm, cost, cb_bytes(taps, bn, bm, sub, t_in_max, 0)});
                }
            }
            std::stable_sort(out.begin(), out.end(), [](const Candidate& a, const Candidate& b) {
                return a.cost < b.cost;
            });
            return out;
        }

        // bool [32]: output pixel 32 m + r reads an in-image input pixel for tap (dy, dx).
        std::array<bool, 32> valid_rows(const Plan& pl, uint32_t m, uint32_t dy, uint32_t dx) {
            std::array<bool, 32> ok{};
            const int64_t hw = static_cast<int64_t>(pl.g.h) * pl.g.w;
            for (uint32_t r = 0; r < 32; ++r) {
                const int64_t px = 32 * static_cast<int64_t>(m) + r;
                const int64_t pc = std::min<int64_t>(px, pl.npix - 1);
                const int64_t rem = pc % hw;
                const int64_t y = rem / pl.g.w + dy - pl.p, x = rem % pl.g.w + dx - pl.p;
                ok[r] = px < pl.npix && y >= 0 && y < pl.g.h && x >= 0 && x < pl.g.w;
            }
            return ok;
        }

        // Table words of A tile (m, t): run count, then per run (dst byte offset in the tile, src
        // byte offset in the planes slot or kZero, bytes); per face, its 16 rows split only where
        // validity changes.
        void tile_runs(const Plan& pl, uint32_t m, uint32_t t, uint32_t tile_lo, uint32_t t_in, std::vector<uint32_t>& words) {
            const uint32_t dy = t / pl.g.k, dx = t % pl.g.k;
            const int64_t off = pl.offset(t);
            const auto ok = valid_rows(pl, m, dy, dx);
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
                    uint32_t src = kZero;
                    if (v) {
                        const int64_t gpx = 32 * static_cast<int64_t>(m) + 16 * half + r + off;
                        const int64_t rel = gpx - 32 * static_cast<int64_t>(tile_lo);
                        if (rel < 0 || gpx + (e - r) > 32 * static_cast<int64_t>(tile_lo + t_in)) {
                            throw std::logic_error("ktt_conv_nhwc: a tap reads outside the core's input range");
                        }
                        src = plane * t_in * 1024 + static_cast<uint32_t>(rel) * 32;
                    }
                    words.insert(words.end(), {f * kFace + r * 32, src, (e - r) * 32});
                    ++runs;
                    r = e;
                }
            }
            words[head] = runs;
        }

        // One compute core of a plan: its output tiles, Cout tiles, input range and A tiles
        // (even tiles [0, k0) built by the reader, odd tiles and even [k0, n_even) by the writer).
        struct Core {
            uint32_t x, y, m0, m1, n0, tile_lo, t_in, n_even, n_odd, k0;
        };

        // A plan with its cores and copy table: page 2 (x rows + y) holds core (x, y)'s reader
        // words, the next page its writer words.
        struct CoreTables {
            Plan plan;
            std::vector<Core> cores;
            uint32_t page_words = 0, t_in_max = 0;
            uint64_t cb_bytes = 0;
            std::vector<uint32_t> table;
            std::optional<ttnn::Tensor> device;
        };

        std::unique_ptr<CoreTables> make_layout(const Plan& pl) {
            auto out = std::make_unique<CoreTables>();
            out->plan = pl;
            std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> pages;
            for (const uint32_t x : pl.xs) {
                for (uint32_t y = 0; y < pl.rows; ++y) {
                    const auto [m0, m1] = pl.tiles(x, y);
                    const uint32_t n0 = pl.col[x].second * pl.bn;
                    const auto [tile_lo, t_in] = input_range(pl, m0, m1);
                    // Per A tile (m - m0) taps + t: its words, even and odd tiles apart.
                    std::vector<std::vector<uint32_t>> tiles[2];
                    uint32_t i = 0;
                    for (uint32_t m = m0; m < m1; ++m) {
                        for (uint32_t t = 0; t < pl.taps; ++t) {
                            tiles[i & 1].emplace_back();
                            tile_runs(pl, m, t, tile_lo, t_in, tiles[i & 1].back());
                            ++i;
                        }
                    }
                    const uint32_t n_even = static_cast<uint32_t>(tiles[0].size());
                    const uint32_t n_odd = static_cast<uint32_t>(tiles[1].size());
                    uint32_t k0 = n_even;
                    if (pl.feed && y == 0 && pl.rows > 1) {
                        // The column top's reader also sends W': its share of the A0 builds
                        // shrinks by the send's time in built tiles.
                        const uint32_t send = cdiv(kSendPerTile * pl.taps * pl.bn, kBuildRisc);
                        const int64_t share = (static_cast<int64_t>(n_even) + n_odd - send) / 2;
                        k0 = static_cast<uint32_t>(std::clamp<int64_t>(share, 0, n_even));
                    }
                    std::vector<uint32_t> reader, writer;
                    for (uint32_t k = 0; k < k0; ++k) {
                        reader.insert(reader.end(), tiles[0][k].begin(), tiles[0][k].end());
                    }
                    for (const auto& w : tiles[1]) {
                        writer.insert(writer.end(), w.begin(), w.end());
                    }
                    for (uint32_t k = k0; k < n_even; ++k) {
                        writer.insert(writer.end(), tiles[0][k].begin(), tiles[0][k].end());
                    }
                    out->cores.push_back({x, y, m0, m1, n0, tile_lo, t_in, n_even, n_odd, k0});
                    out->page_words = std::max<uint32_t>(
                        out->page_words, static_cast<uint32_t>(std::max(reader.size(), writer.size()))
                    );
                    out->t_in_max = std::max(out->t_in_max, t_in);
                    pages.emplace_back(std::move(reader), std::move(writer));
                }
            }
            out->page_words = cdiv(std::max(out->page_words, kPageWords), kPageWords) * kPageWords;
            out->table.assign(static_cast<size_t>(2) * pl.ncols * pl.rows * out->page_words, 0);
            for (size_t c = 0; c < out->cores.size(); ++c) {
                const auto& core = out->cores[c];
                const size_t i = static_cast<size_t>(core.x) * pl.rows + core.y;
                std::copy(pages[c].first.begin(), pages[c].first.end(), out->table.begin() + 2 * i * out->page_words);
                std::copy(
                    pages[c].second.begin(), pages[c].second.end(), out->table.begin() + (2 * i + 1) * out->page_words
                );
            }
            out->cb_bytes = cb_bytes(pl.taps, pl.block_n, pl.block_m, pl.sub, out->t_in_max, out->page_words);
            return out;
        }

        // The copy table of `l` on the device: uint32 ROW_MAJOR [2 ncols rows, page_words] in DRAM.
        ttnn::Tensor upload_table(const CoreTables& l) {
            using namespace tt::tt_metal;
            const uint32_t pages = static_cast<uint32_t>(l.table.size() / l.page_words);
            const TensorSpec spec(
                ttnn::Shape({pages, l.page_words}),
                TensorLayout(DataType::UINT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{})
            );
            return ttnn::Tensor::from_span(ttsl::Span<const uint32_t>(l.table.data(), l.table.size()), spec, detail::device());
        }

        // A geometry's plans, cheapest first, and the layouts built for them.
        struct Entry {
            std::vector<Candidate> order;
            std::map<std::pair<uint32_t, uint32_t>, std::unique_ptr<CoreTables>> layouts;
        };

        std::mutex& cache_mutex() {
            static std::mutex m;
            return m;
        }

        // Never destroyed: device tables must not be freed after the device closes at exit.
        std::map<std::vector<uint32_t>, Entry>& cache() {
            static auto* map = new std::map<std::vector<uint32_t>, Entry>();
            return *map;
        }

        Grid grid_of() {
            const auto size = detail::device()->compute_with_storage_grid_size();
            return Grid{static_cast<uint32_t>(size.x), static_cast<uint32_t>(size.y), program::panels()};
        }

        // Asserts every push size used on a circular buffer divides its capacity (a push must
        // never straddle the ring end).
        void check_pushes(const char* name, uint32_t pages, std::initializer_list<uint32_t> pushes) {
            for (const uint32_t push : pushes) {
                if (push == 0 || pages % push != 0) {
                    throw std::logic_error(
                        std::string("ktt_conv_nhwc: CB ") + name + " of " + std::to_string(pages) +
                        " pages takes pushes of " + std::to_string(push)
                    );
                }
            }
        }

        program::Built describe(
            const CoreTables& l, const std::vector<ttnn::Tensor>& io, const ttnn::ComputeKernelConfig& config
        ) {
            using namespace tt::tt_metal;
            const Plan& pl = l.plan;
            const auto& x = io[0];
            const auto& wp = io[1];
            const auto& tab = io[2];
            const auto& out = io[3];
            auto phys = [](uint32_t cx, uint32_t cy) {
                const auto c = detail::device()->worker_core_from_logical_core(CoreCoord(cx, cy));
                return std::pair<uint32_t, uint32_t>(c.x, c.y);
            };
            std::vector<CoreRange> comp_ranges, feed_ranges;
            for (const auto& [x0, fx] : pl.blocks) {
                comp_ranges.emplace_back(CoreCoord(x0, 0), CoreCoord(x0 + pl.go - 1, pl.rows - 1));
                if (fx >= 0) {
                    feed_ranges.emplace_back(CoreCoord(fx, 0), CoreCoord(fx, pl.rows - 1));
                }
            }
            std::vector<CoreRange> all_ranges = comp_ranges;
            all_ranges.insert(all_ranges.end(), feed_ranges.begin(), feed_ranges.end());
            const CoreRangeSet comp = CoreRangeSet(comp_ranges).merge_ranges();
            const CoreRangeSet all = CoreRangeSet(all_ranges).merge_ranges();

            const uint32_t blocks = static_cast<uint32_t>(pl.blocks.size());
            // A feeder's planes slots (its t_in) land at the same offsets on its group's cores.
            std::map<std::pair<int64_t, uint32_t>, uint32_t> group_t_in;
            for (const auto& core : l.cores) {
                const auto [it, fresh] = group_t_in.emplace(std::make_pair(pl.col[core.x].first, core.y), core.t_in);
                if (!fresh && it->second != core.t_in) {
                    throw std::logic_error("ktt_conv_nhwc: a pixel group's cores read different input ranges");
                }
            }
            KernelDescriptor::RuntimeArgs rd, wr, fd, cp;
            for (const auto& core : l.cores) {
                const uint32_t i = core.x * pl.rows + core.y;
                // W': the column top (low y, where NoC0 flows from) multicasts to rows 1 .. rows - 1.
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
                std::array<uint32_t, 5> mc_p{};
                bool lead = true;
                auto ld = phys(core.x, core.y);
                if (fx >= 0) {
                    // Planes from the block's feeder, right of the block in its panel (NoC1 flows
                    // from high x); it multicasts to the go cores of the group, rectangle given
                    // end-first.
                    lead = false;
                    ld = phys(fx, core.y);
                    if (j == 0) {
                        const auto s = phys(x0 + pl.go - 1, core.y), e = phys(x0, core.y);
                        std::vector<uint32_t> fa{core.tile_lo, core.t_in, s.first, s.second, e.first, e.second, pl.go, rot};
                        fd.push_back({CoreCoord(fx, core.y), fa});
                    }
                }
                const CoreCoord at(core.x, core.y);
                rd.push_back({at, {core.n0, 2 * i, core.n_even, top ? 1u : 0u, mc_w[0], mc_w[1], mc_w[2], mc_w[3],
                                   mc_w[4], snd.first, snd.second, rot, core.t_in, core.k0}});
                wr.push_back({at, {core.m0, core.m1, core.n0, core.tile_lo, core.t_in, 2 * i + 1, core.n_odd,
                                   lead ? 1u : 0u, mc_p[0], mc_p[1], mc_p[2], mc_p[3], mc_p[4], ld.first, ld.second,
                                   rot, core.k0, core.n_even}});
                cp.push_back({at, {core.m0, core.m1}});
            }
            const std::pair<const KernelDescriptor::RuntimeArgs*, size_t> counts[] = {
                {&rd, kReaderArgs}, {&wr, kWriterArgs}, {&fd, kFeederArgs}, {&cp, kComputeArgs}
            };
            for (const auto& [args, count] : counts) {
                for (const auto& entry : *args) {
                    if (entry.second.size() != count) {
                        throw std::logic_error("ktt_conv_nhwc: unexpected runtime argument count");
                    }
                }
            }
            program::mark(program::Phase::Args);

            ProgramDescriptor desc;
            const auto bf16 = tt::DataFormat::Float16_b;
            if (pl.block_m != pl.bm || pl.block_n != pl.bn) {
                throw std::logic_error("ktt_conv_nhwc: the kernels take a core's whole share as one block");
            }
            const uint32_t w_block = pl.taps * pl.block_n;
            const uint32_t n_even = cdiv(pl.block_m * pl.taps, 2);
            const uint32_t n_odd = std::max(1u, pl.block_m * pl.taps / 2);
            const uint32_t n_odd_push = pl.block_m * pl.taps / 2;
            const uint32_t page_bytes = l.page_words * 4;
            const uint32_t part_pages = std::max(1u, pl.block_m * pl.block_n);
            for (const auto& core : l.cores) {
                if (core.n_even != n_even || core.n_odd != n_odd_push) {
                    throw std::logic_error("ktt_conv_nhwc: A ring positions assume equal A tile counts");
                }
            }
            // W' and planes rings are 4 slots deep (step g: compute on g - 1, builds g, multicast
            // g + 1, reads g + 2); A rings 2 blocks.
            check_pushes("W'", kSlots * w_block, {w_block});
            check_pushes("A0", 2 * n_even, {n_even});
            if (n_odd_push > 0) {
                check_pushes("A1", 2 * n_odd, {n_odd_push});
            }
            check_pushes("out", 2 * pl.sub, {pl.sub, 1});
            check_pushes("partials", part_pages, {pl.sub});
            struct Cb {
                uint32_t index, pages, page;
                tt::DataFormat format;
            };
            const std::vector<Cb> cbs = {
                {1, kSlots * w_block, kTile, bf16},
                {2, 2 * n_even, kTile, bf16},
                {3, 2 * n_odd, kTile, bf16},
                {5, kSlots * l.t_in_max, kTile, bf16},
                {8, 1, page_bytes, tt::DataFormat::Float32},
                {9, 1, page_bytes, tt::DataFormat::Float32},
                {16, 2 * pl.sub, kTile, bf16},
                {24, part_pages, kPart, tt::DataFormat::Float32},
            };
            // The byte range each kernel writes into a CB (slot stride x slots) fits it.
            auto check_fits = [&](uint32_t index, uint64_t bytes) {
                for (const auto& cb : cbs) {
                    if (cb.index == index && bytes > static_cast<uint64_t>(cb.pages) * cb.page) {
                        throw std::logic_error(
                            "ktt_conv_nhwc: CB " + std::to_string(index) + " holds " +
                            std::to_string(cb.pages * cb.page) + " B, a kernel writes " + std::to_string(bytes)
                        );
                    }
                }
            };
            check_fits(1, static_cast<uint64_t>(kSlots) * w_block * kTile);  // W' slots
            check_fits(2, 2ull * n_even * kTile);                             // A0 blocks
            check_fits(3, 2ull * n_odd_push * kTile);                         // A1 blocks
            for (const auto& core : l.cores) {
                check_fits(5, static_cast<uint64_t>(kSlots) * core.t_in * kTile);  // planes slots
            }
            check_fits(8, page_bytes);  // reader table page
            check_fits(9, page_bytes);  // writer table page
            check_fits(16, 2ull * pl.sub * kTile);
            check_fits(24, static_cast<uint64_t>(pl.block_m) * pl.block_n * kPart);
            for (const auto& cb : cbs) {
                desc.cbs.push_back(CBDescriptor{
                    .total_size = cb.pages * cb.page,
                    .core_ranges = all,
                    .format_descriptors = {CBFormatDescriptor{
                        .buffer_index = static_cast<uint8_t>(cb.index), .data_format = cb.format, .page_size = cb.page
                    }}
                });
            }
            // 0 W count, 1 W flag, 2 planes ready, 3 planes done, 4 planes count, 5 planes flag,
            // 6 a0 go, 7 a0 done.
            for (uint32_t id = 0; id < 8; ++id) {
                desc.semaphores.push_back(SemaphoreDescriptor{
                    .id = id, .core_type = tt::CoreType::WORKER, .core_ranges = all, .initial_value = 0
                });
            }

            auto accessor = [](const ttnn::Tensor& t) {
                return TensorAccessorArgs(*t.buffer()).get_compile_time_args();
            };
            auto concat = [](std::vector<uint32_t> a, const std::vector<uint32_t>& b) {
                a.insert(a.end(), b.begin(), b.end());
                return a;
            };
            constexpr uint32_t iters = 1;
            // One NoC per mover on every core: reader NoC0 (W' multicast down a column), writer
            // and feeder NoC1 (planes multicast toward low x).
            const NOC reader_noc = NOC::NOC_0;
            const NOC writer_noc = reader_noc == NOC::NOC_0 ? NOC::NOC_1 : NOC::NOC_0;
            if (writer_noc == reader_noc) {
                throw std::logic_error("ktt_conv_nhwc: reader and writer share a NoC");
            }
            desc.kernels.push_back(KernelDescriptor{
                .kernel_source = program::source("nn/ktt_conv_nhwc/reader.cpp"),
                .core_ranges = comp,
                .compile_time_args = concat(accessor(wp), accessor(tab)),
                .runtime_args = rd,
                .common_runtime_args = {wp.buffer()->address(), tab.buffer()->address(), pl.ct, pl.nt, pl.taps, pl.bn,
                                        iters, 0, 1, 2, 3, 6, 7, page_bytes},
                .config = DataMovementConfigDescriptor{
                    .processor = DataMovementProcessor::RISCV_1, .noc = reader_noc,
                    .noc_mode = NOC_MODE::DM_DEDICATED_NOC
                }
            });
            desc.kernels.push_back(KernelDescriptor{
                .kernel_source = program::source("nn/ktt_conv_nhwc/writer.cpp"),
                .core_ranges = comp,
                .compile_time_args = concat(concat(accessor(x), accessor(tab)), accessor(out)),
                .runtime_args = wr,
                .common_runtime_args = {x.buffer()->address(), tab.buffer()->address(), out.buffer()->address(), pl.ct,
                                        pl.nt, pl.bn, iters, 4, 5, 2, 3, 6, 7, page_bytes, pl.sub, pl.mt},
                .config = DataMovementConfigDescriptor{
                    .processor = DataMovementProcessor::RISCV_0, .noc = writer_noc,
                    .noc_mode = NOC_MODE::DM_DEDICATED_NOC
                }
            });
            std::vector<program::Slot> slots = {
                {.kernel = 0, .entry = program::Slot::kCommon, .position = 0, .tensor = 1},
                {.kernel = 0, .entry = program::Slot::kCommon, .position = 1, .tensor = 2},
                {.kernel = 1, .entry = program::Slot::kCommon, .position = 0, .tensor = 0},
                {.kernel = 1, .entry = program::Slot::kCommon, .position = 1, .tensor = 2},
                {.kernel = 1, .entry = program::Slot::kCommon, .position = 2, .tensor = 3},
            };
            if (!feed_ranges.empty()) {
                desc.kernels.push_back(KernelDescriptor{
                    .kernel_source = program::source("nn/ktt_conv_nhwc/feeder.cpp"),
                    .core_ranges = CoreRangeSet(feed_ranges).merge_ranges(),
                    .compile_time_args = accessor(x),
                    .runtime_args = fd,
                    .common_runtime_args = {x.buffer()->address(), pl.ct, iters, 4, 5},
                    .config = DataMovementConfigDescriptor{
                        .processor = DataMovementProcessor::RISCV_0, .noc = writer_noc,
                        .noc_mode = NOC_MODE::DM_DEDICATED_NOC
                    }
                });
                slots.push_back({.kernel = 2, .entry = program::Slot::kCommon, .position = 0, .tensor = 0});
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
            desc.kernels.push_back(KernelDescriptor{
                .kernel_source = program::source("nn/ktt_conv_nhwc/compute.cpp"),
                .core_ranges = comp,
                .runtime_args = cp,
                .common_runtime_args = {pl.ct, pl.taps, pl.bn, pl.sub, iters},
                .config = compute
            });
            return program::Built{std::move(desc), std::move(slots)};
        }

    }  // namespace

    std::vector<uint16_t> prepare(const std::vector<uint16_t>& w, const Geometry& g) {
        const uint32_t ct = cdiv(g.c, 32), nt = cdiv(g.o, 32), taps = g.k * g.k, np = nt * 32;
        if (w.size() != static_cast<size_t>(g.o) * g.c * taps) {
            throw std::invalid_argument("ktt_conv_nhwc: weight values do not match the geometry");
        }
        std::vector<uint16_t> out(static_cast<size_t>(ct) * taps * 32 * np, 0);
        for (uint32_t o = 0; o < g.o; ++o) {
            for (uint32_t c = 0; c < g.c; ++c) {
                const uint32_t cb = c / 32, ci = c % 32;
                for (uint32_t t = 0; t < taps; ++t) {
                    const size_t row = (static_cast<size_t>(cb) * taps + t) * 32 + ci;
                    out[row * np + o] = w[(static_cast<size_t>(o) * g.c + c) * taps + t];
                }
            }
        }
        return out;
    }

    std::vector<uint16_t> matmul_weight(const std::vector<uint16_t>& w, const Geometry& g) {
        if (w.size() != static_cast<size_t>(g.o) * g.c) {
            throw std::invalid_argument("ktt_conv_nhwc: weight values do not match the geometry");
        }
        std::vector<uint16_t> out(static_cast<size_t>(g.c) * g.o);
        for (uint32_t o = 0; o < g.o; ++o) {
            for (uint32_t c = 0; c < g.c; ++c) {
                out[static_cast<size_t>(c) * g.o + o] = w[static_cast<size_t>(o) * g.c + c];
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
        Tables tables
    ) {
        using namespace tt::tt_metal;
        const uint32_t npix = g.n * g.h * g.w;
        auto out = ttnn::empty(
            ttnn::Shape({1, 1, npix, g.o}), DataType::BFLOAT16, Layout::TILE, detail::device(), memory
        );
        program::mark(program::Phase::Alloc);
        // The cheapest plan whose circular buffers fit the budget (read after the output is
        // allocated) and whose copy table `tables` allows.
        const uint64_t budget = program::cb_budget();
        CoreTables* chosen = nullptr;
        std::optional<ttnn::Tensor> once;
        {
            const std::lock_guard lock(cache_mutex());
            const Grid grid = grid_of();
            auto& entry = cache()[{g.n, g.c, g.h, g.w, g.o, g.k}];
            if (entry.order.empty()) {
                entry.order = candidates(g, grid);
            }
            for (const auto& cand : entry.order) {
                if (cand.cb_lower > budget) {
                    continue;
                }
                const auto it = entry.layouts.find({cand.go, cand.bm});
                CoreTables* l = it == entry.layouts.end() ? nullptr : it->second.get();
                if (!l) {
                    if (tables == Tables::CachedOnly) {
                        continue;
                    }
                    const auto pl = make_plan(g, grid, cand.go, cand.bm);
                    if (!pl) {
                        continue;
                    }
                    l = (entry.layouts[{cand.go, cand.bm}] = make_layout(*pl)).get();
                }
                if (l->cb_bytes > budget || (tables == Tables::CachedOnly && !l->device)) {
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
        }
        if (!chosen) {
            return std::nullopt;
        }
        const ttnn::Tensor table = once ? *once : *chosen->device;
        const Plan& pl = chosen->plan;
        program::Key key("ktt_conv_nhwc");
        key.add(x).add(wp).add(table).add(out);
        key.add(static_cast<int64_t>(config.math_fidelity)).add(static_cast<int64_t>(config.math_approx_mode));
        for (const int64_t v : {g.n, g.c, g.h, g.w, g.o, g.k, pl.go, pl.bm}) {
            key.add(v);
        }
        const std::vector<ttnn::Tensor> io = {x, wp, table, out};
        const CoreTables& tables_of = *chosen;
        return program::launch(key, io, [&] {
            return describe(tables_of, io, config);
        });
    }

    void clear() {
        const std::lock_guard lock(cache_mutex());
        cache().clear();
    }

}  // namespace ktt::bridge::ktt_conv_nhwc
