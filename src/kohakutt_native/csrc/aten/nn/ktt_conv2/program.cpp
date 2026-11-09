// Host side of ktt_conv2: gather geometries (forward, input gradient as phases),
// channel-tile ranges, one-shot circular-buffer sizing against the program budget, the
// sparse weight preparation, the gather program's 2-D split (column tile groups x output
// row tile ranges) and the weight-gradient programs.
#include "aten/nn/ktt_conv2/program.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "aten/nn/ktt_conv/args.hpp"
#include "core/device/handle.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/generic/generic_op.hpp"

namespace ktt::bridge::ktt_conv2 {
    namespace {

        constexpr uint32_t kTile = 2048;  // bf16 tile bytes
        constexpr uint32_t kPart = 4096;  // fp32 tile bytes
        // Circular-buffer bytes per core every ktt_conv2 program is sized against.
        uint32_t budget() {
            return program::cb_budget();
        }
        // Work items (column group, row tile range) one core takes at most.
        constexpr uint32_t kMaxItems = 48;

        // True unless KTT_CONV_WINDOW=0 (one read per tap row instead of shared
        // windows).
        bool window_enabled() {
            static const bool on = [] {
                const char* v = std::getenv("KTT_CONV_WINDOW");
                return !(v && std::string(v) == "0");
            }();
            return on;
        }

        // KTT_CONV2_BLOCK=n (tests, read per call): at most n taps per K block.
        uint32_t block_cap() {
            const char* v = std::getenv("KTT_CONV2_BLOCK");
            return v ? static_cast<uint32_t>(std::max(1, std::atoi(v))) : 0xFFFFFFFFu;
        }

        // KTT_CONV2_SPLIT=n (tests, read per call): one-phase gathers take the split
        // pipeline with n tap-row splits (n = 1: fp32 partials and their sum, one split).
        uint32_t split_force() {
            const char* v = std::getenv("KTT_CONV2_SPLIT");
            return v ? static_cast<uint32_t>(std::max(0, std::atoi(v))) : 0;
        }

        uint32_t out_tiles(const Geometry& g) {
            return (g.o + 31) / 32;
        }

        // Channel tiles a core with output column tiles [n0, n0 + nt) reads.
        uint32_t core_tiles(const Geometry& g, uint32_t n0, uint32_t nt) {
            uint32_t k0, k1, unused;
            channel_tiles(g, n0, k0, unused);
            channel_tiles(g, n0 + nt - 1, unused, k1);
            return k1 - k0;
        }

        // The most channel tiles any core with nt column tiles reads.
        uint32_t core_tiles_max(const Geometry& g, uint32_t nt) {
            const uint32_t ot = out_tiles(g);
            uint32_t most = 0;
            for (uint32_t n0 = 0; n0 < ot; n0 += nt) {
                most = std::max(most, core_tiles(g, n0, std::min(nt, ot - n0)));
            }
            return most;
        }

        // Rows of one tap window: Sw input rows per output position, over 32 positions
        // plus the spread of a phase's column offsets.
        uint32_t window_span(const Geometry& g) {
            uint32_t spread = 0;
            for (const auto& taps : g.w.taps) {
                if (taps.empty()) {
                    continue;
                }
                uint32_t lo = taps.front().first, hi = lo;
                for (const auto& [off, idx] : taps) {
                    lo = std::min(lo, off);
                    hi = std::max(hi, off);
                }
                spread = std::max(spread, hi - lo);
            }
            const uint32_t sw = g.w.stride;
            return (32 + spread / sw) * sw;
        }

        // Window sets in CB 3: 2 (the next row tile's windows read during this one's
        // taps), 1 when every item has one row tile (items keep at most half the cores).
        uint32_t window_sets(const Geometry& g, uint32_t nt) {
            const auto grid = detail::device()->compute_with_storage_grid_size();
            const uint32_t items = g.phases() * g.phase_tiles() * ((out_tiles(g) + nt - 1) / nt);
            return 2 * items <= grid.x * grid.y ? 1 : 2;
        }

        uint32_t window_bytes(const Geometry& g, uint32_t kct, uint32_t nt) {
            return window_sets(g, nt) * g.h.most_taps() * window_span(g) * kct * 64;
        }

        tt::DataFormat weight_format(tt::tt_metal::DataType dt) {
            using tt::tt_metal::DataType;
            return dt == DataType::BFLOAT8_B ? tt::DataFormat::Bfp8_b
                                             : tt::DataFormat::Float16_b;
        }

        // Runtime args of an axis: phases S, the S tap counts, then each phase's
        // (offset, kernel index) pairs.
        void push_axis(std::vector<uint32_t>& args, const Axis& a) {
            args.push_back(a.phases());
            for (const auto& taps : a.taps) {
                args.push_back(static_cast<uint32_t>(taps.size()));
            }
            for (const auto& taps : a.taps) {
                for (const auto& [off, idx] : taps) {
                    args.push_back(off);
                    args.push_back(idx);
                }
            }
        }

        void key_axis(program::Key& key, const Axis& a) {
            key.add(a.in).add(a.out).add(a.stride).add(a.phases());
            for (const auto& taps : a.taps) {
                key.add(static_cast<int64_t>(taps.size()));
                for (const auto& [off, idx] : taps) {
                    key.add(off).add(idx);
                }
            }
        }

        // Compile-time args `ct`, then the argument-layout version.
        std::vector<uint32_t> with_version(std::vector<uint32_t> ct) {
            ct.push_back(::ktt_conv::kArgsVersion);
            return ct;
        }

        // The argument-layout version, then common runtime args `args`.
        std::vector<uint32_t> versioned(const std::vector<uint32_t>& args) {
            std::vector<uint32_t> out{::ktt_conv::kArgsVersion};
            out.insert(out.end(), args.begin(), args.end());
            return out;
        }

        int64_t floor_div(int64_t a, int64_t b) {
            return a >= 0 ? a / b : -((-a + b - 1) / b);
        }

        // The input-gradient axis of a forward axis (input `in`, kernel k, stride s,
        // padding p, dilation d, output `out`): phase f holds the forward taps i with
        // (f + p - i d) mod s = 0 at offset (f + p - i d) / s + `front`; `front` (the
        // grad rows' padding in front) makes every offset non-negative.
        Axis grad_axis(
            int64_t in, int64_t k, int64_t s, int64_t p, int64_t d, int64_t out,
            uint32_t& front
        ) {
            Axis a;
            a.taps.assign(s, {});
            int64_t lo = 0, hi = 0;
            bool any = false;
            for (int64_t f = 0; f < s; ++f) {
                for (int64_t i = 0; i < k; ++i) {
                    const int64_t r = f + p - i * d;
                    if (((r % s) + s) % s != 0) {
                        continue;
                    }
                    const int64_t e = floor_div(r, s);
                    lo = any ? std::min(lo, e) : e;
                    hi = any ? std::max(hi, e) : e;
                    any = true;
                }
            }
            front = static_cast<uint32_t>(std::max<int64_t>(0, -lo));
            for (int64_t f = 0; f < s; ++f) {
                for (int64_t i = 0; i < k; ++i) {
                    const int64_t r = f + p - i * d;
                    if (((r % s) + s) % s == 0) {
                        a.taps[f].emplace_back(
                            static_cast<uint32_t>(floor_div(r, s) + front),
                            static_cast<uint32_t>(i)
                        );
                    }
                }
            }
            a.out = static_cast<uint32_t>((in + s - 1) / s);
            a.stride = 1;
            a.in = static_cast<uint32_t>(
                std::max<int64_t>(front + out, a.out - 1 + hi + front + 1)
            );
            return a;
        }

    }  // namespace

    uint32_t Axis::most_taps() const {
        size_t most = 0;
        for (const auto& t : taps) {
            most = std::max(most, t.size());
        }
        return static_cast<uint32_t>(most);
    }

    Geometry forward(
        const std::vector<int64_t>& x,
        const std::vector<int64_t>& w,
        std::array<int64_t, 2> stride,
        std::array<int64_t, 2> padding,
        std::array<int64_t, 2> dilation,
        int64_t groups,
        std::array<uint32_t, 2>& pad
    ) {
        Geometry g;
        g.n = static_cast<uint32_t>(x[0]);
        g.c = static_cast<uint32_t>(x[1]);
        g.cp = (g.c + 31) / 32 * 32;
        g.o = static_cast<uint32_t>(w[0]);
        g.kh = static_cast<uint32_t>(w[2]);
        g.kw = static_cast<uint32_t>(w[3]);
        g.groups = static_cast<uint32_t>(groups);
        Axis* axes[2] = {&g.h, &g.w};
        for (int a = 0; a < 2; ++a) {
            const int64_t k = w[2 + a], in = x[2 + a] + 2 * padding[a];
            Axis& axis = *axes[a];
            axis.in = static_cast<uint32_t>(in);
            axis.stride = static_cast<uint32_t>(stride[a]);
            const int64_t out = (in - dilation[a] * (k - 1) - 1) / stride[a] + 1;
            axis.out = static_cast<uint32_t>(std::max<int64_t>(out, 0));
            axis.taps.assign(1, {});
            for (int64_t i = 0; i < k; ++i) {
                axis.taps[0].emplace_back(
                    static_cast<uint32_t>(i * dilation[a]), static_cast<uint32_t>(i)
                );
            }
            pad[a] = static_cast<uint32_t>(padding[a]);
        }
        // Output rows over ceil(Wp / Sw) columns, so a tap window of an output row never
        // reaches the next one.
        g.pitch = (g.w.in + g.w.stride - 1) / g.w.stride;
        return g;
    }

    Geometry input_grad(
        const std::vector<int64_t>& x,
        const std::vector<int64_t>& w,
        std::array<int64_t, 2> stride,
        std::array<int64_t, 2> padding,
        std::array<int64_t, 2> dilation,
        int64_t groups,
        std::array<uint32_t, 2>& pad
    ) {
        Geometry g;
        g.n = static_cast<uint32_t>(x[0]);
        g.c = static_cast<uint32_t>(w[0]);
        g.cp = (g.c + 31) / 32 * 32;
        g.o = static_cast<uint32_t>(x[1]);
        g.kh = static_cast<uint32_t>(w[2]);
        g.kw = static_cast<uint32_t>(w[3]);
        g.groups = static_cast<uint32_t>(groups);
        Axis* axes[2] = {&g.h, &g.w};
        for (int a = 0; a < 2; ++a) {
            const int64_t k = w[2 + a], in = x[2 + a] + 2 * padding[a];
            const int64_t out = (in - dilation[a] * (k - 1) - 1) / stride[a] + 1;
            *axes[a] = grad_axis(
                x[2 + a], k, stride[a], padding[a], dilation[a], out, pad[a]
            );
        }
        g.pitch = g.w.in;
        return g;
    }

    bool packed(const Geometry& g) {
        return g.c / g.groups == 1;
    }

    void channel_tiles(const Geometry& g, uint32_t j, uint32_t& k0, uint32_t& k1) {
        const uint32_t cg = g.c / g.groups, og = g.o / g.groups;
        const uint32_t lo = 32 * j, hi = std::min(32 * j + 31, g.o - 1);
        k0 = (lo / og) * cg / 32;
        k1 = ((hi / og + 1) * cg - 1) / 32 + 1;
    }

    uint32_t channel_tiles_max(const Geometry& g) {
        uint32_t most = 0;
        for (uint32_t j = 0; j < out_tiles(g); ++j) {
            uint32_t k0, k1;
            channel_tiles(g, j, k0, k1);
            most = std::max(most, k1 - k0);
        }
        return most;
    }

    // Input rows per output position in a tap's rows: Sw in window mode, else 1.
    static uint32_t tap_rows(const Geometry& g, const Plan& p) {
        return p.window ? g.w.stride : 1;
    }

    uint32_t cb_bytes(const Geometry& g, const Plan& p, uint32_t weight_tile) {
        const uint32_t kct = core_tiles_max(g, p.nt);
        const uint32_t blocks = (g.most_taps() + p.block - 1) / p.block;
        // Packed weights: bf16 tiles, a zero tile and one packed row of scratch.
        const bool pack = packed(g);
        const uint32_t tile_bytes = pack ? kTile : weight_tile;
        const uint32_t scratch = pack ? kTile + g.taps() * 64 * (g.c / g.groups) : 0;
        const uint32_t weights =
            p.nt * g.taps() * channel_tiles_max(g) * tile_bytes + scratch;
        const uint32_t rows = 2 * p.group * kct * kTile;
        const uint32_t act = p.block * kct * kTile;
        // Output tiles and the writer's zero tile: fp32 partial sums when split.
        const uint32_t out_tile = (p.split > 1 || split_force() > 0) ? kPart : kTile;
        const uint32_t out = (2 * p.nt + 1) * out_tile;
        const uint32_t part = blocks > 1 ? p.nt * kPart : 0;
        return weights + rows + act + out + part +
            (p.window ? window_bytes(g, kct, p.nt) : 0);
    }

    std::optional<Plan> plan(const Geometry& g, uint32_t weight_tile) {
        const uint32_t ot = out_tiles(g);
        const uint32_t most = std::min(std::max<uint32_t>(g.most_taps(), 1), block_cap());
        if (g.h.phases() > kMaxPhases || g.w.phases() > kMaxPhases ||
            g.kh > kMaxTaps || g.kw > kMaxTaps || g.kh > 32 || g.kw > 32) {
            return std::nullopt;
        }
        const auto grid = detail::device()->compute_with_storage_grid_size();
        const uint32_t cores = grid.x * grid.y;
        const uint32_t mt = g.phases() * g.phase_tiles();
        // Then the tap window and, for one-phase gathers, one read barrier per tap row,
        // each when it fits. The window at column stride Sw > 1 (Sw rows per output
        // position) is used when it reads fewer rows than the taps do, or whole rows.
        const auto with_window = [&](Plan p) {
            const uint32_t sw = g.w.stride;
            bool whole = true;
            for (uint32_t n0 = 0; n0 < ot; n0 += p.nt) {
                whole = whole && core_tiles(g, n0, std::min(p.nt, ot - n0)) * 32 == g.cp;
            }
            const bool fewer = window_span(g) < 32 * g.w.most_taps();
            p.window = window_enabled() && (sw == 1 ? g.kw > 1 : whole || fewer);
            if (p.window && cb_bytes(g, p, weight_tile) > budget()) {
                p.window = false;
            }
            if (g.phases() == 1) {
                p.group = g.w.most_taps();
                if (cb_bytes(g, p, weight_tile) > budget()) {
                    p.group = 1;
                }
            }
            // The tap-row split only when forced (KTT_CONV2_SPLIT): its sum program and fp32
            // partials cost more host time per call than the split saves on the device.
            if (g.phases() == 1 && split_force() > 0) {
                // fp32 output tiles: smaller K blocks make room when needed.
                Plan q = p;
                q.split = std::min(split_force(), g.h.most_taps());
                while (cb_bytes(g, q, weight_tile) > budget() && q.block > 1) {
                    --q.block;
                }
                if (cb_bytes(g, q, weight_tile) <= budget()) {
                    p = q;
                }
            }
            return p;
        };
        // All taps in one K block, the widest column group first (all of Ot when it is
        // at most 4: whole input rows); else one column tile per core and the most taps
        // per block that fit. One column tile per core is taken instead when the wider
        // group keeps at most half the cores busy and its column tiles read disjoint
        // channel tiles (nothing shared).
        const auto shares = [&](uint32_t nt) {
            for (uint32_t n0 = 0; n0 < ot; n0 += nt) {
                const uint32_t n = std::min(nt, ot - n0);
                uint32_t each = 0;
                for (uint32_t j = n0; j < n0 + n; ++j) {
                    uint32_t k0, k1;
                    channel_tiles(g, j, k0, k1);
                    each += k1 - k0;
                }
                if (each > core_tiles(g, n0, n)) {
                    return true;
                }
            }
            return false;
        };
        // One-phase gathers, Ot <= 4, strided or with all taps too large for one block:
        // whole input rows (nt = Ot) in K blocks when the window fits, before narrower
        // column groups (whose windows read every row separately).
        if (g.phases() == 1 && ot <= 4 &&
            (g.w.stride > 1 || cb_bytes(g, Plan{ot, most, false}, weight_tile) > budget())) {
            for (uint32_t block = most; block >= 1; --block) {
                const Plan q = with_window(Plan{ot, block, false});
                if (q.window) {
                    return q;
                }
            }
        }
        for (uint32_t nt : {std::min(4u, ot), 2u, 1u}) {
            if (nt > ot) {
                continue;
            }
            const Plan p{nt, most, false};
            if (cb_bytes(g, p, weight_tile) > budget()) {
                continue;
            }
            // Strided one-phase gathers keep whole rows (nt = Ot) for the window's runs.
            const bool idle = 2 * mt * ((ot + nt - 1) / nt) <= cores;
            const bool whole = nt == ot && g.w.stride > 1 && g.phases() == 1;
            if (nt > 1 && idle && !shares(nt) && !whole) {
                const Plan narrow{1, most, false};
                if (cb_bytes(g, narrow, weight_tile) <= budget()) {
                    return with_window(narrow);
                }
            }
            return with_window(p);
        }
        for (uint32_t block = most - 1; block >= 1; --block) {
            const Plan p{1, block, false};
            if (cb_bytes(g, p, weight_tile) <= budget()) {
                return with_window(p);
            }
        }
        return std::nullopt;
    }

    namespace {

        tt::tt_metal::CBDescriptor cb(
            uint8_t index,
            uint32_t pages,
            uint32_t page,
            tt::DataFormat df,
            const tt::tt_metal::CoreRangeSet& range
        ) {
            return tt::tt_metal::CBDescriptor{
                .total_size = pages * page,
                .core_ranges = range,
                .format_descriptors = {tt::tt_metal::CBFormatDescriptor{
                    .buffer_index = index, .data_format = df, .page_size = page
                }}
            };
        }

        std::pair<uint32_t, tt::tt_metal::CoreRangeSet> first_cores(uint32_t work);

        // Elementwise sums over `chunks` of `pages` blocks of 1024 fp32 values (chunk s's
        // block b at page s pages + b of `partials`) into page b of `out` (bf16 or fp32,
        // one tile or 1024 values per page): SFPU fp32 adds of the blocks unpacked to DST
        // as fp32. Returns `out`.
        ttnn::Tensor chunk_sum(
            const ttnn::Tensor& partials,
            const ttnn::Tensor& out,
            uint32_t chunks,
            uint32_t pages
        ) {
            using namespace tt::tt_metal;
            const auto out_df = out.dtype() == DataType::FLOAT32 ? tt::DataFormat::Float32
                                                                 : tt::DataFormat::Float16_b;
            program::mark(program::Phase::Alloc);
            program::Key key("ktt_conv2_chunk_sum");
            key.add(partials).add(out).add(chunks).add(pages);
            return program::launch(key, {partials, out}, [&] {
                const auto [cores, range] = first_cores(pages);
                ProgramDescriptor desc;
                desc.cbs.push_back(cb(0, 2, kPart, tt::DataFormat::Float32, range));
                desc.cbs.push_back(cb(16, 2, tt::tile_size(out_df), out_df, range));
                const auto args = program::core_index_args(cores);
                program::mark(program::Phase::Args);
                desc.kernels.push_back(KernelDescriptor{
                    .kernel_source = program::source("nn/ktt_conv2/reduce_reader.cpp"),
                    .core_ranges = range,
                    .compile_time_args = with_version(
                        TensorAccessorArgs(*partials.buffer()).get_compile_time_args()
                    ),
                    .runtime_args = args,
                    .common_runtime_args =
                        versioned({partials.buffer()->address(), cores, pages, chunks}),
                    .config = ReaderConfigDescriptor{}
                });
                desc.kernels.push_back(KernelDescriptor{
                    .kernel_source = program::source("nn/ktt_conv2/reduce_writer.cpp"),
                    .core_ranges = range,
                    .compile_time_args = with_version(
                        TensorAccessorArgs(*out.buffer()).get_compile_time_args()
                    ),
                    .runtime_args = args,
                    .common_runtime_args = versioned({out.buffer()->address(), cores, pages}),
                    .config = WriterConfigDescriptor{}
                });
                ComputeConfigDescriptor::UnpackToDestModes unpack(
                    NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default
                );
                unpack[0] = UnpackToDestMode::UnpackToDestFp32;
                desc.kernels.push_back(KernelDescriptor{
                    .kernel_source = program::source("nn/ktt_conv2/reduce_compute.cpp"),
                    .core_ranges = range,
                    .compile_time_args = with_version({}),
                    .runtime_args = args,
                    .common_runtime_args = versioned({cores, pages, chunks}),
                    .config = ComputeConfigDescriptor{
                        .math_fidelity = MathFidelity::HiFi4,
                        .fp32_dest_acc_en = true,
                        .unpack_to_dest_mode = unpack,
                        .math_approx_mode = false
                    }
                });
                constexpr auto all = program::Slot::kCommon;
                return program::Built{
                    std::move(desc),
                    {
                        {.kernel = 0, .entry = all, .position = 1},
                        {.kernel = 1, .entry = all, .position = 1, .tensor = 1}
                    }
                };
            });
        }

        // The first `work` cores of the grid (at least one, at most all).
        std::pair<uint32_t, tt::tt_metal::CoreRangeSet> first_cores(uint32_t work) {
            const auto grid = detail::device()->compute_with_storage_grid_size();
            const uint32_t cores = std::max<uint32_t>(
                1, std::min<uint32_t>(grid.x * grid.y, work)
            );
            std::vector<tt::tt_metal::CoreRange> ranges;
            for (uint32_t c = 0; c < cores; ++c) {
                ranges.emplace_back(tt::tt_metal::CoreCoord(c % grid.x, c / grid.x));
            }
            return {cores, tt::tt_metal::CoreRangeSet(ranges).merge_ranges()};
        }

    }  // namespace

    ttnn::Tensor prepare(
        const ttnn::Tensor& weight, const Geometry& g, bool transposed, bool one_d
    ) {
        using namespace tt::tt_metal;
        const uint32_t ot = out_tiles(g), kmax = channel_tiles_max(g);
        const uint32_t halves = g.kw > 16 ? 2 : 1;
        if (packed(g)) {
            const uint32_t cg = g.c / g.groups;
            auto rows = ttnn::empty(
                ttnn::Shape({ot, g.taps() * 32 * cg}),
                DataType::BFLOAT16,
                Layout::ROW_MAJOR,
                detail::device(),
                MemoryConfig{}
            );
            program::mark(program::Phase::Alloc);
            program::Key key("ktt_conv2_prep_packed");
            key.add(weight).add(rows);
            for (const int64_t v : {g.c, g.o, g.kh, g.kw, g.groups}) {
                key.add(v);
            }
            key.add(static_cast<int64_t>(transposed)).add(static_cast<int64_t>(one_d));
            return program::launch(key, {weight, rows}, [&] {
                const auto [cores, range] = first_cores(ot * g.kh);
                ProgramDescriptor desc;
                const auto bf16 = tt::DataFormat::Float16_b;
                desc.cbs.push_back(cb(0, 1, 32 * cg * halves * 64 + 64, bf16, range));
                desc.cbs.push_back(cb(1, 1, g.kw * 32 * cg * 2, bf16, range));
                program::mark(program::Phase::Args);
                auto ct = TensorAccessorArgs(*weight.buffer()).get_compile_time_args();
                const auto more = TensorAccessorArgs(*rows.buffer()).get_compile_time_args();
                ct.insert(ct.end(), more.begin(), more.end());
                desc.kernels.push_back(
                    KernelDescriptor{
                        .kernel_source = program::source("nn/ktt_conv2/prep_packed.cpp"),
                        .core_ranges = range,
                        .compile_time_args = with_version(ct),
                        .runtime_args = program::core_index_args(cores),
                        .common_runtime_args = versioned({weight.buffer()->address(),
                             rows.buffer()->address(),
                             cores,
                             ot,
                             g.kh,
                             g.kw,
                             cg,
                             g.o / g.groups,
                             g.o,
                             static_cast<uint32_t>(transposed),
                             static_cast<uint32_t>(one_d)}),
                        .config = ReaderConfigDescriptor{}
                    }
                );
                constexpr auto all = program::Slot::kCommon;
                return program::Built{
                    std::move(desc),
                    {
                        {.kernel = 0, .entry = all, .position = 1},
                        {.kernel = 0, .entry = all, .position = 2, .tensor = 1}
                    }
                };
            });
        }
        const uint32_t units = ot * g.kh * kmax;
        auto out = ttnn::empty(
            ttnn::Shape({ot * g.taps() * kmax * 32, 32}),
            DataType::BFLOAT16,
            Layout::TILE,
            detail::device(),
            MemoryConfig{}
        );
        program::mark(program::Phase::Alloc);
        program::Key key("ktt_conv2_prep");
        key.add(weight).add(out);
        for (const int64_t v : {g.c, g.o, g.kh, g.kw, g.groups}) {
            key.add(v);
        }
        key.add(static_cast<int64_t>(transposed)).add(static_cast<int64_t>(one_d));
        return program::launch(key, {weight, out}, [&] {
            const auto [cores, range] = first_cores(units);
            ProgramDescriptor desc;
            desc.cbs.push_back(
                cb(0, 1, 1024 * halves * 64 + 64, tt::DataFormat::Float16_b, range)
            );
            desc.cbs.push_back(cb(1, 1, g.kw * kTile, tt::DataFormat::Float16_b, range));
            program::mark(program::Phase::Args);
            auto ct = TensorAccessorArgs(*weight.buffer()).get_compile_time_args();
            const auto more = TensorAccessorArgs(*out.buffer()).get_compile_time_args();
            ct.insert(ct.end(), more.begin(), more.end());
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("nn/ktt_conv2/prep.cpp"),
                    .core_ranges = range,
                    .compile_time_args = with_version(ct),
                    .runtime_args = program::core_index_args(cores),
                    .common_runtime_args = versioned({weight.buffer()->address(),
                         out.buffer()->address(),
                         cores,
                         ot,
                         g.kh,
                         g.kw,
                         kmax,
                         g.c / g.groups,
                         g.o / g.groups,
                         g.o,
                         g.c,
                         static_cast<uint32_t>(transposed),
                         static_cast<uint32_t>(one_d)}),
                    .config = ReaderConfigDescriptor{}
                }
            );
            constexpr auto all = program::Slot::kCommon;
            return program::Built{
                std::move(desc),
                {
                    {.kernel = 0, .entry = all, .position = 1},
                    {.kernel = 0, .entry = all, .position = 2, .tensor = 1}
                }
            };
        });
    }

    ttnn::Tensor run(
        const ttnn::Tensor& x_rows,
        const ttnn::Tensor& w,
        const Geometry& g,
        const ttnn::ComputeKernelConfig& config
    ) {
        using namespace tt::tt_metal;
        const uint32_t ot = out_tiles(g), kmax = channel_tiles_max(g);
        const auto wf = weight_format(w.dtype());
        const uint32_t weight_tile = tt::tile_size(wf);
        const auto chosen = plan(g, weight_tile);
        if (!chosen) {
            throw std::invalid_argument(
                "ktt_conv2: one column tile's weights exceed the circular-buffer budget"
            );
        }
        const Plan p = *chosen;
        const uint32_t mt = g.phases() * g.phase_tiles();
        const uint32_t in_rows = g.n * g.h.in * g.w.in;
        const uint32_t nh = g.h.most_taps();

        auto out = ttnn::empty(
            ttnn::Shape({g.out_rows(), g.o}),
            DataType::BFLOAT16,
            Layout::TILE,
            detail::device(),
            MemoryConfig{}
        );
        // A split gather writes fp32 partial sums, split row-tile blocks of mt tiles.
        const bool split = p.split > 1 || (g.phases() == 1 && split_force() > 0);
        auto target = split ? ttnn::empty(
                                  ttnn::Shape({p.split * g.out_rows(), g.o}),
                                  DataType::FLOAT32,
                                  Layout::TILE,
                                  detail::device(),
                                  MemoryConfig{}
                              )
                            : out;
        program::mark(program::Phase::Alloc);
        program::Key key("ktt_conv2");
        key.add(x_rows).add(w).add(target).add(static_cast<int64_t>(config.math_fidelity));
        key.add(p.nt).add(p.block).add(static_cast<int64_t>(p.window)).add(p.group);
        key.add(p.split);
        for (const int64_t v : {g.n, g.c, g.o, g.kh, g.kw, g.groups, g.pitch}) {
            key.add(v);
        }
        key_axis(key, g.h);
        key_axis(key, g.w);
        target = program::launch(key, {x_rows, w, target}, [&] {
            const auto grid = detail::device()->compute_with_storage_grid_size();
            const uint32_t cores = grid.x * grid.y;
            const uint32_t col_groups = (ot + p.nt - 1) / p.nt;
            const uint32_t per_group = std::max<uint32_t>(
                1,
                std::min<uint32_t>(
                    cores / std::max<uint32_t>(col_groups * p.split, 1), mt
                )
            );
            // Work items (m0, m1, n0, nt, first tap row, end tap row, row-tile base),
            // dealt to the cores round-robin.
            std::vector<std::vector<uint32_t>> items(cores);
            uint32_t next = 0;
            for (uint32_t gi = 0; gi < col_groups; ++gi) {
                const uint32_t n0 = gi * p.nt, n = std::min(p.nt, ot - n0);
                for (uint32_t q = 0; q < per_group; ++q) {
                    const uint32_t m0 = mt * q / per_group, m1 = mt * (q + 1) / per_group;
                    if (m1 == m0) {
                        continue;
                    }
                    for (uint32_t s = 0; s < p.split; ++s) {
                        const uint32_t a0 = nh * s / p.split, a1 = nh * (s + 1) / p.split;
                        auto& list = items[next % cores];
                        list.insert(list.end(), {m0, m1, n0, n, a0, a1, s * mt});
                        ++next;
                    }
                }
            }
            if ((next + cores - 1) / cores > kMaxItems) {
                throw std::invalid_argument("ktt_conv2: too many output column tiles");
            }
            const uint32_t used = std::min(next, cores);
            std::vector<CoreRange> ranges;
            KernelDescriptor::RuntimeArgs args;
            for (uint32_t c = 0; c < used; ++c) {
                const CoreCoord core(c % grid.x, c / grid.x);
                ranges.emplace_back(core, core);
                std::vector<uint32_t> a{static_cast<uint32_t>(items[c].size() / 7)};
                a.insert(a.end(), items[c].begin(), items[c].end());
                args.push_back({core, a});
            }
            const CoreRangeSet range = CoreRangeSet(ranges).merge_ranges();

            ProgramDescriptor desc;
            const auto bf16 = tt::DataFormat::Float16_b;
            const auto out_df = split ? tt::DataFormat::Float32 : bf16;
            const uint32_t out_tile = split ? kPart : kTile;
            const uint32_t kct = core_tiles_max(g, p.nt);
            const uint32_t blocks = (g.most_taps() + p.block - 1) / p.block;
            const uint32_t per = tap_rows(g, p);
            desc.cbs.push_back(cb(0, 2 * p.group * kct, kTile, bf16, range));
            desc.cbs.push_back(cb(1, p.nt * g.taps() * kmax, weight_tile, wf, range));
            desc.cbs.push_back(cb(2, p.block * kct, kTile, bf16, range));
            desc.cbs.push_back(cb(4, 1, out_tile, out_df, range));
            desc.cbs.push_back(cb(16, 2 * p.nt, out_tile, out_df, range));
            if (blocks > 1) {
                desc.cbs.push_back(cb(24, p.nt, kPart, tt::DataFormat::Float32, range));
            }
            if (p.window) {
                // One page per tap row: span rows of the item's K tiles.
                const uint32_t page = window_span(g) * kct * 64;
                const uint32_t pages = window_sets(g, p.nt) * g.h.most_taps();
                desc.cbs.push_back(cb(3, pages, page, bf16, range));
            }
            const bool pack = packed(g);
            if (pack) {
                desc.cbs.push_back(cb(5, 1, kTile, bf16, range));
                desc.cbs.push_back(cb(6, 1, g.taps() * 64 * (g.c / g.groups), bf16, range));
            }

            const uint32_t row_bytes = g.cp * 2;
            const uint32_t per_page = x_rows.buffer()->page_size() / row_bytes;
            const uint32_t cg = g.c / g.groups, og = g.o / g.groups;
            program::mark(program::Phase::Args);

            // Shared by reader, compute and writer: the output split and the taps.
            std::vector<uint32_t> shape{
                ot, kmax, cg, og, g.o, g.kh, g.kw, kct, p.nt, g.phase_tiles(),
                g.w.phases(), g.h.most_taps()
            };
            push_axis(shape, g.h);
            push_axis(shape, g.w);
            std::vector<uint32_t> reader_args{
                x_rows.buffer()->address(),
                w.buffer()->address(),
                g.w.in,
                g.h.in * g.w.in,
                in_rows,
                row_bytes,
                per_page,
                static_cast<uint32_t>(p.window),
                g.n * g.h.out * g.pitch,
                g.h.out * g.pitch,
                g.pitch,
                g.h.stride,
                g.w.stride,
                window_span(g),
                static_cast<uint32_t>(pack),
                p.group
            };
            reader_args.insert(reader_args.end(), shape.begin(), shape.end());
            std::vector<uint32_t> compute_args{
                p.block, per, static_cast<uint32_t>(p.window), window_span(g)
            };
            compute_args.insert(compute_args.end(), shape.begin(), shape.end());
            std::vector<uint32_t> writer_args{
                target.buffer()->address(), w.buffer()->address(), static_cast<uint32_t>(pack)
            };
            writer_args.insert(writer_args.end(), shape.begin(), shape.end());

            auto reader_ct = TensorAccessorArgs(*x_rows.buffer()).get_compile_time_args();
            const auto w_ct = TensorAccessorArgs(*w.buffer()).get_compile_time_args();
            reader_ct.insert(reader_ct.end(), w_ct.begin(), w_ct.end());
            auto writer_ct = TensorAccessorArgs(*target.buffer()).get_compile_time_args();
            writer_ct.insert(writer_ct.end(), w_ct.begin(), w_ct.end());
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("nn/ktt_conv2/reader.cpp"),
                    .core_ranges = range,
                    .compile_time_args = with_version(reader_ct),
                    .runtime_args = args,
                    .common_runtime_args = versioned(reader_args),
                    .config = ReaderConfigDescriptor{}
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("nn/ktt_conv2/writer.cpp"),
                    .core_ranges = range,
                    .compile_time_args = with_version(writer_ct),
                    .runtime_args = args,
                    .common_runtime_args = versioned(writer_args),
                    .config = WriterConfigDescriptor{}
                }
            );
            // K-block sums (CB 24) unpack straight into DST as fp32.
            ComputeConfigDescriptor::UnpackToDestModes unpack(
                NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default
            );
            if (blocks > 1) {
                unpack[24] = UnpackToDestMode::UnpackToDestFp32;
            }
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("nn/ktt_conv2/compute.cpp"),
                    .core_ranges = range,
                    .compile_time_args = with_version({}),
                    .runtime_args = args,
                    .common_runtime_args = versioned(compute_args),
                    .config = ComputeConfigDescriptor{
                        .math_fidelity = config.math_fidelity,
                        .fp32_dest_acc_en = true,
                        .unpack_to_dest_mode = unpack,
                        .math_approx_mode = false
                    }
                }
            );
            constexpr auto all = program::Slot::kCommon;
            return program::Built{
                std::move(desc),
                {
                    {.kernel = 0, .entry = all, .position = 1},
                    {.kernel = 0, .entry = all, .position = 2, .tensor = 1},
                    {.kernel = 1, .entry = all, .position = 1, .tensor = 2},
                    {.kernel = 1, .entry = all, .position = 2, .tensor = 1}
                }
            };
        });
        if (!split) {
            return target;
        }
        // The splits' partial tiles summed into the bf16 output.
        return chunk_sum(target, out, p.split, mt * ot);
    }

    namespace {

        // Weight-gradient circular-buffer bytes with nt column tiles and per_chunk row
        // tiles per chunk.
        uint32_t weight_grad_bytes(
            const Geometry& g, uint32_t nt, uint32_t per_chunk, bool window
        ) {
            const uint32_t kct = core_tiles_max(g, nt);
            const uint32_t rows = 2 * (kct + nt) * kTile;
            const uint32_t grad = (nt + per_chunk * nt) * kTile;
            const uint32_t act = per_chunk * kct * kTile;
            const uint32_t pack = packed(g) ? 32 * (g.c / g.groups) * 4 : 0;
            const uint32_t windows = window ? per_chunk * window_span(g) * kct * 64 : 0;
            return rows + grad + act + 2 * kPart + pack + windows;
        }

    }  // namespace

    std::optional<WeightGradPlan> weight_grad_plan(const Geometry& g) {
        if (g.phases() != 1 || g.kh > kMaxTaps || g.kw > kMaxTaps) {
            return std::nullopt;
        }
        const auto grid = detail::device()->compute_with_storage_grid_size();
        const uint32_t cores = grid.x * grid.y;
        const uint32_t ot = out_tiles(g), mt = g.phase_tiles();
        for (uint32_t nt : {std::min(4u, ot), 2u, 1u}) {
            if (nt > ot) {
                continue;
            }
            const uint32_t col_groups = (ot + nt - 1) / nt;
            // Chunks: enough units for the grid, at least `least` row tiles each (a
            // chunk writes one fp32 partial per weight tile); each chunk's tiles
            // resident in L1.
            const bool pack = packed(g);
            const uint32_t least = std::min<uint32_t>(pack ? 1 : 4, mt);
            uint32_t chunks = std::clamp<uint32_t>(cores / col_groups, 1, mt / least);
            uint32_t per_chunk = (mt + chunks - 1) / chunks;
            while (per_chunk > 1 && weight_grad_bytes(g, nt, per_chunk, false) > budget()) {
                per_chunk = (per_chunk + 1) / 2;
            }
            if (weight_grad_bytes(g, nt, per_chunk, false) > budget()) {
                continue;
            }
            chunks = (mt + per_chunk - 1) / per_chunk;
            const bool window =
                (window_enabled() && g.w.stride == 1 && g.kw > 1 &&
                 weight_grad_bytes(g, nt, per_chunk, true) <= budget());
            return WeightGradPlan{nt, per_chunk, chunks, window};
        }
        return std::nullopt;
    }

    ttnn::Tensor weight_grad(
        const ttnn::Tensor& x_rows,
        const ttnn::Tensor& g_rows,
        const Geometry& g,
        const ttnn::ComputeKernelConfig& config,
        bool one_d
    ) {
        using namespace tt::tt_metal;
        const auto chosen = weight_grad_plan(g);
        if (!chosen) {
            throw std::invalid_argument("ktt_conv2: weight gradient exceeds the budget");
        }
        const auto [nt, per_chunk, chunks, window] = *chosen;
        const uint32_t ot = out_tiles(g), kmax = channel_tiles_max(g);
        const uint32_t tiles = ot * g.taps() * kmax;
        const uint32_t mt = g.phase_tiles();
        const uint32_t col_groups = (ot + nt - 1) / nt;
        const uint32_t units = col_groups * chunks;
        const uint32_t cg = g.c / g.groups, og = g.o / g.groups;
        const uint32_t kct = core_tiles_max(g, nt);
        const auto bf16 = tt::DataFormat::Float16_b, fp32 = tt::DataFormat::Float32;
        const bool pack = packed(g);
        const uint32_t entries = 32 * cg;

        // Partials: per chunk B pages of 1024 fp32, holding one tile per weight tile, or
        // packed one row of 32 Cg entries per (column tile, tap).
        const uint32_t pages = pack ? (ot * g.taps() * entries + 1023) / 1024 : tiles;
        auto partials = ttnn::empty(
            ttnn::Shape({chunks * pages, 1024}),
            DataType::FLOAT32,
            Layout::ROW_MAJOR,
            detail::device(),
            MemoryConfig{}
        );
        program::mark(program::Phase::Alloc);
        program::Key key("ktt_conv2_wgrad");
        key.add(x_rows).add(g_rows).add(partials);
        key.add(nt).add(per_chunk).add(chunks).add(static_cast<int64_t>(window));
        key.add(static_cast<int64_t>(config.math_fidelity));
        for (const int64_t v : {g.n, g.c, g.o, g.kh, g.kw, g.groups, g.pitch}) {
            key.add(v);
        }
        key_axis(key, g.h);
        key_axis(key, g.w);
        partials = program::launch(key, {x_rows, g_rows, partials}, [&] {
            const auto [cores, range] = first_cores(units);
            ProgramDescriptor desc;
            desc.cbs.push_back(cb(0, 2 * kct, kTile, bf16, range));
            desc.cbs.push_back(cb(1, 2 * nt, kTile, bf16, range));
            desc.cbs.push_back(cb(2, per_chunk * kct, kTile, bf16, range));
            desc.cbs.push_back(cb(4, nt, kTile, bf16, range));
            desc.cbs.push_back(cb(5, per_chunk * nt, kTile, bf16, range));
            desc.cbs.push_back(cb(16, 2, kPart, fp32, range));
            if (pack) {
                desc.cbs.push_back(cb(6, 1, entries * 4, fp32, range));
            }
            if (window) {
                const uint32_t bytes = per_chunk * window_span(g) * kct * 64;
                desc.cbs.push_back(cb(3, 1, bytes, bf16, range));
            }
            const uint32_t x_row_bytes = g.cp * 2;
            const uint32_t x_per_page = x_rows.buffer()->page_size() / x_row_bytes;
            const uint32_t g_row_bytes = (g.o + 31) / 32 * 64;
            const uint32_t g_per_page = g_rows.buffer()->page_size() / g_row_bytes;
            const auto args = program::core_index_args(cores);
            program::mark(program::Phase::Args);
            std::vector<uint32_t> shape{
                cores, units, chunks, per_chunk, mt, ot, kmax, cg, og, g.o, g.kh, g.kw,
                kct, nt
            };
            std::vector<uint32_t> reader_args{
                x_rows.buffer()->address(),
                g_rows.buffer()->address(),
                g.w.in,
                g.h.in * g.w.in,
                g.n * g.h.in * g.w.in,
                x_row_bytes,
                x_per_page,
                g_row_bytes,
                g_per_page,
                g.n * g.h.out * g.pitch,
                g.h.out * g.pitch,
                g.pitch,
                g.h.stride,
                g.w.stride,
                static_cast<uint32_t>(window),
                window_span(g)
            };
            reader_args.insert(reader_args.end(), shape.begin(), shape.end());
            push_axis(reader_args, g.h);
            push_axis(reader_args, g.w);
            std::vector<uint32_t> writer_args{partials.buffer()->address()};
            writer_args.insert(writer_args.end(), shape.begin(), shape.end());
            writer_args.push_back(static_cast<uint32_t>(pack));
            writer_args.push_back(pages);
            auto reader_ct = TensorAccessorArgs(*x_rows.buffer()).get_compile_time_args();
            const auto more = TensorAccessorArgs(*g_rows.buffer()).get_compile_time_args();
            reader_ct.insert(reader_ct.end(), more.begin(), more.end());
            desc.kernels.push_back(KernelDescriptor{
                .kernel_source = program::source("nn/ktt_conv2/wgrad_reader.cpp"),
                .core_ranges = range,
                .compile_time_args = with_version(reader_ct),
                .runtime_args = args,
                .common_runtime_args = versioned(reader_args),
                .config = ReaderConfigDescriptor{}
            });
            desc.kernels.push_back(KernelDescriptor{
                .kernel_source = program::source("nn/ktt_conv2/wgrad_writer.cpp"),
                .core_ranges = range,
                .compile_time_args = with_version(TensorAccessorArgs(*partials.buffer()).get_compile_time_args()),
                .runtime_args = args,
                .common_runtime_args = versioned(writer_args),
                .config = WriterConfigDescriptor{}
            });
            desc.kernels.push_back(KernelDescriptor{
                .kernel_source = program::source("nn/ktt_conv2/wgrad_compute.cpp"),
                .core_ranges = range,
                .compile_time_args = with_version({}),
                .runtime_args = args,
                .common_runtime_args = versioned(shape),
                .config = ComputeConfigDescriptor{
                    .math_fidelity = config.math_fidelity,
                    .fp32_dest_acc_en = true,
                    .math_approx_mode = false
                }
            });
            constexpr auto all = program::Slot::kCommon;
            return program::Built{
                std::move(desc),
                {
                    {.kernel = 0, .entry = all, .position = 1},
                    {.kernel = 0, .entry = all, .position = 2, .tensor = 1},
                    {.kernel = 1, .entry = all, .position = 1, .tensor = 2}
                }
            };
        });

        // The chunk sums: [B, 1024] fp32, summed on the SFPU.
        auto sums = chunk_sum(
            partials,
            ttnn::empty(
                ttnn::Shape({pages, 1024}),
                DataType::FLOAT32,
                Layout::ROW_MAJOR,
                detail::device(),
                MemoryConfig{}
            ),
            chunks,
            pages
        );

        auto grad = ttnn::empty(
            one_d ? ttnn::Shape({g.o, cg, g.kw}) : ttnn::Shape({g.o, cg, g.kh, g.kw}),
            DataType::BFLOAT16,
            Layout::TILE,
            detail::device(),
            MemoryConfig{}
        );
        // Output tiles: one per (o, q), or per (o, 32 q) in 1-D.
        const uint32_t out_count = one_d ? g.o * ((cg + 31) / 32) : g.o * cg;
        program::mark(program::Phase::Alloc);
        program::Key skey("ktt_conv2_wgrad_scatter");
        skey.add(sums).add(grad);
        for (const int64_t v : {g.c, g.o, g.kh, g.kw, g.groups}) {
            skey.add(v);
        }
        return program::launch(skey, {sums, grad}, [&] {
            const auto [cores, range] = first_cores(out_count);
            ProgramDescriptor desc;
            desc.cbs.push_back(cb(0, 1, g.taps() * 64 + 64, fp32, range));
            desc.cbs.push_back(cb(1, 1, kTile, bf16, range));
            program::mark(program::Phase::Args);
            auto ct = TensorAccessorArgs(*sums.buffer()).get_compile_time_args();
            const auto more = TensorAccessorArgs(*grad.buffer()).get_compile_time_args();
            ct.insert(ct.end(), more.begin(), more.end());
            desc.kernels.push_back(KernelDescriptor{
                .kernel_source = program::source("nn/ktt_conv2/wgrad_scatter.cpp"),
                .core_ranges = range,
                .compile_time_args = with_version(ct),
                .runtime_args = program::core_index_args(cores),
                .common_runtime_args = versioned({sums.buffer()->address(),
                     grad.buffer()->address(),
                     cores,
                     static_cast<uint32_t>(pack),
                     static_cast<uint32_t>(one_d),
                     g.kh,
                     g.kw,
                     cg,
                     og,
                     g.o,
                     kmax}),
                .config = ReaderConfigDescriptor{}
            });
            constexpr auto all = program::Slot::kCommon;
            return program::Built{
                std::move(desc),
                {
                    {.kernel = 0, .entry = all, .position = 1},
                    {.kernel = 0, .entry = all, .position = 2, .tensor = 1}
                }
            };
        });
    }

}  // namespace ktt::bridge::ktt_conv2
