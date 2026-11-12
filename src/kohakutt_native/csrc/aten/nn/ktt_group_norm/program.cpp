// Host side of ktt_group_norm: planes split over the grid (strided), circular buffers,
// runtime args of the moments and affine programs.
#include "aten/nn/ktt_group_norm/program.hpp"

#include "aten/nn/ktt_group_norm/rows.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <initializer_list>
#include <stdexcept>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "core/device/handle.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/full/full.hpp"
#include "ttnn/operations/generic/generic_op.hpp"

namespace ktt::bridge::ktt_group_norm {
    namespace {

        using namespace tt::tt_metal;

        struct Kernel {
            std::string source;
            std::vector<const ttnn::Tensor*> accessors;
            std::vector<uint32_t> head;  // io tensors whose addresses lead common args
            std::vector<uint32_t> tail;  // common args after the core count
            int kind;                    // 0 reader, 1 compute, 2 writer
            uint32_t common;             // common runtime args the source reads
        };

        // A circular buffer: tiles, format, and the pages of the pushes its kernels
        // issue.
        struct Cb {
            uint8_t index;
            uint32_t tiles;
            tt::DataFormat format;
            std::vector<uint32_t> pushes;
        };

        // Runtime arguments the kernel sources read: common per kernel (moments and
        // affine: reader, compute, writer), one per core each; the statistics kernel's
        // per core.
        constexpr uint32_t kMomentsCommon[] = {8, 3, 4}, kAffineCommon[] = {7, 3, 4};
        constexpr uint32_t kCoreArgs = 1, kStatsArgs = 20;

        tt::DataFormat format_of(DataType t) {
            return (
                t == DataType::FLOAT32 ? tt::DataFormat::Float32
                                       : tt::DataFormat::Float16_b
            );
        }

        // Runs `kernels` on min(grid, planes) cores with CBs {index, tiles, format}.
        void run_kernels(
            const std::vector<ttnn::Tensor>& io,
            const std::vector<Cb>& cbs,
            const std::vector<Kernel>& kernels,
            uint32_t planes
        ) {
            program::mark(program::Phase::Alloc);
            program::Key key("ktt_group_norm");
            key.add(planes);
            for (const auto& t : io) {
                key.add(t);
            }
            for (const auto& k : kernels) {
                key.add(k.source).add(k.kind);
                key.add(static_cast<int64_t>(k.head.size()));
                for (const int64_t t : k.head) {
                    key.add(t);
                }
                // Accessors as io indices (-1: a tensor outside io).
                key.add(static_cast<int64_t>(k.accessors.size()));
                for (const auto* a : k.accessors) {
                    int64_t index = -1;
                    for (size_t j = 0; j < io.size(); ++j) {
                        if (io[j].buffer() == a->buffer()) {
                            index = static_cast<int64_t>(j);
                            break;
                        }
                    }
                    key.add(index);
                }
                key.add(static_cast<int64_t>(k.tail.size()));
                for (const int64_t v : k.tail) {
                    key.add(v);
                }
            }
            for (const auto& cb : cbs) {
                key.add(cb.index).add(cb.tiles).add(static_cast<int64_t>(cb.format));
            }
            program::launch(key, io, [&] {
                program::Built built;
                const auto grid = detail::device()->compute_with_storage_grid_size();
                const uint32_t cores = (std::max<uint32_t>(
                    1, std::min<uint32_t>(grid.x * grid.y, planes)
                ));
                std::vector<CoreRange> ranges;
                for (uint32_t c = 0; c < cores; ++c) {
                    ranges.emplace_back(CoreCoord(c % grid.x, c / grid.x));
                }
                const CoreRangeSet range = CoreRangeSet(ranges).merge_ranges();
                auto& desc = built.desc;
                for (const auto& cb : cbs) {
                    const uint32_t page = cb.format == tt::DataFormat::Float32 ? 4096
                                                                               : 2048;
                    desc.cbs.push_back(
                        CBDescriptor{
                            .total_size = cb.tiles * page,
                            .core_ranges = range,
                            .format_descriptors = {CBFormatDescriptor{
                                .buffer_index = cb.index,
                                .data_format = cb.format,
                                .page_size = page
                            }}
                        }
                    );
                    built.rings.push_back({cb.index, cb.pushes, 0});
                }
                // Per core: the core index; common: head addresses, cores, tail.
                const auto args = program::core_index_args(cores);
                for (const auto& k : kernels) {
                    std::vector<uint32_t> common;
                    for (const uint32_t t : k.head) {
                        built.slots.push_back(
                            {.kernel = static_cast<uint32_t>(desc.kernels.size()),
                             .entry = program::Slot::kCommon,
                             .position = static_cast<uint32_t>(common.size()),
                             .tensor = t}
                        );
                        common.push_back(io[t].buffer()->address());
                    }
                    common.push_back(cores);
                    common.insert(common.end(), k.tail.begin(), k.tail.end());
                    program::mark(program::Phase::Args);
                    std::vector<uint32_t> ct;
                    for (const auto* t : k.accessors) {
                        const auto more = (TensorAccessorArgs(*t->buffer())
                                               .get_compile_time_args());
                        ct.insert(ct.end(), more.begin(), more.end());
                    }
                    built.args.push_back(
                        {.kernel = static_cast<uint32_t>(desc.kernels.size()),
                         .per_core = kCoreArgs,
                         .common = k.common}
                    );
                    KernelDescriptor kd{
                        .kernel_source = program::source(k.source),
                        .core_ranges = range,
                        .compile_time_args = ct,
                        .runtime_args = args,
                        .common_runtime_args = common,
                    };
                    if (k.kind == 0) {
                        kd.config = ReaderConfigDescriptor{};
                    } else if (k.kind == 2) {
                        kd.config = WriterConfigDescriptor{};
                    } else {
                        // fp32 CBs unpack to DST unrounded.
                        std::vector<UnpackToDestMode> unpack(
                            NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default
                        );
                        for (const auto& cb : cbs) {
                            if (cb.format == tt::DataFormat::Float32) {
                                unpack[cb.index] = UnpackToDestMode::UnpackToDestFp32;
                            }
                        }
                        kd.config = ComputeConfigDescriptor{
                            .math_fidelity = MathFidelity::HiFi4,
                            .fp32_dest_acc_en = true,
                            .unpack_to_dest_mode = unpack,
                            .math_approx_mode = false
                        };
                    }
                    desc.kernels.push_back(kd);
                }
                return built;
            });
        }

        ttnn::Tensor scalar_planes(uint32_t n, uint32_t c) {
            return ttnn::empty(
                ttnn::Shape({n, c, 1, 1}),
                DataType::FLOAT32,
                Layout::TILE,
                detail::device(),
                MemoryConfig{}
            );
        }

    }  // namespace

    std::pair<ttnn::Tensor, ttnn::Tensor> moments(const ttnn::Tensor& x) {
        const auto& s = x.logical_shape();
        const uint32_t n = s[0], c = s[1], h = s[2], w = s[3];
        const uint32_t ht = (h + 31) / 32, wt = (w + 31) / 32, planes = n * c;
        const uint32_t elem = x.dtype() == DataType::FLOAT32 ? 4 : 2;
        auto s1 = scalar_planes(n, c), s2 = scalar_planes(n, c);
        const auto xf = format_of(x.dtype());
        const auto f32 = tt::DataFormat::Float32;
        run_kernels(
            {x, s1, s2},
            {{0, 2, xf, {1}}, {16, 1, f32, {1}}, {17, 1, f32, {1}}},
            {
                {"nn/ktt_group_norm/moments_reader.cpp",
                 {&x},
                 {0},
                 {planes, ht, wt, h - (ht - 1) * 32, w - (wt - 1) * 32, elem},
                 0,
                 kMomentsCommon[0]},
                {"nn/ktt_group_norm/moments_compute.cpp",
                 {},
                 {},
                 {planes, ht * wt},
                 1,
                 kMomentsCommon[1]},
                {"nn/ktt_group_norm/moments_writer.cpp",
                 {&s1, &s2},
                 {1, 2},
                 {planes},
                 2,
                 kMomentsCommon[2]},
            },
            planes
        );
        return {s1, s2};
    }

    Stats stats(
        const ttnn::Tensor& s1,
        const ttnn::Tensor& s2,
        const ttnn::Tensor* weight,
        const ttnn::Tensor* bias,
        uint32_t groups,
        float count,
        float eps,
        DataType out
    ) {
        const auto& s = s1.logical_shape();
        const uint32_t n = s[0], c = s[1];
        Stats r{
            scalar_planes(n, c),
            scalar_planes(n, c),
            ttnn::empty(
                ttnn::Shape({n, groups}),
                out,
                Layout::TILE,
                detail::device(),
                MemoryConfig{}
            ),
            ttnn::empty(
                ttnn::Shape({n, groups}),
                out,
                Layout::TILE,
                detail::device(),
                MemoryConfig{}
            ),
        };
        std::vector<ttnn::Tensor> io{s1, s2};
        if (weight) {
            io.push_back(*weight);
        }
        if (bias) {
            io.push_back(*bias);
        }
        io.insert(io.end(), {r.a, r.b, r.mean, r.rstd});
        program::mark(program::Phase::Alloc);
        program::Key key("ktt_group_norm_stats");
        for (const auto& t : io) {
            key.add(t);
        }
        key.add(weight != nullptr).add(bias != nullptr).add(groups);
        key.add(std::bit_cast<uint32_t>(count)).add(std::bit_cast<uint32_t>(eps));
        // (image, group) units dealt over the grid; each unit's mean / rstd collected
        // on core 0 (16 B a unit), which writes the [N, G] tiles.
        const auto grid = detail::device()->compute_with_storage_grid_size();
        const uint32_t units = n * groups;
        const uint32_t parts_bytes = std::max<uint32_t>(
            64, (units * 16 + 63) / 64 * 64
        );
        const uint32_t cores =
            (parts_bytes > 65536
                 ? 1
                 : std::max<uint32_t>(1, std::min<uint32_t>(grid.x * grid.y, units)));
        program::launch(key, io, [&] {
            const auto elem = [](const ttnn::Tensor* t) -> uint32_t {
                return t == nullptr ? 0 : t->dtype() == DataType::FLOAT32 ? 4 : 2;
            };
            const auto addr = [](const ttnn::Tensor* t) -> uint32_t {
                return t == nullptr ? 0 : t->buffer()->address();
            };
            const auto bits = [](float f) {
                return __builtin_bit_cast(uint32_t, f);
            };
            std::vector<CoreRange> ranges;
            for (uint32_t j = 0; j < cores; ++j) {
                ranges.emplace_back(CoreCoord(j % grid.x, j / grid.x));
            }
            const CoreRangeSet core = CoreRangeSet(ranges).merge_ranges();
            ProgramDescriptor desc;
            for (
                auto [index, bytes] :
                std::initializer_list<std::pair<uint8_t, uint32_t>>{
                    {0, 128 * 64},
                    {1, 128 * 64},
                    {2, 128 * 64},
                    {3, 4096},
                    {4, 4096},
                    {5, parts_bytes}
                }
            ) {
                desc.cbs.push_back(
                    CBDescriptor{
                        .total_size = bytes,
                        .core_ranges = core,
                        .format_descriptors = {CBFormatDescriptor{
                            .buffer_index = index,
                            .data_format = tt::DataFormat::Float32,
                            .page_size = bytes
                        }}
                    }
                );
            }
            std::vector<uint32_t> ct;
            for (
                const ttnn::Tensor* t :
                {&s1,
                 &s2,
                 weight ? weight : &s1,
                 bias ? bias : &s1,
                 static_cast<const ttnn::Tensor*>(&r.a),
                 static_cast<const ttnn::Tensor*>(&r.b),
                 static_cast<const ttnn::Tensor*>(&r.mean),
                 static_cast<const ttnn::Tensor*>(&r.rstd)}
            ) {
                const auto more = (TensorAccessorArgs(*t->buffer())
                                       .get_compile_time_args());
                ct.insert(ct.end(), more.begin(), more.end());
            }
            const std::vector<uint32_t> shared{
                s1.buffer()->address(),
                s2.buffer()->address(),
                addr(weight),
                addr(bias),
                r.a.buffer()->address(),
                r.b.buffer()->address(),
                r.mean.buffer()->address(),
                r.rstd.buffer()->address(),
                n,
                c,
                groups,
                bits(count),
                bits(eps),
                elem(weight),
                elem(bias),
                out == DataType::FLOAT32 ? 4u : 2u
            };
            desc.semaphores.push_back(
                SemaphoreDescriptor{
                    .id = 0,
                    .core_type = tt::CoreType::WORKER,
                    .core_ranges = core,
                    .initial_value = 0
                }
            );
            const auto head = detail::device()->worker_core_from_logical_core(
                CoreCoord(0, 0)
            );
            KernelDescriptor::RuntimeArgs args;
            for (uint32_t j = 0; j < cores; ++j) {
                auto a = shared;
                a.insert(
                    a.end(),
                    {j,
                     cores,
                     static_cast<uint32_t>(head.x),
                     static_cast<uint32_t>(head.y)}
                );
                args.push_back({CoreCoord(j % grid.x, j / grid.x), std::move(a)});
            }
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("nn/ktt_group_norm/stats.cpp"),
                    .core_ranges = core,
                    .compile_time_args = ct,
                    .runtime_args = args,
                    .config = ReaderConfigDescriptor{}
                }
            );
            // Address args 0..7: s1, s2, weight, bias (0 when absent), then the
            // outputs.
            program::Built built{std::move(desc), {}};
            built.args = {{.kernel = 0, .per_core = kStatsArgs}};
            // Reserved once and written in place: 2 x 64 slots of 64 B each for the
            // sums, weight / bias reads and A / B; the mean and rstd tiles; the unit
            // parts.
            const uint64_t out_tile = out == DataType::FLOAT32 ? 4096 : 2048;
            built.rings = {
                {0, {}, 128 * 64},
                {1, {}, 128 * 64},
                {2, {}, 128 * 64},
                {3, {}, out_tile},
                {4, {}, out_tile},
                {5, {}, parts_bytes}
            };
            uint32_t tensor = 0;
            for (uint32_t position = 0; position < 8; ++position) {
                const bool
                    absent = ((position == 2 && !weight) || (position == 3 && !bias));
                if (!absent) {
                    built.slots.push_back({.position = position, .tensor = tensor++});
                }
            }
            return built;
        });
        return r;
    }

    ttnn::Tensor affine(
        const ttnn::Tensor& x,
        const ttnn::Tensor& a,
        const ttnn::Tensor& b,
        const tt::tt_metal::MemoryConfig& memory
    ) {
        const auto& s = x.logical_shape();
        const uint32_t n = s[0], c = s[1], h = s[2], w = s[3];
        const uint32_t tiles = ((h + 31) / 32) * ((w + 31) / 32), planes = n * c;
        const uint32_t elem = x.dtype() == DataType::FLOAT32 ? 4 : 2;
        auto y = ttnn::empty(s, x.dtype(), Layout::TILE, detail::device(), memory);
        const auto xf = format_of(x.dtype());
        const auto f32 = tt::DataFormat::Float32;
        run_kernels(
            {x, a, b, y},
            // Output: two tiles per push when a plane has an even tile count, else one.
            {{0, 4, xf, {1}},
             {1, 1, f32, {1}},
             {2, 1, f32, {1}},
             {16, 4, xf, {tiles % 2 == 0 ? 2u : 1u}}},
            {
                {"nn/ktt_group_norm/affine_reader.cpp",
                 {&x, &a, &b},
                 {0, 1, 2},
                 {planes, tiles, elem},
                 0,
                 kAffineCommon[0]},
                {"nn/ktt_group_norm/affine_compute.cpp",
                 {},
                 {},
                 {planes, tiles},
                 1,
                 kAffineCommon[1]},
                {"nn/ktt_group_norm/affine_writer.cpp",
                 {&y},
                 {3},
                 {planes, tiles},
                 2,
                 kAffineCommon[2]},
            },
            planes
        );
        return y;
    }

    namespace {

        // x tiles per read barrier and y tiles per write barrier of the row-form
        // programs; tiles a core takes at least; most cores of one segment (rows of a
        // partial page).
        constexpr uint32_t kBlock = 8, kMinTiles = 4, kMaxSegmentCores = 32;
        // Runtime args the row-form kernel sources read. Common: moments reader,
        // compute, writer; affine reader, compute, writer. Per core: q0, q1, then the
        // moments writer's index in its first segment, the affine kernels' core count
        // per segment read.
        constexpr uint32_t kRowsMomentsCommon[] = {5, 4, 4},
                           kRowsAffineCommon[] = {12, 11, 8};
        constexpr uint32_t kRowsRange = 2;
        // The backward's: sums reader, compute; affine reader, compute (the writers are
        // the forward's).
        constexpr uint32_t kRowsBackwardMomentsCommon[] = {6, 4},
                           kRowsBackwardAffineCommon[] = {13, 9};
        // Most channel tiles the groups of one channel tile may reach, and most
        // (segment, reached tile) pairs of a call (they bound a core's runtime args).
        constexpr uint32_t kMaxReach = 8, kMaxReads = 2048;

        // The cores of a row-form program and their ranges of the tile sequence.
        struct RowsSplit {
            ::ktt_group_norm::Geometry g{1, 1, 1};
            std::vector<CoreCoord> cores;
            CoreRangeSet range;
            // Per core: q0, q1, its index among the cores of its first segment; the
            // core count of the next image's segment per shared row tile its pieces end
            // in; the partial rows of each segment its pieces read, in reading order.
            std::vector<std::array<uint32_t, 3>> spans;
            std::vector<std::vector<uint32_t>> heads, counts;
        };

        RowsSplit rows_split(const Rows& r) {
            using ::ktt_group_norm::Piece;
            RowsSplit s;
            s.g = {(r.channels + 31) / 32, r.rows, r.images};
            const auto& g = s.g;
            const uint64_t segments = static_cast<uint64_t>(r.images) * g.ct;
            const uint64_t tiles = static_cast<uint64_t>(g.ct) * g.first(r.images);
            const auto grid = detail::device()->compute_with_storage_grid_size();
            // Ranges of at least 1 / 30 of a segment: at most 32 cores touch one.
            const uint64_t cap = (kMaxSegmentCores - 2) * segments;
            const uint32_t count = static_cast<uint32_t>(std::max<uint64_t>(
                1, std::min<uint64_t>({grid.x * grid.y, tiles / kMinTiles, cap})
            ));
            // Pieces of the range [q0, q1).
            const auto pieces = [&](uint32_t q0, uint32_t q1) {
                std::vector<Piece> out;
                for (uint32_t q = q0; q < q1; q += out.back().len) {
                    out.emplace_back(g, q, q1);
                }
                return out;
            };
            std::vector<uint32_t> first(segments, 0), per_segment(segments, 0);
            std::vector<CoreRange> ranges;
            for (uint32_t j = 0; j < count; ++j) {
                const uint32_t q0 = static_cast<uint32_t>(tiles * j / count);
                const uint32_t q1 = static_cast<uint32_t>(tiles * (j + 1) / count);
                const auto mine = pieces(q0, q1);
                for (const auto& p : mine) {
                    if (per_segment[p.segment(g)]++ == 0) {
                        first[p.segment(g)] = j;
                    }
                }
                s.spans.push_back({q0, q1, j - first[mine.front().segment(g)]});
                const CoreCoord core(j % grid.x, j / grid.x);
                s.cores.push_back(core);
                ranges.emplace_back(core, core);
            }
            // Partial rows of segment (n, cb): its cores', then one from the row tile
            // it shares with the image before.
            const auto rows_of = [&](uint32_t n, uint32_t cb) {
                const uint32_t rows = per_segment[n * g.ct + cb] +
                    (n > 0 && g.shared(n - 1) ? 1 : 0);
                if (rows > kMaxSegmentCores) {
                    throw std::logic_error(
                        "ktt_group_norm: a segment has more cores than a partial page "
                        "has rows"
                    );
                }
                return rows;
            };
            const uint32_t cg = r.channels / r.groups;
            for (const auto& span : s.spans) {
                std::vector<uint32_t> heads, counts;
                for (const auto& p : pieces(span[0], span[1])) {
                    const ::ktt_group_norm::Span reach(p.cb, cg, r.channels);
                    const bool shared = g.shared(p.n) && p.last;
                    for (uint32_t n = p.n; n <= p.n + (shared ? 1 : 0); ++n) {
                        for (uint32_t t = reach.t_lo; t <= reach.t_hi; ++t) {
                            counts.push_back(rows_of(n, t));
                        }
                    }
                    if (shared) {
                        heads.push_back(per_segment[(p.n + 1) * g.ct + p.cb]);
                    }
                }
                s.heads.push_back(std::move(heads));
                s.counts.push_back(std::move(counts));
            }
            s.range = CoreRangeSet(ranges).merge_ranges();
            return s;
        }

        CBDescriptor rows_cb(
            uint8_t index,
            uint32_t pages,
            tt::DataFormat format,
            const CoreRangeSet& range
        ) {
            const uint32_t page = tt::tile_size(format);
            return CBDescriptor{
                .total_size = pages * page,
                .core_ranges = range,
                .format_descriptors = {CBFormatDescriptor{
                    .buffer_index = index, .data_format = format, .page_size = page
                }}
            };
        }

        // The compute config of the row-form kernels: fp32 DST, fp32 CBs unpacked
        // unrounded.
        ComputeConfigDescriptor rows_compute(const ProgramDescriptor& desc) {
            std::vector<UnpackToDestMode> unpack(
                NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default
            );
            for (const auto& cb : desc.cbs) {
                for (const auto& f : cb.format_descriptors) {
                    if (f.data_format == tt::DataFormat::Float32) {
                        unpack[f.buffer_index] = UnpackToDestMode::UnpackToDestFp32;
                    }
                }
            }
            return ComputeConfigDescriptor{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true,
                .unpack_to_dest_mode = unpack,
                .math_approx_mode = false
            };
        }

        // A zero fp32 TILE [images, C] (the kernels write row n of the images' tiles
        // only), filled by a device program so that a trace can record it.
        ttnn::Tensor per_channel_zeros(const Rows& r) {
            return ttnn::moreh_full(
                {r.images, r.channels}, 0.0f, detail::device(), DataType::FLOAT32
            );
        }

        std::vector<uint32_t> accessors(
            std::initializer_list<const ttnn::Tensor*> tensors
        ) {
            std::vector<uint32_t> ct;
            for (const auto* t : tensors) {
                const auto more = TensorAccessorArgs(*t->buffer())
                                      .get_compile_time_args();
                ct.insert(ct.end(), more.begin(), more.end());
            }
            return ct;
        }

    }  // namespace

    bool rows_supported(const Rows& r) {
        // Several images sharing row tiles: at most two per tile.
        if (r.groups == 0 || r.channels % r.groups != 0 || r.rows == 0 ||
            (r.images > 1 && r.rows % 32 != 0 && r.rows < 32)) {
            return false;
        }
        const uint32_t cg = r.channels / r.groups, ct = (r.channels + 31) / 32;
        uint32_t reach = 0;
        for (uint32_t cb = 0; cb < ct; ++cb) {
            reach = std::max(reach, ::ktt_group_norm::Span(cb, cg, r.channels).tiles());
        }
        return reach <= kMaxReach &&
            static_cast<uint64_t>(r.images) * ct * reach <= kMaxReads;
    }

    ttnn::Tensor rows_norm(
        const ttnn::Tensor& x,
        const ttnn::Tensor* weight,
        const ttnn::Tensor* bias,
        const Rows& r,
        const tt::tt_metal::MemoryConfig& memory,
        std::pair<ttnn::Tensor, ttnn::Tensor>* sums
    ) {
        const uint32_t cg = r.channels / r.groups;
        const uint32_t ct = (r.channels + 31) / 32;
        const auto xf = format_of(x.dtype());
        const auto f32 = tt::DataFormat::Float32;
        // Partials: per segment two fp32 tiles (sum x, sum x^2), row k from its k-th
        // core.
        auto partials = ttnn::empty(
            ttnn::Shape({r.images * ct * 2 * 32, 32}),
            DataType::FLOAT32,
            Layout::TILE,
            detail::device(),
            MemoryConfig{}
        );
        auto y = ttnn::empty(
            x.logical_shape(), x.dtype(), Layout::TILE, detail::device(), memory
        );
        program::mark(program::Phase::Alloc);

        program::Key moments_key("ktt_group_norm_rows_moments");
        moments_key.add(x).add(partials).add(r.images).add(r.rows).add(r.channels);
        program::launch(moments_key, {x, partials}, [&] {
            const RowsSplit s = rows_split(r);
            program::Built built;
            auto& desc = built.desc;
            desc.cbs.push_back(rows_cb(0, 2 * kBlock, xf, s.range));
            desc.cbs.push_back(rows_cb(16, 4, f32, s.range));
            built.rings = {{.cb = 0, .pushes = {kBlock}}, {.cb = 16, .pushes = {2}}};
            KernelDescriptor::RuntimeArgs range_args, writer_args;
            for (size_t j = 0; j < s.cores.size(); ++j) {
                range_args.push_back({s.cores[j], {s.spans[j][0], s.spans[j][1]}});
                std::vector<uint32_t> w{s.spans[j][0], s.spans[j][1], s.spans[j][2]};
                w.insert(w.end(), s.heads[j].begin(), s.heads[j].end());
                writer_args.push_back({s.cores[j], w});
            }
            program::mark(program::Phase::Args);
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source(
                        "nn/ktt_group_norm/rows_moments_reader.cpp"
                    ),
                    .core_ranges = s.range,
                    .compile_time_args = accessors({&x}),
                    .runtime_args = range_args,
                    .common_runtime_args =
                        {x.buffer()->address(), s.g.ct, s.g.rows, s.g.images, kBlock},
                    .config = ReaderConfigDescriptor{}
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source(
                        "nn/ktt_group_norm/rows_moments_compute.cpp"
                    ),
                    .core_ranges = s.range,
                    .runtime_args = range_args,
                    .common_runtime_args = {s.g.ct, s.g.rows, s.g.images, kBlock},
                    .config = rows_compute(desc)
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source(
                        "nn/ktt_group_norm/rows_moments_writer.cpp"
                    ),
                    .core_ranges = s.range,
                    .compile_time_args = accessors({&partials}),
                    .runtime_args = writer_args,
                    .common_runtime_args =
                        {partials.buffer()->address(), s.g.ct, s.g.rows, s.g.images},
                    .config = WriterConfigDescriptor{}
                }
            );
            built.slots = {
                {.kernel = 0,
                 .entry = program::Slot::kCommon,
                 .position = 0,
                 .tensor = 0},
                {.kernel = 2,
                 .entry = program::Slot::kCommon,
                 .position = 0,
                 .tensor = 1},
            };
            built.args = {
                {.kernel = 0, .per_core = kRowsRange, .common = kRowsMomentsCommon[0]},
                {.kernel = 1, .per_core = kRowsRange, .common = kRowsMomentsCommon[1]},
                {.kernel = 2,
                 .per_core = kRowsRange + 1,
                 .common = kRowsMomentsCommon[2],
                 .per_item = 1},
            };
            return built;
        });

        std::vector<ttnn::Tensor> io{x, partials};
        if (weight) {
            io.push_back(*weight);
        }
        if (bias) {
            io.push_back(*bias);
        }
        const uint32_t sums_index = static_cast<uint32_t>(io.size());
        if (sums) {
            // Per (image, channel) its group's mean and rstd: row n of fp32 [images,
            // C].
            for (auto* t : {&sums->first, &sums->second}) {
                *t = per_channel_zeros(r);
                io.push_back(*t);
            }
        }
        io.push_back(y);
        const float inv_count = 1.0f /
            (static_cast<float>(r.rows) * static_cast<float>(cg));
        program::Key key("ktt_group_norm_rows_affine");
        for (const auto& t : io) {
            key.add(t);
        }
        key.add(weight != nullptr)
            .add(bias != nullptr)
            .add(r.images)
            .add(r.rows)
            .add(r.channels)
            .add(r.groups);
        key.add(sums != nullptr);
        key.add(std::bit_cast<uint32_t>(r.eps));
        return program::launch(key, io, [&] {
            const RowsSplit s = rows_split(r);
            program::Built built;
            auto& desc = built.desc;
            const auto wf = weight ? format_of(weight->dtype()) : f32;
            const auto bf = bias ? format_of(bias->dtype()) : f32;
            desc.cbs.push_back(rows_cb(0, 2 * kBlock, xf, s.range));
            desc.cbs.push_back(rows_cb(1, 2, wf, s.range));
            desc.cbs.push_back(rows_cb(2, 2, bf, s.range));
            for (const uint8_t index : {3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) {
                desc.cbs.push_back(rows_cb(index, 2, f32, s.range));
            }
            desc.cbs.push_back(rows_cb(16, 2 * kBlock, xf, s.range));
            built.rings = {
                {.cb = 0, .pushes = {kBlock}},
                {.cb = 1, .pushes = {1}},
                {.cb = 2, .pushes = {1}},
                {.cb = 3, .pushes = {2}},
                {.cb = 4, .pushes = {1}},
                {.cb = 5, .pushes = {2}},
                {.cb = 6, .pushes = {2}},
                {.cb = 7, .pushes = {2}},
                {.cb = 8, .pushes = {2}},
                {.cb = 9, .pushes = {2}},
                {.cb = 10, .pushes = {2}},
                {.cb = 11, .pushes = {2}},
                {.cb = 12, .pushes = {2}},
                {.cb = 16, .pushes = {kBlock}},
            };
            KernelDescriptor::RuntimeArgs args;
            for (size_t j = 0; j < s.cores.size(); ++j) {
                std::vector<uint32_t> a{s.spans[j][0], s.spans[j][1]};
                a.insert(a.end(), s.counts[j].begin(), s.counts[j].end());
                args.push_back({s.cores[j], a});
            }
            program::mark(program::Phase::Args);
            const uint32_t y_index = static_cast<uint32_t>(io.size() - 1);
            const uint32_t w_index = 2, b_index = weight ? 3 : 2;
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source(
                        "nn/ktt_group_norm/rows_affine_reader.cpp"
                    ),
                    .core_ranges = s.range,
                    .compile_time_args = accessors(
                        {&x, &partials, weight ? weight : &x, bias ? bias : &x}
                    ),
                    .runtime_args = args,
                    .common_runtime_args =
                        {x.buffer()->address(),
                         partials.buffer()->address(),
                         weight ? weight->buffer()->address() : 0u,
                         bias ? bias->buffer()->address() : 0u,
                         s.g.ct,
                         s.g.rows,
                         s.g.images,
                         kBlock,
                         weight ? 1u : 0u,
                         bias ? 1u : 0u,
                         cg,
                         r.channels},
                    .config = ReaderConfigDescriptor{}
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source(
                        "nn/ktt_group_norm/rows_affine_compute.cpp"
                    ),
                    .core_ranges = s.range,
                    .runtime_args = args,
                    .common_runtime_args =
                        {s.g.ct,
                         s.g.rows,
                         s.g.images,
                         kBlock,
                         weight ? 1u : 0u,
                         bias ? 1u : 0u,
                         std::bit_cast<uint32_t>(inv_count),
                         std::bit_cast<uint32_t>(r.eps),
                         cg,
                         r.channels,
                         sums ? 1u : 0u},
                    .config = rows_compute(desc)
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source(
                        "nn/ktt_group_norm/rows_affine_writer.cpp"
                    ),
                    .core_ranges = s.range,
                    .compile_time_args = accessors(
                        {&y, sums ? &sums->first : &y, sums ? &sums->second : &y}
                    ),
                    .runtime_args = args,
                    .common_runtime_args =
                        {y.buffer()->address(),
                         s.g.ct,
                         s.g.rows,
                         s.g.images,
                         kBlock,
                         sums ? 1u : 0u,
                         sums ? sums->first.buffer()->address() : 0u,
                         sums ? sums->second.buffer()->address() : 0u},
                    .config = WriterConfigDescriptor{}
                }
            );
            if (sums) {
                built.slots.push_back(
                    {.kernel = 2,
                     .entry = program::Slot::kCommon,
                     .position = 6,
                     .tensor = sums_index}
                );
                built.slots.push_back(
                    {.kernel = 2,
                     .entry = program::Slot::kCommon,
                     .position = 7,
                     .tensor = sums_index + 1}
                );
            }
            built.slots.insert(
                built.slots.end(),
                {
                    {.kernel = 0,
                     .entry = program::Slot::kCommon,
                     .position = 0,
                     .tensor = 0},
                    {.kernel = 0,
                     .entry = program::Slot::kCommon,
                     .position = 1,
                     .tensor = 1},
                    {.kernel = 2,
                     .entry = program::Slot::kCommon,
                     .position = 0,
                     .tensor = y_index},
                }
            );
            if (weight) {
                built.slots.push_back(
                    {.kernel = 0,
                     .entry = program::Slot::kCommon,
                     .position = 2,
                     .tensor = w_index}
                );
            }
            if (bias) {
                built.slots.push_back(
                    {.kernel = 0,
                     .entry = program::Slot::kCommon,
                     .position = 3,
                     .tensor = b_index}
                );
            }
            for (uint32_t k = 0; k < 3; ++k) {
                built.args.push_back(
                    {.kernel = k,
                     .per_core = kRowsRange,
                     .common = kRowsAffineCommon[k],
                     .per_item = 1}
                );
            }
            return built;
        });
    }

    ttnn::Tensor rows_backward(
        const ttnn::Tensor& x,
        const ttnn::Tensor& dy,
        const ttnn::Tensor& mean,
        const ttnn::Tensor& rstd,
        const ttnn::Tensor* weight,
        const Rows& r,
        const tt::tt_metal::MemoryConfig& memory,
        std::pair<ttnn::Tensor, ttnn::Tensor>& terms
    ) {
        const uint32_t cg = r.channels / r.groups;
        const uint32_t ct = (r.channels + 31) / 32;
        const auto xf = format_of(x.dtype());
        const auto f32 = tt::DataFormat::Float32;
        const auto small = [&](uint32_t rows, uint32_t columns) {
            return ttnn::empty(
                ttnn::Shape({rows, columns}),
                DataType::FLOAT32,
                Layout::TILE,
                detail::device(),
                MemoryConfig{}
            );
        };
        // Partials: per segment two fp32 tiles (sum dy, sum dy x), row k from its k-th
        // core.
        auto partials = small(r.images * ct * 2 * 32, 32);
        terms = {per_channel_zeros(r), per_channel_zeros(r)};
        auto dx = ttnn::empty(
            x.logical_shape(), x.dtype(), Layout::TILE, detail::device(), memory
        );
        program::mark(program::Phase::Alloc);

        program::Key moments_key("ktt_group_norm_rows_bwd_moments");
        moments_key.add(x).add(dy).add(partials).add(r.images).add(r.rows).add(
            r.channels
        );
        program::launch(moments_key, {x, dy, partials}, [&] {
            const RowsSplit s = rows_split(r);
            program::Built built;
            auto& desc = built.desc;
            desc.cbs.push_back(rows_cb(0, 2 * kBlock, xf, s.range));
            desc.cbs.push_back(rows_cb(1, 2 * kBlock, xf, s.range));
            desc.cbs.push_back(rows_cb(16, 4, f32, s.range));
            built.rings = {
                {.cb = 0, .pushes = {kBlock}},
                {.cb = 1, .pushes = {kBlock}},
                {.cb = 16, .pushes = {2}}
            };
            KernelDescriptor::RuntimeArgs range_args, writer_args;
            for (size_t j = 0; j < s.cores.size(); ++j) {
                range_args.push_back({s.cores[j], {s.spans[j][0], s.spans[j][1]}});
                std::vector<uint32_t> w{s.spans[j][0], s.spans[j][1], s.spans[j][2]};
                w.insert(w.end(), s.heads[j].begin(), s.heads[j].end());
                writer_args.push_back({s.cores[j], w});
            }
            program::mark(program::Phase::Args);
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source(
                        "nn/ktt_group_norm/rows_bwd_moments_reader.cpp"
                    ),
                    .core_ranges = s.range,
                    .compile_time_args = accessors({&x, &dy}),
                    .runtime_args = range_args,
                    .common_runtime_args =
                        {x.buffer()->address(),
                         dy.buffer()->address(),
                         s.g.ct,
                         s.g.rows,
                         s.g.images,
                         kBlock},
                    .config = ReaderConfigDescriptor{}
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source(
                        "nn/ktt_group_norm/rows_bwd_moments_compute.cpp"
                    ),
                    .core_ranges = s.range,
                    .runtime_args = range_args,
                    .common_runtime_args = {s.g.ct, s.g.rows, s.g.images, kBlock},
                    .config = rows_compute(desc)
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source(
                        "nn/ktt_group_norm/rows_moments_writer.cpp"
                    ),
                    .core_ranges = s.range,
                    .compile_time_args = accessors({&partials}),
                    .runtime_args = writer_args,
                    .common_runtime_args =
                        {partials.buffer()->address(), s.g.ct, s.g.rows, s.g.images},
                    .config = WriterConfigDescriptor{}
                }
            );
            built.slots = {
                {.kernel = 0,
                 .entry = program::Slot::kCommon,
                 .position = 0,
                 .tensor = 0},
                {.kernel = 0,
                 .entry = program::Slot::kCommon,
                 .position = 1,
                 .tensor = 1},
                {.kernel = 2,
                 .entry = program::Slot::kCommon,
                 .position = 0,
                 .tensor = 2},
            };
            built.args = {
                {.kernel = 0,
                 .per_core = kRowsRange,
                 .common = kRowsBackwardMomentsCommon[0]},
                {.kernel = 1,
                 .per_core = kRowsRange,
                 .common = kRowsBackwardMomentsCommon[1]},
                {.kernel = 2,
                 .per_core = kRowsRange + 1,
                 .common = kRowsMomentsCommon[2],
                 .per_item = 1},
            };
            return built;
        });

        // io: x, dy, partials, mean, rstd, weight when given, the two term tensors, dx.
        std::vector<ttnn::Tensor> io{x, dy, partials, mean, rstd};
        if (weight) {
            io.push_back(*weight);
        }
        const uint32_t terms_index = static_cast<uint32_t>(io.size());
        io.push_back(terms.first);
        io.push_back(terms.second);
        io.push_back(dx);
        const float inv_count = 1.0f /
            (static_cast<float>(r.rows) * static_cast<float>(cg));
        program::Key key("ktt_group_norm_rows_bwd_affine");
        for (const auto& t : io) {
            key.add(t);
        }
        key.add(weight != nullptr)
            .add(r.images)
            .add(r.rows)
            .add(r.channels)
            .add(r.groups);
        return program::launch(key, io, [&] {
            const RowsSplit s = rows_split(r);
            program::Built built;
            auto& desc = built.desc;
            const auto wf = weight ? format_of(weight->dtype()) : f32;
            desc.cbs.push_back(rows_cb(0, 2 * kBlock, xf, s.range));
            desc.cbs.push_back(rows_cb(1, 2, wf, s.range));
            desc.cbs.push_back(rows_cb(2, 2 * kBlock, xf, s.range));
            // fp32 tiles: pairs, and triples for the three coefficient rows.
            for (const uint8_t index : {3, 4, 5, 6, 8, 10, 12, 13, 14}) {
                desc.cbs.push_back(rows_cb(index, 2, f32, s.range));
            }
            for (const uint8_t index : {7, 9, 11}) {
                desc.cbs.push_back(rows_cb(index, 3, f32, s.range));
            }
            desc.cbs.push_back(rows_cb(16, 2 * kBlock, xf, s.range));
            built.rings = {
                {.cb = 0, .pushes = {kBlock}},
                {.cb = 1, .pushes = {1}},
                {.cb = 2, .pushes = {kBlock}},
                {.cb = 3, .pushes = {2}},
                {.cb = 4, .pushes = {1}},
                {.cb = 5, .pushes = {2}},
                {.cb = 6, .pushes = {2}},
                {.cb = 7, .pushes = {3}},
                {.cb = 8, .pushes = {2}},
                {.cb = 9, .pushes = {3}},
                {.cb = 10, .pushes = {2}},
                {.cb = 11, .pushes = {3}},
                {.cb = 12, .pushes = {2}},
                {.cb = 13, .pushes = {2}},
                {.cb = 14, .pushes = {2}},
                {.cb = 16, .pushes = {kBlock}},
            };
            KernelDescriptor::RuntimeArgs args;
            for (size_t j = 0; j < s.cores.size(); ++j) {
                std::vector<uint32_t> a{s.spans[j][0], s.spans[j][1]};
                a.insert(a.end(), s.counts[j].begin(), s.counts[j].end());
                args.push_back({s.cores[j], a});
            }
            program::mark(program::Phase::Args);
            const uint32_t dx_index = static_cast<uint32_t>(io.size() - 1);
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source(
                        "nn/ktt_group_norm/rows_bwd_affine_reader.cpp"
                    ),
                    .core_ranges = s.range,
                    .compile_time_args = accessors(
                        {&x, &dy, &partials, weight ? weight : &x, &mean, &rstd}
                    ),
                    .runtime_args = args,
                    .common_runtime_args =
                        {x.buffer()->address(),
                         dy.buffer()->address(),
                         partials.buffer()->address(),
                         weight ? weight->buffer()->address() : 0u,
                         mean.buffer()->address(),
                         rstd.buffer()->address(),
                         s.g.ct,
                         s.g.rows,
                         s.g.images,
                         kBlock,
                         weight ? 1u : 0u,
                         cg,
                         r.channels},
                    .config = ReaderConfigDescriptor{}
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source(
                        "nn/ktt_group_norm/rows_bwd_affine_compute.cpp"
                    ),
                    .core_ranges = s.range,
                    .runtime_args = args,
                    .common_runtime_args =
                        {s.g.ct,
                         s.g.rows,
                         s.g.images,
                         kBlock,
                         weight ? 1u : 0u,
                         std::bit_cast<uint32_t>(inv_count),
                         cg,
                         r.channels,
                         1u},
                    .config = rows_compute(desc)
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source(
                        "nn/ktt_group_norm/rows_affine_writer.cpp"
                    ),
                    .core_ranges = s.range,
                    .compile_time_args = accessors({&dx, &terms.first, &terms.second}),
                    .runtime_args = args,
                    .common_runtime_args =
                        {dx.buffer()->address(),
                         s.g.ct,
                         s.g.rows,
                         s.g.images,
                         kBlock,
                         1u,
                         terms.first.buffer()->address(),
                         terms.second.buffer()->address()},
                    .config = WriterConfigDescriptor{}
                }
            );
            constexpr auto common = program::Slot::kCommon;
            built.slots = {
                {.kernel = 0, .entry = common, .position = 0, .tensor = 0},
                {.kernel = 0, .entry = common, .position = 1, .tensor = 1},
                {.kernel = 0, .entry = common, .position = 2, .tensor = 2},
                {.kernel = 0, .entry = common, .position = 4, .tensor = 3},
                {.kernel = 0, .entry = common, .position = 5, .tensor = 4},
                {.kernel = 2, .entry = common, .position = 0, .tensor = dx_index},
                {.kernel = 2, .entry = common, .position = 6, .tensor = terms_index},
                {.kernel = 2,
                 .entry = common,
                 .position = 7,
                 .tensor = terms_index + 1},
            };
            if (weight) {
                built.slots.push_back(
                    {.kernel = 0, .entry = common, .position = 3, .tensor = 5}
                );
            }
            built.args = {
                {.kernel = 0,
                 .per_core = kRowsRange,
                 .common = kRowsBackwardAffineCommon[0],
                 .per_item = 1},
                {.kernel = 1,
                 .per_core = kRowsRange,
                 .common = kRowsBackwardAffineCommon[1],
                 .per_item = 1},
                {.kernel = 2,
                 .per_core = kRowsRange,
                 .common = kRowsAffineCommon[2],
                 .per_item = 1},
            };
            return built;
        });
    }

}  // namespace ktt::bridge::ktt_group_norm
