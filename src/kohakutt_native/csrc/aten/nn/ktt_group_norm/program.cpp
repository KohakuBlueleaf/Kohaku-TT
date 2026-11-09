// Host side of ktt_group_norm: planes split over the grid (strided), circular buffers,
// runtime args of the moments and affine programs.
#include "aten/nn/ktt_group_norm/program.hpp"

#include <algorithm>
#include <bit>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "core/device/handle.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"
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
        };

        tt::DataFormat format_of(DataType t) {
            return (
                t == DataType::FLOAT32 ? tt::DataFormat::Float32
                                       : tt::DataFormat::Float16_b
            );
        }

        // Runs `kernels` on min(grid, planes) cores with CBs {index, tiles, format}.
        void run_kernels(
            const std::vector<ttnn::Tensor>& io,
            const std::vector<std::tuple<uint8_t, uint32_t, tt::DataFormat>>& cbs,
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
            for (auto [index, tiles, df] : cbs) {
                key.add(index).add(tiles).add(static_cast<int64_t>(df));
            }
            program::launch(key, io, [&] {
                program::Built built;
                const auto grid = detail::device()->compute_with_storage_grid_size();
                const uint32_t cores = (
                    std::max<uint32_t>(1, std::min<uint32_t>(grid.x * grid.y, planes))
                );
                std::vector<CoreRange> ranges;
                for (uint32_t c = 0; c < cores; ++c) {
                    ranges.emplace_back(CoreCoord(c % grid.x, c / grid.x));
                }
                const CoreRangeSet range = CoreRangeSet(ranges).merge_ranges();
                auto& desc = built.desc;
                for (auto [index, tiles, df] : cbs) {
                    const uint32_t page = df == tt::DataFormat::Float32 ? 4096 : 2048;
                    desc.cbs.push_back(
                        CBDescriptor{
                            .total_size = tiles * page,
                            .core_ranges = range,
                            .format_descriptors = {CBFormatDescriptor{
                                .buffer_index = index,
                                .data_format = df,
                                .page_size = page
                            }}
                        }
                    );
                }
                // Per core: the core index; common: head addresses, cores, tail.
                const auto args = program::core_index_args(cores);
                for (const auto& k : kernels) {
                    std::vector<uint32_t> common;
                    for (const uint32_t t : k.head) {
                        built.slots.push_back({
                            .kernel = static_cast<uint32_t>(desc.kernels.size()),
                            .entry = program::Slot::kCommon,
                            .position = static_cast<uint32_t>(common.size()),
                            .tensor = t
                        });
                        common.push_back(io[t].buffer()->address());
                    }
                    common.push_back(cores);
                    common.insert(common.end(), k.tail.begin(), k.tail.end());
                    program::mark(program::Phase::Args);
                    std::vector<uint32_t> ct;
                    for (const auto* t : k.accessors) {
                        const auto more = (
                            TensorAccessorArgs(*t->buffer()).get_compile_time_args()
                        );
                        ct.insert(ct.end(), more.begin(), more.end());
                    }
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
                        for (auto [index, tiles, df] : cbs) {
                            if (df == tt::DataFormat::Float32) {
                                unpack[index] = UnpackToDestMode::UnpackToDestFp32;
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
            {{0, 2, xf}, {16, 1, f32}, {17, 1, f32}},
            {
                {
                    "nn/ktt_group_norm/moments_reader.cpp",
                    {&x},
                    {0},
                    {planes, ht, wt, h - (ht - 1) * 32, w - (wt - 1) * 32, elem},
                    0
                },
                {"nn/ktt_group_norm/moments_compute.cpp", {}, {}, {planes, ht * wt}, 1},
                {
                    "nn/ktt_group_norm/moments_writer.cpp",
                    {&s1, &s2},
                    {1, 2},
                    {planes},
                    2
                },
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
            const CoreRangeSet core(CoreRange(CoreCoord(0, 0)));
            ProgramDescriptor desc;
            for (
                auto [index, bytes] :
                std::initializer_list<std::pair<uint8_t, uint32_t>>{
                    {0, 128 * 64}, {1, 64}, {2, 128 * 64}, {3, 4096}, {4, 4096}
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
                {
                    &s1,
                    &s2,
                    weight ? weight : &s1,
                    bias ? bias : &s1,
                    static_cast<const ttnn::Tensor*>(&r.a),
                    static_cast<const ttnn::Tensor*>(&r.b),
                    static_cast<const ttnn::Tensor*>(&r.mean),
                    static_cast<const ttnn::Tensor*>(&r.rstd)
                }
            ) {
                const auto more = (
                    TensorAccessorArgs(*t->buffer()).get_compile_time_args()
                );
                ct.insert(ct.end(), more.begin(), more.end());
            }
            KernelDescriptor::RuntimeArgs args{
                {CoreCoord(0, 0),
                 {s1.buffer()->address(),
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
                  out == DataType::FLOAT32 ? 4u : 2u}}
            };
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
            uint32_t tensor = 0;
            for (uint32_t position = 0; position < 8; ++position) {
                const bool absent = (
                    (position == 2 && !weight) || (position == 3 && !bias)
                );
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
            {{0, 4, xf}, {1, 1, f32}, {2, 1, f32}, {16, 4, xf}},
            {
                {
                    "nn/ktt_group_norm/affine_reader.cpp",
                    {&x, &a, &b},
                    {0, 1, 2},
                    {planes, tiles, elem},
                    0
                },
                {"nn/ktt_group_norm/affine_compute.cpp", {}, {}, {planes, tiles}, 1},
                {"nn/ktt_group_norm/affine_writer.cpp", {&y}, {3}, {planes, tiles}, 2},
            },
            planes
        );
        return y;
    }

}  // namespace ktt::bridge::ktt_group_norm
