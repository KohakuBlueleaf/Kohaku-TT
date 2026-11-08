// Bridge convolution: TT-NN's sliding-window conv2d on channels-last activations.
#include <map>
#include <set>
#include <string>

#include "aten/nn/ktt_conv/program.hpp"
#include "core/device/handle.hpp"
#include "core/device/numerics.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/core/to_layout/to_layout_op.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

namespace ktt::bridge {
    namespace {

        // Prepared weights per (weight handle, conv geometry); the handle is held
        // weakly, so a freed or rebound weight never matches.
        struct Prepared {
            std::weak_ptr<Handle> weight;
            ttnn::Tensor tensor;
        };

        std::map<std::pair<const Handle*, std::string>, Prepared>& prepared() {
            // Never destroyed: device tensors must not be freed after the device
            // closes at exit.
            static auto& cache = *new std::map<std::pair<const Handle*, std::string>, Prepared>;
            return cache;
        }

        uint32_t u32(int64_t v) {
            return static_cast<uint32_t>(v);
        }

        // Stride-1 single-group convs with C a multiple of 32 run Kohaku-TT's
        // implicit-GEMM conv (aten/nn/ktt_conv); KTT_OWN_CONV=0: TT-NN's conv2d.
        bool own_conv_enabled() {
            static const bool on = [] {
                const char* v = std::getenv("KTT_OWN_CONV");
                return !(v && std::string(v) == "0");
            }();
            return on;
        }

        // True unless KTT_CONV_LAYOUT=0 (TT-NN permute / pad / tilize around ktt_conv
        // instead of its layout kernels).
        bool own_layout_enabled() {
            static const bool on = [] {
                const char* v = std::getenv("KTT_CONV_LAYOUT");
                return !(v && std::string(v) == "0");
            }();
            return on;
        }

        // ktt_conv: channels-last zero-padded input rows [N Hp Wp, C] (row-major), the
        // tap-major weight [KH KW C, O] (cached per weight handle: no size in it),
        // output [N OH Wp, O] cropped to OW columns and back to [N, O, OH, OW].
        ttnn::Tensor own_conv(
            const HandlePtr& x,
            const HandlePtr& w,
            const ktt_conv::Geometry& g,
            int64_t ph,
            int64_t pw,
            int64_t ow,
            const ttnn::ComputeKernelConfig& config,
            int64_t bytes
        ) {
            using tt::tt_metal::Layout;
            const bool own_layout =
                (own_layout_enabled() && x->tensor.layout() == Layout::TILE &&
                 x->tensor.dtype() == tt::tt_metal::DataType::BFLOAT16);
            ttnn::Tensor rows;
            if (own_layout) {
                rows = ktt_conv::to_rows(x->tensor, g, u32(ph), u32(pw));
            } else {
                auto nhwc = ttnn::to_layout(
                    ttnn::permute(
                        x->tensor,
                        ttsl::SmallVector<int64_t>{0, 2, 3, 1},
                        tt::tt_metal::MemoryConfig{}
                    ),
                    Layout::ROW_MAJOR
                );
                if (ph > 0 || pw > 0) {
                    ttsl::SmallVector<std::array<uint32_t, 2>> widths{
                        {0, 0}, {u32(ph), u32(ph)}, {u32(pw), u32(pw)}, {0, 0}
                    };
                    nhwc = ttnn::pad(nhwc, widths, 0.0f);
                }
                rows = ttnn::reshape(nhwc, ttnn::Shape({g.n * g.hp * g.wp, g.c}));
            }

            auto& cache = prepared();
            const auto key = std::make_pair(w.get(), std::string("own"));
            auto it = cache.find(key);
            if (it == cache.end() || it->second.weight.expired()) {
                auto taps = ttnn::permute(
                    w->tensor, ttsl::SmallVector<int64_t>{2, 3, 1, 0}
                );
                auto matrix = ttnn::reshape(
                    taps, ttnn::Shape({g.kh * g.kw * g.c, g.o})
                );
                it = cache.insert_or_assign(key, Prepared{w, matrix}).first;
            }

            auto out = ktt_conv::run(rows, it->second.tensor, g, config);
            if (own_layout) {
                return (
                    ktt_conv::to_nchw(out, g, u32(ow), detail::activation_config(bytes))
                );
            }
            auto grid = ttnn::reshape(
                ttnn::to_layout(out, Layout::ROW_MAJOR),
                ttnn::Shape({g.n, g.oh, g.wp, g.o})
            );
            if (u32(ow) != g.wp) {
                grid = ttnn::slice(
                    grid,
                    ttsl::SmallVector<int32_t>{0, 0, 0, 0},
                    ttsl::SmallVector<int32_t>{
                        static_cast<int32_t>(g.n),
                        static_cast<int32_t>(g.oh),
                        static_cast<int32_t>(ow),
                        static_cast<int32_t>(g.o)
                    },
                    ttsl::SmallVector<int32_t>{1, 1, 1, 1}
                );
            }
            auto nchw = ttnn::permute(grid, ttsl::SmallVector<int64_t>{0, 3, 1, 2});
            return (
                ttnn::to_layout(
                    nchw, Layout::TILE, std::nullopt, detail::activation_config(bytes)
                )
            );
        }

    }  // namespace

    HandlePtr conv2d(
        const HandlePtr& x,
        const HandlePtr& w,
        std::array<int64_t, 2> stride,
        std::array<int64_t, 2> padding,
        std::array<int64_t, 2> dilation,
        int64_t groups
    ) {
        const int64_t n = x->shape[0], c = x->shape[1], h = x->shape[2],
                      wd = x->shape[3];
        const int64_t o = w->shape[0], kh = w->shape[2], kw = w->shape[3];
        const int64_t oh = (
            (h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) / stride[0] + 1
        );
        const int64_t ow = (
            (wd + 2 * padding[1] - dilation[1] * (kw - 1) - 1) / stride[1] + 1
        );
        const auto config = numerics::compute_config(numerics::effective(), true);
        const std::vector<int64_t> shape{n, o, oh, ow};
        const int64_t bytes = detail::padded_bytes(shape, x->dtype);
        const ktt_conv::Geometry g{
            u32(n),
            u32(c),
            u32(h + 2 * padding[0]),
            u32(wd + 2 * padding[1]),
            u32(o),
            u32(kh),
            u32(kw),
            u32(dilation[0]),
            u32(dilation[1]),
            u32(oh)
        };
        if (own_conv_enabled() && groups == 1 && stride[0] == 1 && stride[1] == 1 &&
            c % 32 == 0 && ktt_conv::columns_per_core(g)) {
            const int64_t ph = padding[0], pw = padding[1];
            return detail::timed_op(x->dtype, shape, [=] {
                return own_conv(x, w, g, ph, pw, ow, config, bytes);
            });
        }
        // Kernel geometry (prepared weights of the height-sharded config depend only on
        // it) and the full geometry (TT-NN's own config: its weights depend on H, W).
        std::string kernel_key;
        for (
            int64_t v :
            {
                c,
                o,
                kh,
                kw,
                stride[0],
                stride[1],
                padding[0],
                padding[1],
                dilation[0],
                dilation[1],
                groups
            }
        ) {
            kernel_key += std::to_string(v) + ',';
        }
        const std::string full_key =
            (kernel_key + std::to_string(n) + ',' + std::to_string(h) + ',' +
             std::to_string(wd));
        const auto dtype = detail::to_ttnn(x->dtype);
        return detail::timed_op(x->dtype, shape, [=] {
            // Channels last in DRAM: TT-NN's DRAM path slices large activations.
            auto nhwc = ttnn::permute(
                x->tensor,
                ttsl::SmallVector<int64_t>{0, 2, 3, 1},
                tt::tt_metal::MemoryConfig{}
            );
            // Shapes without a valid height-sharded config (never retried).
            static auto& unpinned = *new std::set<std::string>;
            auto run = [&](bool pinned) {
                auto& cache = prepared();
                const auto key = (
                    std::make_pair(
                        w.get(), (pinned ? "h:" + kernel_key : "a:" + full_key)
                    )
                );
                auto it = cache.find(key);
                const bool hit = it != cache.end() && !it->second.weight.expired();
                ttnn::Tensor weight;
                if (hit) {
                    weight = it->second.tensor;
                } else {
                    weight = (
                        ttnn::to_layout(
                            ttnn::operations::core::from_device(w->tensor),
                            tt::tt_metal::Layout::ROW_MAJOR
                        )
                    );
                }
                ttnn::Conv2dConfig conv_config;
                conv_config.weights_dtype = tt::tt_metal::DataType::BFLOAT16;
                conv_config.act_block_h_override = 32;
                if (pinned) {
                    conv_config.shard_layout = (
                        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED
                    );
                }
                auto result = ttnn::conv2d(
                    nhwc,
                    weight,
                    detail::device(),
                    u32(c),
                    u32(o),
                    u32(n),
                    u32(h),
                    u32(wd),
                    {u32(kh), u32(kw)},
                    {u32(stride[0]), u32(stride[1])},
                    std::array<uint32_t, 2>{u32(padding[0]), u32(padding[1])},
                    {u32(dilation[0]), u32(dilation[1])},
                    u32(groups),
                    dtype,
                    std::nullopt,
                    conv_config,
                    config,
                    tt::tt_metal::MemoryConfig{},
                    std::nullopt,
                    false,  // return_output_dim
                    true    // return_weights_and_bias
                );
                auto& [out, weights] = std::get<std::tuple<
                    ttnn::Tensor,
                    std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>>>>(result);
                if (!hit) {
                    cache[key] = Prepared{w, std::get<0>(weights)};
                }
                return out;
            };
            ttnn::Tensor out;
            bool done = false;
            if (unpinned.count(full_key) == 0) {
                try {
                    out = run(true);
                    done = true;
                } catch (const std::exception& e) {
                    // No height-sharded config or slicing for this shape: TT-NN's
                    // own config from now on (L1_SMALL exhaustion is retried by
                    // timed_op instead).
                    if (detail::is_l1_small_full(e)) {
                        throw;
                    }
                    unpinned.insert(full_key);
                }
            }
            if (!done) {
                out = run(false);
            }
            // [1, 1, N OH OW, O] -> [N, OH, OW, O] -> [N, O, OH, OW], tiled (a
            // row-major detour measured slower cold).
            auto rows = ttnn::reshape(
                ttnn::to_layout(out, tt::tt_metal::Layout::TILE),
                ttnn::Shape({u32(n), u32(oh), u32(ow), u32(o)})
            );
            return ttnn::permute(
                rows,
                ttsl::SmallVector<int64_t>{0, 3, 1, 2},
                detail::activation_config(bytes)
            );
        });
    }

}  // namespace ktt::bridge
