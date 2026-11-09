// Bridge nearest upsampling: Kohaku-TT's gather kernel (ktt_upsample) on Nchw, TT-NN's
// upsample on Nhwc.
#include "aten/nn/ktt_upsample/program.hpp"
#include "core/device/handle.hpp"
#include "ttnn/operations/core/to_layout/to_layout_op.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/pool/upsample/upsample.hpp"

namespace ktt::bridge {
    namespace {

        // Nhwc x (bf16, C % 32 == 0) by kh, kw: TT-NN's upsample of [N, H, W, C]
        // (row-major unless W % 32 == 0), tiled back as [1, 1, N H' W', C].
        HandlePtr upsample_nhwc(const HandlePtr& x, int64_t kh, int64_t kw) {
            const int64_t n = x->shape[0], c = x->shape[1], h = x->shape[2],
                          w = x->shape[3];
            const std::vector<int64_t> shape{n, c, h * kh, w * kw};
            const std::vector<int64_t> rows{1, 1, n * h * kh * w * kw, c};
            const int64_t bytes = detail::padded_bytes(rows, x->dtype);
            return detail::timed_op(
                x->dtype,
                shape,
                [=] {
                    using tt::tt_metal::Layout;
                    const auto grid = detail::device_shape({n, h, w, c});
                    auto in = (
                        w % 32 == 0 ? ttnn::reshape(x->tensor, grid)
                                    : ttnn::reshape(
                                          ttnn::to_layout(x->tensor, Layout::ROW_MAJOR),
                                          grid
                                      )
                    );
                    auto up = ttnn::upsample(
                        in,
                        std::array<int, 2>{static_cast<int>(kh), static_cast<int>(kw)},
                        "nearest",
                        tt::tt_metal::MemoryConfig{}
                    );
                    auto flat = ttnn::reshape(up, detail::device_shape(rows));
                    auto out = ttnn::to_layout(
                        flat, Layout::TILE, std::nullopt, detail::activation_config(bytes)
                    );
                    return detail::has_padding(rows) ? detail::zero_padding(out) : out;
                },
                Format::Nhwc
            );
        }

    }  // namespace

    HandlePtr upsample_nearest(const HandlePtr& x, int64_t kh, int64_t kw) {
        if (x->format == Format::Nhwc) {
            if (x->dtype == DType::BFloat16 && x->shape[1] % 32 == 0) {
                return upsample_nhwc(x, kh, kw);
            }
            return upsample_nearest(to_nchw(x), kh, kw);
        }
        auto shape = x->shape;
        shape[shape.size() - 2] *= kh;
        shape.back() *= kw;
        const int64_t bytes = detail::padded_bytes(shape, x->dtype);
        return detail::timed_op(x->dtype, shape, [=] {
            return (
                ktt_upsample::run(
                    x->tensor,
                    static_cast<uint32_t>(kh),
                    static_cast<uint32_t>(kw),
                    detail::activation_config(bytes)
                )
            );
        });
    }

    HandlePtr pixel_shuffle(const HandlePtr& x, int64_t r) {
        auto shape = x->shape;
        const size_t rank = shape.size();
        shape[rank - 3] /= r * r;
        shape[rank - 2] *= r;
        shape[rank - 1] *= r;
        const int64_t bytes = detail::padded_bytes(shape, x->dtype);
        return detail::timed_op(x->dtype, shape, [=] {
            return (
                ktt_upsample::shuffle(
                    x->tensor,
                    static_cast<uint32_t>(r),
                    detail::activation_config(bytes)
                )
            );
        });
    }

}  // namespace ktt::bridge
