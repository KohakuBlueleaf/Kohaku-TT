// Bridge physical formats: the channels-last (Nhwc) storage of [N, C, H, W] handles and
// the cached conversions between it and Nchw, on the stream.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

#include <tt-metalium/bfloat16.hpp>

#include "core/device/handle.hpp"
#include "core/device/trace.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"

namespace ktt::bridge {
    namespace {

        std::mutex g_twin_mutex;
        std::atomic<int64_t> g_conversions{0};

        void check_nhwc(const HandlePtr& a) {
            if (a->shape.size() != 4 || detail::is_empty(a) ||
                !a->placement.is_replicated() ||
                (a->dtype != DType::BFloat16 && a->dtype != DType::Float32)) {
                throw std::invalid_argument(
                    "kohakutt_native: Nhwc storage needs a non-empty, replicated bf16 / "
                    "fp32 [N, C, H, W] tensor"
                );
            }
        }

        // `a` as `format` from its cache, built in the current write epoch; a capture
        // and its prepare pass neither read nor fill the cache.
        HandlePtr twin(const HandlePtr& a, Format format) {
            const bool in_capture = tracing() || capturing();
            const int64_t epoch = write_epoch();
            std::lock_guard lock(g_twin_mutex);
            if (a->twin && !in_capture && a->twin_epoch == epoch) {
                return a->twin;
            }
            auto out = convert_layout(a, format);
            if (!in_capture) {
                a->twin = out;
                a->twin_epoch = epoch;
            }
            return out;
        }

    }  // namespace

    Format format(const HandlePtr& handle) {
        return handle->format;
    }

    HandlePtr convert_layout(const HandlePtr& a, Format format) {
        if (a->format == format) {
            return a;
        }
        if (format == Format::Nhwc) {
            check_nhwc(a);
        }
        g_conversions++;
        const auto shape = a->shape;
        const int64_t bytes = (
            detail::padded_bytes(detail::stored_dims(shape, format), a->dtype)
        );
        return detail::timed_op(
            a->dtype,
            shape,
            [a, shape, bytes, format] {
                return (
                    format == Format::Nhwc ? detail::nhwc_of(a->tensor, shape, bytes)
                                           : detail::nchw_of(a->tensor, shape, bytes)
                );
            },
            format
        );
    }

    HandlePtr to_nchw(const HandlePtr& a) {
        return a->format == Format::Nchw ? a : twin(a, Format::Nchw);
    }

    HandlePtr to_nhwc(const HandlePtr& a) {
        if (a->format == Format::Nhwc) {
            return a;
        }
        check_nhwc(a);
        return twin(a, Format::Nhwc);
    }

    int64_t layout_conversions() {
        return g_conversions.load();
    }

    bool nhwc_propagation() {
        const char* v = std::getenv("KTT_NHWC");
        return v && std::string(v) == "1";
    }

    HandlePtr view_as(
        const HandlePtr& stored, const std::vector<int64_t>& shape, Format format
    ) {
        if (stored->shape != detail::stored_dims(shape, format) ||
            !stored->placement.is_replicated()) {
            throw std::invalid_argument(
                "kohakutt_native: view_as: the stored handle's shape does not match"
            );
        }
        auto out = detail::make_handle(shape, stored->dtype, format);
        detail::submit([out, stored] {
            out->tensor = stored->tensor;
        });
        return out;
    }

    double padding_max(const HandlePtr& a) {
        if (a->format != Format::Nhwc) {
            throw std::invalid_argument("kohakutt_native: padding_max of an Nchw handle");
        }
        double worst = 0;
        detail::run_sync([&] {
            const auto& t = a->tensor;
            const auto& logical = t.logical_shape();
            const auto& padded = t.padded_shape();
            // The padded tensor viewed as logical: every stored value, row-major.
            const auto all = ttnn::experimental::view(t, padded, padded);
            std::vector<float> values;
            if (a->dtype == DType::BFloat16) {
                for (const auto v : all.to_vector<bfloat16>()) {
                    values.push_back(static_cast<float>(v));
                }
            } else {
                values = all.to_vector<float>();
            }
            const uint32_t rows = logical[-2], cols = logical[-1];
            const uint32_t prows = padded[-2], pcols = padded[-1];
            for (size_t i = 0; i < values.size(); ++i) {
                const uint32_t r = (i / pcols) % prows, c = i % pcols;
                if (r >= rows || c >= cols) {
                    worst = std::max(worst, std::abs(static_cast<double>(values[i])));
                }
            }
        });
        return worst;
    }

    namespace detail {

        ttnn::Tensor nhwc_of(
            const ttnn::Tensor& x, const std::vector<int64_t>& shape, int64_t bytes
        ) {
            const int64_t n = shape[0], c = shape[1], h = shape[2], w = shape[3];
            // [N, H, W, C] (the permute pads tiles with zeros), then the rows merged
            // with zero tile padding (a view when W is tile-aligned).
            auto t = ttnn::permute(x, {0, 2, 3, 1}, activation_config(bytes), 0.0f);
            return ttnn::reshape(
                t, device_shape({1, 1, n * h * w, c}), std::nullopt, ttnn::PadValue(0.0f)
            );
        }

        ttnn::Tensor nchw_of(
            const ttnn::Tensor& y, const std::vector<int64_t>& shape, int64_t bytes
        ) {
            const int64_t n = shape[0], c = shape[1], h = shape[2], w = shape[3];
            auto t = ttnn::reshape(y, device_shape({n, h, w, c}));
            return ttnn::permute(t, {0, 3, 1, 2}, activation_config(bytes), 0.0f);
        }

        bool has_padding(const std::vector<int64_t>& dims) {
            const size_t r = dims.size();
            return r >= 2 && (dims[r - 1] % 32 != 0 || dims[r - 2] % 32 != 0);
        }

        ttnn::Tensor zero_padding(const ttnn::Tensor& t) {
            return ttnn::fill_implicit_tile_padding(t, ttnn::PadValue(0.0f));
        }

    }  // namespace detail
}  // namespace ktt::bridge
