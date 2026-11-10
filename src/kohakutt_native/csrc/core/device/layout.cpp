// Bridge physical formats: channels-last (Nhwc) [N, C, H, W] handles, Flat handles
// (small trailing dims merged into one row), HeadMajor handles (dims -3 and -2
// swapped), and the cached conversions between them and Nchw, on the stream.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>

#include <tt-metalium/bfloat16.hpp>

#include "core/device/handle.hpp"
#include "core/device/trace.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"

namespace ktt::bridge {
    namespace {

        std::mutex g_twin_mutex;
        std::atomic<int64_t> g_conversions{0};
        std::atomic<int64_t> g_expansions{0};

        // Expansions above this many bytes warn (once per shape).
        constexpr int64_t kExpansionWarnBytes = 64ll << 20;

        bool float_dtype(DType d) {
            return d == DType::BFloat16 || d == DType::Float32;
        }

        void check_target(const HandlePtr& a, Format format) {
            const bool rank =
                (format == Format::Nhwc ? a->shape.size() == 4 : a->shape.size() >= 3);
            if (!rank || detail::is_empty(a) || !a->placement.is_replicated() ||
                !float_dtype(a->dtype)) {
                throw std::invalid_argument(
                    format == Format::Nhwc
                        ? "kohakutt_native: Nhwc storage needs a non-empty, replicated "
                          "bf16 / fp32 [N, C, H, W] tensor"
                        : "kohakutt_native: Flat and head-major storage need a "
                          "non-empty, replicated bf16 / fp32 tensor of rank >= 3"
                );
            }
        }

        std::mutex g_reads_mutex;
        std::map<std::string, int64_t> g_head_major_reads;

        std::string& reader() {
            thread_local std::string op;
            return op;
        }

        // Warns (once per shape, dtype and format) of a Flat or head-major tensor read
        // as a tile form of more than KTT_FLAT_WARN_BYTES (default 64 MB) bytes.
        void warn_expansion(
            const std::vector<int64_t>& shape, DType dtype, int64_t bytes, Format from
        ) {
            const char* env = std::getenv("KTT_FLAT_WARN_BYTES");
            if (bytes <= (env ? std::atoll(env) : kExpansionWarnBytes)) {
                return;
            }
            static std::mutex mutex;
            static std::set<std::vector<int64_t>> warned;
            std::lock_guard lock(mutex);
            auto key = shape;
            key.push_back(static_cast<int64_t>(dtype));
            key.push_back(static_cast<int64_t>(from));
            if (!warned.insert(key).second) {
                return;
            }
            std::string dims;
            for (const auto d : shape) {
                dims += (dims.empty() ? "" : ", ") + std::to_string(d);
            }
            std::fprintf(
                stderr,
                "kohakutt: %s read a [%s] %s tensor stored %s as its tile form "
                "(%lld MB); these reads are counted in %s\n",
                from == Format::Flat || reader().empty() ? "an op" : reader().c_str(),
                dims.c_str(),
                dtype == DType::Float32 ? "fp32" : "bf16",
                from == Format::Flat ? "Flat" : "head-major",
                static_cast<long long>(bytes >> 20),
                from == Format::Flat ? "layout_expansions" : "head_major_reads()"
            );
        }

        // `a` as `format` from its cache, built in the current write epoch; a capture
        // and its prepare pass neither read nor fill the cache.
        HandlePtr twin(const HandlePtr& a, Format format) {
            const bool in_capture = tracing() || capturing();
            const int64_t epoch = write_epoch();
            std::lock_guard lock(g_twin_mutex);
            if (a->twin && a->twin->format == format && !in_capture &&
                a->twin_epoch == epoch) {
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
        if (a->format != Format::Nchw && format != Format::Nchw) {
            return convert_layout(convert_layout(a, Format::Nchw), format);
        }
        if (format != Format::Nchw) {
            check_target(a, format);
        }
        g_conversions++;
        const auto shape = a->shape;
        const auto from = a->format;
        const int64_t bytes = (detail::padded_bytes(
            detail::stored_dims(shape, format), a->dtype
        ));
        if (from == Format::Flat) {
            g_expansions++;
            warn_expansion(shape, a->dtype, bytes, from);
        }
        if (from == Format::HeadMajor) {
            count_head_major_read();
            warn_expansion(shape, a->dtype, bytes, from);
        }
        return detail::timed_op(
            a->dtype,
            shape,
            [a, shape, bytes, from, format] {
                if (format == Format::Nhwc) {
                    return detail::nhwc_of(a->tensor, shape, bytes);
                }
                if (from == Format::Nhwc) {
                    return detail::nchw_of(a->tensor, shape, bytes);
                }
                if (format == Format::HeadMajor || from == Format::HeadMajor) {
                    return detail::swap_heads(
                        a->tensor, shape, bytes, format == Format::HeadMajor
                    );
                }
                // Flat <-> Nchw: the same elements in order, re-tiled; placed by the
                // output's own size.
                const auto to = detail::stored_dims(shape, format);
                return ttnn::reshape(
                    a->tensor,
                    detail::device_shape(to),
                    detail::activation_config(bytes)
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
        if (a->format != Format::Nchw) {
            return to_nhwc(to_nchw(a));
        }
        check_target(a, Format::Nhwc);
        return twin(a, Format::Nhwc);
    }

    HandlePtr to_flat(const HandlePtr& a) {
        if (a->format == Format::Flat) {
            return a;
        }
        if (a->format != Format::Nchw) {
            return to_flat(to_nchw(a));
        }
        check_target(a, Format::Flat);
        return twin(a, Format::Flat);
    }

    HandlePtr to_head_major(const HandlePtr& a) {
        if (a->format == Format::HeadMajor) {
            return a;
        }
        if (a->format != Format::Nchw) {
            return to_head_major(to_nchw(a));
        }
        check_target(a, Format::HeadMajor);
        return twin(a, Format::HeadMajor);
    }

    bool head_major_rule(const std::vector<int64_t>& dims, DType dtype) {
        const size_t r = dims.size();
        if (r < 3 || !float_dtype(dtype) || dims[r - 3] < 32 || dims[r - 1] < 32 ||
            dims[r - 2] < 1) {
            return false;
        }
        const int64_t tiled = detail::padded_bytes(dims, dtype);
        const int64_t major = (detail::padded_bytes(
            detail::stored_dims(dims, Format::HeadMajor), dtype
        ));
        if (major <= 0 || 2 * major > tiled) {
            return false;
        }
        const char* v = std::getenv("KTT_HEAD_MAJOR");
        return !(v && v[0] == '0' && v[1] == '\0');
    }

    void layout_reader(std::string_view op) {
        reader().assign(op);
    }

    void count_head_major_read() {
        std::lock_guard lock(g_reads_mutex);
        g_head_major_reads[reader().empty() ? "(unnamed)" : reader()]++;
    }

    std::vector<std::string> head_major_reads() {
        std::lock_guard lock(g_reads_mutex);
        std::vector<std::string> out;
        for (const auto& [name, count] : g_head_major_reads) {
            out.push_back(name + "=" + std::to_string(count));
        }
        return out;
    }

    void reset_head_major_reads() {
        std::lock_guard lock(g_reads_mutex);
        g_head_major_reads.clear();
    }

    HandlePtr to_format(const HandlePtr& a, Format format) {
        switch (format) {
            case Format::HeadMajor:
                return to_head_major(a);
            case Format::Nhwc:
                return to_nhwc(a);
            case Format::Flat:
                return to_flat(a);
            default:
                return to_nchw(a);
        }
    }

    int64_t layout_conversions() {
        return g_conversions.load();
    }

    int64_t layout_expansions() {
        return g_expansions.load();
    }

    std::vector<int64_t> stored_sizes_of(
        const std::vector<int64_t>& dims, Format format
    ) {
        return detail::stored_dims(dims, format);
    }

    HandlePtr as_stored(const HandlePtr& a) {
        if (a->format == Format::Nchw) {
            return a;
        }
        auto out = (detail::make_handle(
            detail::stored_dims(a->shape, a->format), a->dtype
        ));
        // The view shares a's buffer: its bytes are a's.
        detail::track_bytes(-out->bytes);
        out->bytes = 0;
        out->owner = a;
        detail::submit([out, a] {
            out->tensor = a->tensor;
        });
        return out;
    }

    bool aliases(const HandlePtr& h, const HandlePtr& base) {
        return h == base || (h && h->owner && h->owner == base);
    }

    HandlePtr tap_major(const HandlePtr& w, bool blocked) {
        const int64_t o = w->shape[0], c = w->shape[1];
        const int64_t taps = detail::stored_dims(w->shape, Format::Flat)[1] / c;
        const int64_t rows = blocked ? (c + 31) / 32 * 32 * taps : c * taps;
        return detail::timed_op(w->dtype, {rows, o}, [w, blocked] {
            return blocked ? detail::block_tap_major(*w) : detail::tap_major(*w);
        });
    }

    HandlePtr flat_from_taps(const HandlePtr& m, const std::vector<int64_t>& shape) {
        const auto rows = detail::stored_dims(shape, Format::Flat);
        if (shape.size() < 3 || m->format != Format::Nchw || m->shape != rows ||
            !m->placement.is_replicated() || !float_dtype(m->dtype)) {
            throw std::invalid_argument(
                "kohakutt_native: flat_from_taps needs a replicated bf16 / fp32 "
                "[d0, taps d1] matrix for a shape of rank >= 3"
            );
        }
        return detail::timed_op(
            m->dtype,
            shape,
            [m, shape] {
                return detail::flat_from_taps(m->tensor, shape);
            },
            Format::Flat
        );
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
            throw std::invalid_argument(
                "kohakutt_native: padding_max of a non-Nhwc handle"
            );
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
                t,
                device_shape({1, 1, n * h * w, c}),
                std::nullopt,
                ttnn::PadValue(0.0f)
            );
        }

        ttnn::Tensor nchw_of(
            const ttnn::Tensor& y, const std::vector<int64_t>& shape, int64_t bytes
        ) {
            const int64_t n = shape[0], c = shape[1], h = shape[2], w = shape[3];
            auto t = ttnn::reshape(y, device_shape({n, h, w, c}));
            return ttnn::permute(t, {0, 3, 1, 2}, activation_config(bytes), 0.0f);
        }

        ttnn::Tensor swap_heads(
            const ttnn::Tensor& x,
            const std::vector<int64_t>& shape,
            int64_t bytes,
            bool to_major
        ) {
            const size_t r = shape.size();
            const auto mc = activation_config(bytes);
            // A size-1 T or H: the same elements in order, re-tiled.
            if (shape[r - 3] == 1 || shape[r - 2] == 1) {
                const auto to = to_major ? stored_dims(shape, Format::HeadMajor)
                                         : shape;
                return ttnn::reshape(x, device_shape(to), mc);
            }
            ttsl::SmallVector<int64_t> order(r);
            for (size_t k = 0; k < r; ++k) {
                order[k] = static_cast<int64_t>(k);
            }
            std::swap(order[r - 3], order[r - 2]);
            return ttnn::permute(x, order, mc);
        }

        bool has_padding(const std::vector<int64_t>& dims) {
            const size_t r = dims.size();
            return r >= 2 && (dims[r - 1] % 32 != 0 || dims[r - 2] % 32 != 0);
        }

        ttnn::Tensor zero_padding(const ttnn::Tensor& t) {
            return ttnn::fill_implicit_tile_padding(t, ttnn::PadValue(0.0f));
        }

        Format auto_format(const std::vector<int64_t>& dims, DType dtype) {
            const char* v = std::getenv("KTT_FLAT");
            const size_t r = dims.size();
            // dims[1] == 1: a depthwise conv weight, which ktt_conv2 reads in tile
            // form.
            if ((v && std::string(v) == "0") || r < 3 || !float_dtype(dtype) ||
                dims[0] < 32 || dims[1] == 1 || dims[r - 1] >= 32 ||
                dims[r - 2] >= 32) {
                return Format::Nchw;
            }
            int64_t n = 1;
            for (const auto d : dims) {
                n *= d;
            }
            const int64_t tiled = padded_bytes(dims, dtype);
            const int64_t flat = padded_bytes(stored_dims(dims, Format::Flat), dtype);
            return n > 0 && 4 * flat <= tiled ? Format::Flat : Format::Nchw;
        }

        namespace {

            // The reorders' tensors go to DRAM: their per-tap planes pad to whole
            // tiles, far larger than the weight they come from.
            const tt::tt_metal::MemoryConfig kDram{};

            // w's [C KH KW, O] matrix: the transposed Flat rows (the Nchw tile form
            // re-tiled first).
            ttnn::Tensor transposed_matrix(const Handle& w) {
                const auto rows = stored_dims(w.shape, Format::Flat);
                const auto matrix =
                    (w.format == Format::Flat
                         ? w.tensor
                         : ttnn::reshape(w.tensor, device_shape(rows), kDram));
                return ttnn::transpose(matrix, -2, -1, kDram);
            }

        }  // namespace

        ttnn::Tensor tap_major(const Handle& w) {
            const int64_t o = w.shape[0], c = w.shape[1];
            const int64_t taps = stored_dims(w.shape, Format::Flat)[1] / c;
            auto t = transposed_matrix(w);
            if (taps == 1) {
                return t;
            }
            auto grid = ttnn::reshape(t, device_shape({c, taps, o}), kDram);
            auto by_tap = ttnn::permute(grid, {1, 0, 2}, kDram);
            return ttnn::reshape(by_tap, device_shape({taps * c, o}), kDram);
        }

        ttnn::Tensor flat_from_taps(
            const ttnn::Tensor& m, const std::vector<int64_t>& shape
        ) {
            const int64_t o = shape[0], c = shape[1];
            const int64_t taps = stored_dims(shape, Format::Flat)[1] / c;
            if (taps == 1) {
                return m;
            }
            auto grid = ttnn::reshape(m, device_shape({o, taps, c}), kDram);
            auto by_channel = ttnn::transpose(grid, -2, -1, kDram);
            return ttnn::reshape(by_channel, device_shape({o, c * taps}), kDram);
        }

        ttnn::Tensor block_tap_major(const Handle& w) {
            const int64_t o = w.shape[0], c = w.shape[1];
            const int64_t taps = stored_dims(w.shape, Format::Flat)[1] / c;
            const int64_t blocks = (c + 31) / 32;
            auto t = transposed_matrix(w);
            if (c % 32 != 0) {
                const auto extra = static_cast<uint32_t>((blocks * 32 - c) * taps);
                const ttsl::SmallVector<std::array<uint32_t, 2>> rows{
                    {0, extra}, {0, 0}
                };
                t = ttnn::pad(t, rows, 0.0f);
            }
            auto grid = ttnn::reshape(t, device_shape({blocks, 32, taps, o}));
            auto by_block = ttnn::permute(grid, {0, 2, 1, 3});
            return ttnn::reshape(by_block, device_shape({blocks * taps * 32, o}));
        }

    }  // namespace detail
}  // namespace ktt::bridge
