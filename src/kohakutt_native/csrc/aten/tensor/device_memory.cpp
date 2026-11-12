// Bridge memory ops: host upload / download and device allocation (TILE, DRAM
// interleaved), on the stream.
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>

#include "core/device/handle.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/full/full.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

namespace ktt::bridge {
    namespace {

        using detail::to_ttnn;

        // TILE layout in interleaved DRAM at the device shape of `shape`.
        tt::tt_metal::TensorSpec tile_spec(
            const std::vector<int64_t>& shape, DType dtype
        ) {
            return tt::tt_metal::TensorSpec(
                detail::device_shape(shape),
                tt::tt_metal::TensorLayout(
                    to_ttnn(dtype),
                    tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                    tt::tt_metal::MemoryConfig{}
                )
            );
        }

        size_t volume(const std::vector<int64_t>& shape) {
            size_t n = 1;
            for (int64_t d : shape) {
                n *= static_cast<size_t>(d);
            }
            return n;
        }

        size_t element_size(DType dtype) {
            return dtype == DType::BFloat16 ? 2 : 4;
        }

        // Row-major `src` of dims `from` [.., A, B, D] (elements of `size` bytes) into
        // `dst` as [.., B, A, D].
        void swap_rows(
            const uint8_t* src,
            uint8_t* dst,
            const std::vector<int64_t>& from,
            size_t size
        ) {
            const size_t r = from.size();
            const size_t a = from[r - 3], b = from[r - 2], row = from[r - 1] * size;
            const size_t lead = volume(from) / std::max<size_t>(a * b * from[r - 1], 1);
            for (size_t l = 0; l < lead; ++l) {
                for (size_t i = 0; i < a; ++i) {
                    for (size_t j = 0; j < b; ++j) {
                        std::memcpy(
                            dst + ((l * b + j) * a + i) * row,
                            src + ((l * a + i) * b + j) * row,
                            row
                        );
                    }
                }
            }
        }

    }  // namespace

    HandlePtr upload(
        const void* data, const std::vector<int64_t>& shape, DType dtype, Format format
    ) {
        if (is_block(dtype)) {
            throw std::runtime_error("kohakutt_native: block floats are device-only");
        }
        if (capturing()) {
            throw std::runtime_error(
                "kohakutt_native: host upload inside a trace capture"
            );
        }
        detail::counters().uploads++;
        const size_t n = volume(shape);

        // The host data is copied now, so the caller's buffer may go away at once.
        auto bytes = std::make_shared<std::vector<uint8_t>>(n * element_size(dtype));
        std::memcpy(bytes->data(), data, bytes->size());
        const auto dims = detail::device_dims(shape);
        auto out = detail::make_handle(dims, dtype, format);
        const auto stored = detail::stored_dims(dims, format);
        detail::submit([out, bytes, stored, dtype, n] {
            // from_span converts row-major host data to TILE and writes it to the
            // device (Flat: the same values in order, tiled as its stored rows).
            auto spec = tile_spec(stored, dtype);
            auto put = [&]<typename T>(const T*) {
                auto* p = reinterpret_cast<const T*>(bytes->data());
                out->tensor = (ttnn::Tensor::from_span(
                    ttsl::Span<const T>(p, n), spec, detail::device()
                ));
            };
            switch (dtype) {
                case DType::BFloat16:
                    put(static_cast<const bfloat16*>(nullptr));
                    break;
                case DType::Float32:
                    put(static_cast<const float*>(nullptr));
                    break;
                case DType::Int32:
                    put(static_cast<const int32_t*>(nullptr));
                    break;
                default:
                    break;
            }
        });
        return out;
    }

    void download(const HandlePtr& handle, void* dst, int chip) {
        if (chip < 0 || chip >= mesh_size()) {
            throw std::out_of_range(
                "kohakutt_native: no mesh chip " + std::to_string(chip)
            );
        }
        if (capturing()) {
            throw std::runtime_error(
                "kohakutt_native: host download inside a trace capture"
            );
        }
        // Flat holds the values in their logical order: read as stored.
        if (handle->format == Format::Nhwc) {
            return download(to_nchw(handle), dst, chip);
        }
        // Head-major: the stored rows read, then reordered on the host.
        if (handle->format == Format::HeadMajor) {
            const auto stored = detail::stored_dims(handle->shape, handle->format);
            const size_t size = element_size(handle->dtype);
            std::vector<uint8_t> rows(volume(stored) * size);
            download(as_stored(handle), rows.data(), chip);
            swap_rows(rows.data(), static_cast<uint8_t*>(dst), stored, size);
            return;
        }
        detail::counters().downloads++;
        if (!handle->placement.is_replicated()) {
            return detail::download_placed(handle, dst);
        }

        // to_vector reads the device tensor back as row-major host data (after every
        // earlier op on the stream); on a multi-chip mesh, chip `chip`'s replica.
        detail::run_sync([&] {
            ttnn::Tensor tensor =
                (mesh_size() > 1
                     ? ttnn::distributed::get_device_tensors(handle->tensor)[chip]
                     : handle->tensor);
            auto get = [&]<typename T>(T*) {
                auto values = tensor.to_vector<T>();
                std::copy(values.begin(), values.end(), static_cast<T*>(dst));
            };
            switch (handle->dtype) {
                case DType::BFloat16:
                    return get(static_cast<bfloat16*>(nullptr));
                case DType::Float32:
                    return get(static_cast<float*>(nullptr));
                case DType::Int32:
                    return get(static_cast<int32_t*>(nullptr));
                default:
                    throw std::runtime_error(
                        "kohakutt_native: download of a block float (typecast first)"
                    );
            }
        });
    }

    HandlePtr empty(const std::vector<int64_t>& shape, DType dtype, Format format) {
        const auto dims = detail::device_dims(shape);
        auto out = detail::make_handle(dims, dtype, format);
        const auto stored = detail::stored_dims(dims, format);
        detail::submit([out, stored, dtype] {
            auto spec = tile_spec(stored, dtype);
            out->tensor = ttnn::empty(
                spec.logical_shape(),
                to_ttnn(dtype),
                tt::tt_metal::Layout::TILE,
                detail::device(),
                tt::tt_metal::MemoryConfig{}
            );
        });
        return out;
    }

    namespace {

        // A NaN-filled tensor uploaded from host (TT-NN's fill turns NaN into inf).
        HandlePtr upload_nan(const std::vector<int64_t>& shape, DType dtype) {
            const size_t n = volume(shape);
            const float nan = std::numeric_limits<float>::quiet_NaN();
            if (dtype == DType::BFloat16) {
                std::vector<bfloat16> values(n, bfloat16(nan));
                return upload(values.data(), shape, dtype);
            }
            std::vector<float> values(n, nan);
            return upload(values.data(), shape, dtype);
        }

    }  // namespace

    namespace {

        // Filled float tensors per (device shape, dtype, value bits): TT-NN's fill is
        // slow
        // (~0.7 GB/s), so each fill is built once and later fills are device copies of
        // it.
        using FillKey = std::tuple<std::vector<int64_t>, DType, uint64_t>;
        constexpr size_t kCachedFills = 64;

        std::map<FillKey, HandlePtr>& fill_cache() {
            static auto* map = new std::map<FillKey, HandlePtr>();
            return *map;
        }

        HandlePtr fill_uncached(
            const std::vector<int64_t>& shape, double value, DType dtype
        );

        // A copy of the cached fill of `dims` (device shape) for float dtypes;
        // `shared`: the cached fill itself (read-only).
        HandlePtr cached_fill(
            const std::vector<int64_t>& dims,
            double value,
            DType dtype,
            bool shared = false
        ) {
            uint64_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            auto& cache = fill_cache();
            const FillKey key{dims, dtype, bits};
            if (auto it = cache.find(key); it != cache.end()) {
                return shared ? it->second : clone(it->second);
            }
            // No new entries in a capture (its fill runs only at replay), and none
            // dropped while a trace is alive (it may read one); at most kCachedFills.
            if (capturing()) {
                return fill_uncached(dims, value, dtype);
            }
            if (cache.size() >= kCachedFills) {
                if (traces_alive()) {
                    return fill_uncached(dims, value, dtype);
                }
                cache.erase(cache.begin());
            }
            auto slot = fill_uncached(dims, value, dtype);
            cache[key] = slot;
            return shared ? slot : clone(slot);
        }

    }  // namespace

    HandlePtr shared_full(
        const std::vector<int64_t>& shape, double value, DType dtype
    ) {
        if (dtype == DType::Int32 || std::isnan(value) || volume(shape) == 0) {
            return full(shape, value, dtype);
        }
        return cached_fill(detail::device_dims(shape), value, dtype, true);
    }

    HandlePtr full(const std::vector<int64_t>& shape, double value, DType dtype) {
        if (std::isnan(value) && dtype != DType::Int32) {
            return upload_nan(shape, dtype);
        }
        if (dtype != DType::Int32 && volume(shape) > 0) {
            return cached_fill(detail::device_dims(shape), value, dtype);
        }
        return fill_uncached(shape, value, dtype);
    }

    namespace {

        // A TILE tensor of `shape` holding `value`, written by a device program (no
        // host write, so a trace can record it); block floats are cast from bf16.
        ttnn::Tensor device_full(const ttnn::Shape& shape, double value, DType dtype) {
            if (is_block(dtype)) {
                return ttnn::typecast(
                    device_full(shape, value, DType::BFloat16), to_ttnn(dtype)
                );
            }
            if (shape.volume() == 0) {
                return ttnn::empty(
                    shape,
                    to_ttnn(dtype),
                    tt::tt_metal::Layout::TILE,
                    detail::device(),
                    tt::tt_metal::MemoryConfig{}
                );
            }
            const ttsl::SmallVector<uint32_t> dims(shape.cbegin(), shape.cend());
            const std::variant<float, int> typed = dtype == DType::Int32
                ? std::variant<float, int>(static_cast<int>(value))
                : std::variant<float, int>(static_cast<float>(value));
            return ttnn::moreh_full(dims, typed, detail::device(), to_ttnn(dtype));
        }

        HandlePtr fill_uncached(
            const std::vector<int64_t>& shape, double value, DType dtype
        ) {
            auto device_shape = detail::device_shape(shape);
            // TT-NN's full writes from the host, which no trace can record: a device
            // program while traces are alive (a capture's prepare pass included).
            const bool on_device = traces_alive();
            return detail::timed_op(dtype, detail::device_dims(shape), [=] {
                if (on_device) {
                    return device_full(device_shape, value, dtype);
                }
                auto fill = [&](auto typed_value) {
                    return (ttnn::full(
                        device_shape,
                        typed_value,
                        to_ttnn(dtype),
                        tt::tt_metal::Layout::TILE,
                        std::ref(*detail::device())
                    ));
                };
                return (
                    dtype == DType::Int32 ? fill(static_cast<int>(value))
                                          : fill(static_cast<float>(value))
                );
            });
        }

    }  // namespace

    HandlePtr full_like(const HandlePtr& like, double value) {
        // A Flat or Fold `like`: its stored rows filled, kept in its format.
        if (like->format == Format::Flat || like->format == Format::Fold) {
            const auto rows = detail::stored_dims(like->shape, like->format);
            return view_as(full(rows, value, like->dtype), like->shape, like->format);
        }
        if (std::isnan(value) && like->dtype != DType::Int32) {
            return upload_nan(like->shape, like->dtype);
        }
        // A placed `like` (R / S axes) keeps its placement: each chip fills its shard.
        if (!like->placement.is_replicated() && !like->placement.has_partial() &&
            !capturing() && !detail::is_empty(like)) {
            return full_placed(like->shape, value, like->dtype, like->placement);
        }
        if (like->dtype == DType::Int32) {
            // TT-NN's int32 fill writes from host (not trace-safe): like * 0 + value on
            // device instead (exact for any int32 value).
            auto zeros = binary_scalar(BinaryOp::Mul, like, 0.0);
            return binary_scalar(BinaryOp::Add, zeros, value);
        }
        if (detail::is_empty(like)) {
            return detail::empty_of(like, like->dtype);
        }
        return cached_fill(like->shape, value, like->dtype);
    }

    namespace {

        // arange handles per (n, dtype): created on first (eager) use, reused inside
        // captures. Never destroyed at exit (after TT-Metal's teardown); close() clears
        // it.
        std::map<std::pair<int64_t, DType>, HandlePtr>& arange_cache() {
            static auto* map = new std::map<std::pair<int64_t, DType>, HandlePtr>();
            return *map;
        }

    }  // namespace

    namespace detail {

        void clear_constants() {
            arange_cache().clear();
            fill_cache().clear();
        }

    }  // namespace detail

    HandlePtr arange(int64_t n, DType dtype) {
        auto& slot = arange_cache()[{n, dtype}];
        if (slot) {
            return slot;
        }
        slot = detail::timed_op(dtype, detail::device_dims({1, n}), [=] {
            auto row = (ttnn::arange(
                0,
                n,
                1,
                to_ttnn(dtype),
                std::ref(*detail::device()),
                ttnn::DRAM_MEMORY_CONFIG,
                tt::tt_metal::Layout::TILE
            ));
            return ttnn::reshape(row, detail::device_shape({1, n}));
        });
        return slot;
    }

    std::vector<int64_t> device_sizes(const HandlePtr& handle) {
        return handle->shape;
    }

    std::vector<int64_t> device_sizes_of(const std::vector<int64_t>& logical) {
        return detail::device_dims(logical);
    }

    DType dtype(const HandlePtr& handle) {
        return handle->dtype;
    }

}  // namespace ktt::bridge
