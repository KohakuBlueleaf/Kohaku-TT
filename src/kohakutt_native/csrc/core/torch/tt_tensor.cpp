// tt tensor construction: a StorageImpl whose DataPtr context is a TtStorage. The data
// pointer is the context address itself and is never dereferenced as host memory.
#include "core/torch/tt_tensor.hpp"

namespace ktt {
    namespace {

        void delete_storage(void* ctx) {
            delete static_cast<TtStorage*>(ctx);
        }

    }  // namespace

    at::ScalarType physical(at::ScalarType dtype) {
        switch (dtype) {
            case at::kBFloat16:
            case at::kBool:
                return at::kBFloat16;
            case at::kFloat:
                return at::kFloat;
            case at::kInt:
            case at::kLong:
            case at::kByte:
            case at::kChar:
            case at::kShort:
                return at::kInt;
            default:
                TORCH_CHECK_NOT_IMPLEMENTED(
                    false,
                    "KohakuTTUnsupportedOperator: dtype: supported are bfloat16, "
                    "float32, int32, bool, and int64 / int16 / int8 / uint8 (stored "
                    "as int32), got ",
                    dtype
                );
        }
    }

    bridge::DType to_bridge(at::ScalarType dtype) {
        switch (physical(dtype)) {
            case at::kBFloat16:
                return bridge::DType::BFloat16;
            case at::kFloat:
                return bridge::DType::Float32;
            default:
                return bridge::DType::Int32;
        }
    }

    std::vector<int64_t> to_vector(c10::IntArrayRef sizes) {
        return std::vector<int64_t>(sizes.begin(), sizes.end());
    }

    at::Tensor make_tt(
        bridge::HandlePtr handle, c10::IntArrayRef sizes, at::ScalarType dtype
    ) {
        auto* ctx = new TtStorage{std::move(handle), to_vector(sizes), dtype, {}};
        auto device = c10::Device(c10::DeviceType::PrivateUse1, 0);
        size_t nbytes = c10::multiply_integers(sizes) * c10::elementSize(dtype);
        c10::Storage storage(
            c10::Storage::use_byte_size_t(),
            nbytes,
            c10::DataPtr(ctx, ctx, &delete_storage, device),
            nullptr,  // allocator
            false     // resizable
        );
        auto tensor = (at::detail::make_tensor<c10::TensorImpl>(
            std::move(storage),
            c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
            caffe2::TypeMeta::fromScalarType(dtype)
        ));
        tensor.unsafeGetTensorImpl()->set_sizes_contiguous(sizes);
        return tensor;
    }

    TtStorage& storage_of(const at::Tensor& t) {
        TORCH_CHECK(t.is_privateuseone(), "tt: expected a tt tensor, got ", t.device());
        auto& st = *static_cast<TtStorage*>(t.storage().data_ptr().get_context());
        TORCH_CHECK(
            !bridge::stale(st.base),
            "kohakutt: this tt tensor belongs to a device that has been closed ",
            "(kohakutt.close()); it cannot be used after the device is opened again"
        );
        return st;
    }

    void ensure_base(TtStorage& st, at::ScalarType dtype, size_t nbytes) {
        if (st.base) {
            return;
        }
        int64_t numel = static_cast<int64_t>(nbytes / c10::elementSize(dtype));
        if (st.pending.defined()) {
            auto values = st.pending.view(dtype).to(physical(dtype)).contiguous();
            st.base = bridge::upload_auto(values.data_ptr(), {numel}, to_bridge(dtype));
            st.pending = at::Tensor();
        } else {
            st.base = bridge::empty_auto({numel}, to_bridge(dtype));
        }
        st.sizes = {numel};
        st.dtype = dtype;
    }

    bridge::HandlePtr block_base(const at::Tensor& t, bool& transposed) {
        transposed = false;
        if (!t.is_privateuseone() || t.dim() != 2 || t.storage_offset() != 0) {
            return nullptr;
        }
        auto& st = storage_of(t);
        if (!st.base || !bridge::is_block(bridge::dtype(st.base))) {
            return nullptr;
        }
        const std::vector<int64_t> sizes{t.size(0), t.size(1)};
        const std::vector<int64_t> swapped{t.size(1), t.size(0)};
        if (t.is_contiguous() && sizes == st.sizes) {
            return st.base;
        }
        if (t.t().is_contiguous() && swapped == st.sizes) {
            transposed = true;
            return st.base;
        }
        return nullptr;
    }

    at::Tensor storage_bytes(TtStorage& st, int64_t offset, int64_t count) {
        at::Tensor bytes;
        if (st.pending.defined()) {
            bytes = st.pending;
        } else if (st.base) {
            auto host = (at::empty(
                st.sizes, at::TensorOptions().dtype(physical(st.dtype))
            ));
            auto base = st.base;
            if (bridge::is_block(bridge::dtype(base))) {
                base = bridge::typecast(base, bridge::DType::BFloat16);
            }
            bridge::download(base, host.data_ptr());
            bytes = host.to(st.dtype).contiguous().view(at::kByte).flatten();
        } else {
            bytes = at::zeros({offset + count}, at::kByte);
        }
        return bytes.narrow(0, offset, count).clone();
    }

    void set_storage_bytes(TtStorage& st, const at::Tensor& bytes) {
        st.base = nullptr;
        st.sizes = {};
        st.dtype = at::kByte;
        st.pending = bytes.contiguous().clone();
        st.scalar.reset();
    }

    std::optional<double> known_scalar(const at::Tensor& t) {
        if (t.numel() != 1 || t.storage_offset() != 0) {
            return std::nullopt;
        }
        const auto& st = storage_of(t);
        return st.base && st.sizes == to_vector(t.sizes()) ? st.scalar : std::nullopt;
    }

    void set_known_scalar(const at::Tensor& t, double value) {
        auto& st = storage_of(t);
        if (t.numel() == 1 && t.storage_offset() == 0 &&
            st.sizes == to_vector(t.sizes())) {
            st.scalar = value;
        }
    }

    namespace {

        // Storages allocated by size: no base until first typed use (see ensure_base).
        struct TtAllocator final : c10::Allocator {
            c10::DataPtr allocate(size_t) override {
                auto* ctx = new TtStorage{};
                return {
                    ctx,
                    ctx,
                    &delete_storage,
                    c10::Device(c10::DeviceType::PrivateUse1, 0)
                };
            }

            c10::DeleterFnPtr raw_deleter() const override {
                return nullptr;
            }

            void copy_data(void*, const void*, std::size_t) const override {
                TORCH_CHECK(false, "tt: storages are not host-addressable");
            }
        };

        TtAllocator g_allocator;

    }  // namespace

    c10::Allocator* tt_allocator() {
        return &g_allocator;
    }

    REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &g_allocator);

}  // namespace ktt
