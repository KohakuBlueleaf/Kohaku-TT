// Kernel contract v1 helpers, the generic table kernels, and the boxed fallback that
// names every overload without a tt kernel.
#include "core/torch/contract.hpp"

#include "core/torch/complex.hpp"
#include "core/torch/mesh_rules.hpp"

#include <cstdlib>
#include <map>
#include <mutex>
#include <sstream>

#include <ATen/ExpandUtils.h>
#include <ATen/native/CPUFallback.h>
#include <torch/library.h>

namespace ktt {

    void unsupported(const std::string& op, const std::string& reason) {
        TORCH_CHECK_NOT_IMPLEMENTED(
            false, "KohakuTTUnsupportedOperator: ", op, ": ", reason
        );
    }

    bridge::HandlePtr dense(const std::string& op, const at::Tensor& t) {
        return resolve(op, t);
    }

    namespace {

        // The Nhwc or Flat storage base `t` covers whole at its sizes, else null
        // (nothing resolved).
        bridge::HandlePtr whole_base(const at::Tensor& t) {
            if (!t.is_privateuseone() || t.numel() == 0 || !t.is_contiguous() ||
                t.storage_offset() != 0) {
                return nullptr;
            }
            const auto& st = storage_of(t);
            if (st.base && st.dtype == t.scalar_type() &&
                bridge::format(st.base) != bridge::Format::Nchw &&
                bridge::format(st.base) != bridge::Format::HeadMajor &&
                st.sizes == to_vector(t.sizes())) {
                return st.base;
            }
            return nullptr;
        }

        // `h` cast on device to `dtype` (its format kept).
        bridge::HandlePtr cast_to(const bridge::HandlePtr& h, at::ScalarType dtype) {
            const auto cast = to_bridge(dtype);
            return bridge::dtype(h) == cast ? h : bridge::typecast(h, cast);
        }

        // whole_base() for ops that keep the format: Nhwc only under KTT_NHWC=1.
        bridge::HandlePtr kept_base(const at::Tensor& t) {
            auto h = whole_base(t);
            if (h && bridge::format(h) == bridge::Format::Nhwc &&
                !bridge::nhwc_propagation()) {
                return nullptr;
            }
            return h;
        }

    }  // namespace

    bridge::HandlePtr dense_any_layout(const std::string& op, const at::Tensor& t) {
        auto h = whole_base(t);
        return h ? h : dense(op, t);
    }

    bridge::HandlePtr dense_propagate(const std::string& op, const at::Tensor& t) {
        auto h = kept_base(t);
        return h ? h : dense(op, t);
    }

    bridge::HandlePtr dense_stored(const std::string& op, const at::Tensor& t) {
        bridge::layout_reader(op);
        auto h = head_stored(t);
        return h ? h : dense_propagate(op, t);
    }

    bridge::HandlePtr dense_propagate_as(
        const std::string& op,
        const at::Tensor& t,
        at::IntArrayRef shape,
        at::ScalarType dtype
    ) {
        if (t.sizes() != shape) {
            return dense_as(op, t.expand(shape), dtype);
        }
        return cast_to(dense_stored(op, t), dtype);
    }

    bool is_flat(const at::Tensor& t) {
        auto h = whole_base(t);
        return h && bridge::format(h) == bridge::Format::Flat;
    }

    at::Tensor wrap_flat(
        const std::string& op, const at::Tensor& rows, const at::Tensor& like
    ) {
        auto h = bridge::view_as(
            dense(op, rows),
            bridge::device_sizes_of(to_vector(like.sizes())),
            bridge::Format::Flat
        );
        return wrap_output(h, like.sizes(), rows.scalar_type());
    }

    at::Tensor& write_into(
        const std::string& op, at::Tensor& dst, const at::Tensor& result
    ) {
        // Complex: the result's fp32 pairs written into dst's.
        if (holds_complex(dst) || holds_complex(result)) {
            KTT_REQUIRE(
                holds_complex(dst) && dst.sizes() == result.sizes(),
                op,
                "complex out must be complex of the result's sizes"
            );
            KTT_REQUIRE(!dst.is_conj(), op, "writing into a conjugated view");
            copy_pairs(dst, result);
            return dst;
        }
        KTT_REQUIRE(
            dst.sizes() == result.sizes(),
            op,
            "out tensor of sizes ",
            dst.sizes(),
            " for a result of sizes ",
            result.sizes(),
            " (resizing out is not supported yet)"
        );
        KTT_REQUIRE(
            dst.scalar_type() == result.scalar_type(),
            op,
            "out dtype differs from result"
        );
        // A mesh-placed result is stored as placed (whole destination) or converted by
        // commit (partial destination).
        auto placed = mesh::placed_base(result);
        commit(op, dst, placed ? placed : dense_stored(op, result));
        return dst;
    }

    at::Tensor& into(const char* op, at::Tensor& out, const at::Tensor& result) {
        if (out.sizes() != result.sizes()) {
            out.resize_(result.sizes());
        }
        return write_into(op, out, result.to(out.scalar_type()));
    }

    at::Tensor& write_out(
        const std::string& op, at::Tensor& out, const at::Tensor& result
    ) {
        if (out.sizes() != result.sizes()) {
            out.resize_(result.sizes());
        }
        return write_into(op, out, result);
    }

    at::Tensor wrap_output(
        bridge::HandlePtr out, c10::IntArrayRef sizes, at::ScalarType dtype
    ) {
        return make_tt(std::move(out), sizes, dtype);
    }

    std::vector<int64_t> broadcast_shape(const at::Tensor& a, const at::Tensor& b) {
        auto shape = at::infer_size(a.sizes(), b.sizes());
        return std::vector<int64_t>(shape.begin(), shape.end());
    }

    at::ScalarType result_type(const at::Tensor& a, const at::Tensor& b) {
        return at::result_type(a, b);
    }

    bridge::HandlePtr dense_as(
        const std::string& op, const at::Tensor& t, at::ScalarType dtype
    ) {
        auto h = dense(op, t);
        return (
            to_bridge(t.scalar_type()) == to_bridge(dtype)
                ? h
                : bridge::typecast(h, to_bridge(dtype))
        );
    }

    namespace {

        bool is_compare(bridge::BinaryOp id) {
            using B = bridge::BinaryOp;
            return (
                id == B::Eq || id == B::Ne || id == B::Lt || id == B::Le ||
                id == B::Gt || id == B::Ge
            );
        }

        // TT-NN broadcasts this op's operands natively (binary_ng, rank <= 6).
        bool native_broadcast_op(bridge::BinaryOp id, size_t rank) {
            using B = bridge::BinaryOp;
            return (
                id != B::Pow && id != B::BitAnd && id != B::BitOr && id != B::BitXor &&
                id != B::Atan2 && rank <= 6
            );
        }

        // Ops whose result is floating even for integer inputs.
        bool floating_unary(bridge::UnaryOp id) {
            using U = bridge::UnaryOp;
            return (
                id != U::Neg && id != U::Abs && id != U::Relu && id != U::Sign &&
                id != U::Floor && id != U::Ceil && id != U::Trunc && id != U::Frac
            );
        }

        // A wrapped Python number (0-d CPU tensor) with its own type (int stays int).
        std::optional<at::Scalar> wrapped_scalar(const at::Tensor& t) {
            if (t.is_cpu() && t.dim() == 0) {
                return t.item();
            }
            return std::nullopt;
        }

        // dense_stored(), then cast on device to `dtype` (the format kept).
        bridge::HandlePtr propagate_as(
            const std::string& op, const at::Tensor& t, at::ScalarType dtype
        ) {
            return cast_to(dense_stored(op, t), dtype);
        }

        // `t` (aligned sizes `s`, the result's rank) as a head-major operand in
        // `dtype`; null when it must be expanded first (`broadcast` unset).
        bridge::HandlePtr head_operand(
            const char* op,
            const at::Tensor& t,
            const std::vector<int64_t>& s,
            at::ScalarType dtype,
            bool broadcast
        ) {
            const size_t r = s.size();
            if (!broadcast && t.sizes() != at::IntArrayRef(s)) {
                return nullptr;
            }
            const bool ranked = t.dim() == static_cast<int64_t>(r);
            if (ranked) {
                if (auto h = head_stored(t)) {
                    return cast_to(h, dtype);
                }
                // A whole storage: its head-major copy, cached per write epoch.
                const auto& st = storage_of(t);
                if (t.is_contiguous() && t.storage_offset() == 0 && st.base &&
                    st.dtype == t.scalar_type() && st.sizes == s &&
                    bridge::format(st.base) == bridge::Format::Nchw &&
                    bridge::placement(st.base).is_replicated() &&
                    to_bridge(st.dtype) == to_bridge(dtype)) {
                    return bridge::to_head_major(st.base);
                }
            }
            // A size-1 T or H: the head-major tensor holds the elements in order.
            if (s[r - 3] == 1 || s[r - 2] == 1) {
                const auto rows = (bridge::stored_sizes_of(
                    s, bridge::Format::HeadMajor
                ));
                return bridge::view_as(
                    dense_as(op, t.reshape(rows), dtype), s, bridge::Format::HeadMajor
                );
            }
            return bridge::to_head_major(dense_as(op, t.reshape(s), dtype));
        }

        // self and other in `dtype` (bf16 / fp32) as stored when one of them is a whole
        // non-Nchw storage (or a head view) of the result's sizes `shape`; nullopt
        // otherwise. `broadcast`: the op broadcasts operands of other sizes itself.
        std::optional<std::pair<bridge::HandlePtr, bridge::HandlePtr>> stored_operands(
            const char* op,
            const at::Tensor& self,
            const at::Tensor& other,
            const std::vector<int64_t>& shape,
            at::ScalarType dtype,
            bool broadcast
        ) {
            const auto cast = to_bridge(dtype);
            if (shape.size() < 3 ||
                (cast != bridge::DType::BFloat16 && cast != bridge::DType::Float32)) {
                return std::nullopt;
            }
            const at::IntArrayRef full(shape);
            auto stored = [&](const at::Tensor& t) -> bridge::HandlePtr {
                if (t.sizes() != full) {
                    return nullptr;
                }
                auto h = kept_base(t);
                return h ? h : head_stored(t);
            };
            auto x = stored(self), y = stored(other);
            if (!x && !y) {
                return std::nullopt;
            }
            // Nhwc and Flat operands decide the format before head-major ones.
            const auto major = bridge::Format::HeadMajor;
            const auto format =
                (x && y && bridge::format(x) == major ? bridge::format(y)
                                                      : bridge::format(x ? x : y));
            if (format == bridge::Format::Nhwc && shape.size() != 4) {
                return std::nullopt;
            }
            if (format == major) {
                auto operand = [&](const at::Tensor& t, const bridge::HandlePtr& h) {
                    if (h && bridge::format(h) == format) {
                        return cast_to(h, dtype);
                    }
                    std::vector<int64_t> s(shape.size() - t.dim(), 1);
                    s.insert(s.end(), t.sizes().begin(), t.sizes().end());
                    return head_operand(op, t, s, dtype, broadcast);
                };
                auto a = operand(self, x), b = operand(other, y);
                if (!a || !b) {
                    return std::nullopt;
                }
                return std::make_pair(a, b);
            }
            // The other operand: of the full sizes, converted (cached per write epoch);
            // one element, or an Nhwc [1, C, 1, 1] vector, read as the row it is stored
            // as.
            auto in_format = [&](const at::Tensor& t,
                                 const bridge::HandlePtr& h) -> bridge::HandlePtr {
                if (h && bridge::format(h) == format) {
                    return bridge::dtype(h) == cast ? h : bridge::typecast(h, cast);
                }
                if (t.sizes() == full) {
                    return bridge::to_format(dense_as(op, t, dtype), format);
                }
                std::vector<int64_t> s(shape.size() - t.dim(), 1);
                s.insert(s.end(), t.sizes().begin(), t.sizes().end());
                const int64_t n = t.numel();
                const bool vector =
                    (format == bridge::Format::Nhwc && s[0] == 1 && s[2] == 1 &&
                     s[3] == 1 && (s[1] == shape[1] || s[1] == 1));
                if (n != 1 && !vector) {
                    return nullptr;
                }
                const auto rows = bridge::stored_sizes_of(s, format);
                auto row = dense_as(op, t.reshape(rows), dtype);
                return bridge::view_as(row, s, format);
            };
            auto a = in_format(self, x), b = in_format(other, y);
            if (!a || !b) {
                return std::nullopt;
            }
            return std::make_pair(a, b);
        }

    }  // namespace

    at::Tensor unary_kernel(
        const char* op, bridge::UnaryOp id, const at::Tensor& self
    ) {
        if (holds_complex(self)) {
            return complex_unary(op, id, self);
        }
        auto dtype = self.scalar_type();
        const bool include_bool = true;
        if (floating_unary(id) && at::isIntegralType(dtype, include_bool)) {
            dtype = at::typeMetaToScalarType(at::get_default_dtype());
        }
        // Mesh-placed input: per chip on its shard (neg keeps a partial sum).
        if (auto base = mesh::placed_base(self)) {
            const auto dims = bridge::device_sizes(base);
            auto f = mesh::elementwise(1, dims.size());
            if (id == bridge::UnaryOp::Neg) {
                f.partial = mesh::Factors::Partial::All;
            }
            const auto cast = to_bridge(dtype);
            f.dtype = cast;
            auto out = mesh::run_placed(op, f, {base}, dims, [&](const auto& in) {
                auto h =
                    (bridge::dtype(in[0]) == cast ? in[0]
                                                  : bridge::typecast(in[0], cast));
                return bridge::unary(id, h);
            });
            if (out) {
                return wrap_output(*out, self.sizes(), dtype);
            }
        }
        return (wrap_output(
            bridge::unary(id, propagate_as(op, self, dtype)), self.sizes(), dtype
        ));
    }

    std::optional<at::Tensor> placed_pointwise(
        const char* op,
        const std::vector<at::Tensor>& operands,
        at::IntArrayRef shape,
        at::ScalarType dtype,
        const std::function<bridge::HandlePtr(const std::vector<bridge::HandlePtr>&)>&
            run
    ) {
        std::vector<bridge::HandlePtr> placed;
        bool any = false;
        for (const auto& t : operands) {
            if (t.sizes() != shape) {
                return std::nullopt;
            }
            auto h = mesh::placed_base(t);
            any = any || h;
            placed.push_back(h);
        }
        if (!any) {
            return std::nullopt;
        }
        const auto cast = to_bridge(dtype);
        for (size_t i = 0; i < operands.size(); ++i) {
            if (!placed[i]) {
                placed[i] = dense_as(op, operands[i], dtype);
            }
        }
        const auto dims = bridge::device_sizes_of(to_vector(shape));
        auto f = mesh::elementwise(static_cast<int>(operands.size()), dims.size());
        f.dtype = cast;
        auto out = mesh::run_placed(op, f, placed, dims, [&](const auto& in) {
            std::vector<bridge::HandlePtr> typed;
            for (const auto& h : in) {
                typed.push_back(
                    bridge::dtype(h) == cast ? h : bridge::typecast(h, cast)
                );
            }
            return run(typed);
        });
        if (!out) {
            return std::nullopt;
        }
        return wrap_output(*out, shape, dtype);
    }

    at::Tensor binary_scalar_kernel(
        const char* op,
        bridge::BinaryOp id,
        const at::Tensor& self,
        const at::Scalar& scalar
    ) {
        if (holds_complex(self) || scalar.isComplex()) {
            return complex_binary_scalar(op, id, self, scalar);
        }
        // Comparisons compute in the promoted type; division of integers is floating.
        auto dtype = at::result_type(self, scalar);
        if (id == bridge::BinaryOp::Div && at::isIntegralType(dtype, true)) {
            dtype = at::typeMetaToScalarType(at::get_default_dtype());
        }
        // Mesh-placed input: per chip on its shard (scaling keeps a partial sum).
        if (auto base = mesh::placed_base(self)) {
            const auto dims = bridge::device_sizes(base);
            auto f = mesh::elementwise(1, dims.size());
            if (id == bridge::BinaryOp::Mul || id == bridge::BinaryOp::Div) {
                f.partial = mesh::Factors::Partial::All;
            }
            const auto cast = to_bridge(dtype);
            f.dtype = cast;
            const double s = scalar.toDouble();
            auto out = mesh::run_placed(op, f, {base}, dims, [&](const auto& in) {
                auto h =
                    (bridge::dtype(in[0]) == cast ? in[0]
                                                  : bridge::typecast(in[0], cast));
                return bridge::binary_scalar(id, h, s);
            });
            if (out) {
                return (
                    wrap_output(*out, self.sizes(), is_compare(id) ? at::kBool : dtype)
                );
            }
        }
        auto h = (bridge::binary_scalar(
            id, propagate_as(op, self, dtype), scalar.toDouble()
        ));
        return wrap_output(h, self.sizes(), is_compare(id) ? at::kBool : dtype);
    }

    at::Tensor binary_kernel(
        const char* op,
        bridge::BinaryOp id,
        const at::Tensor& self,
        const at::Tensor& other,
        const at::Scalar& alpha
    ) {
        using B = bridge::BinaryOp;
        if (holds_complex(self) || holds_complex(other)) {
            return complex_binary(op, id, self, other, alpha);
        }
        const double a = alpha.toDouble();

        // A Python number on the right: one tensor-scalar op.
        if (auto s = wrapped_scalar(other)) {
            bool scaled = (id == B::Add || id == B::Sub) && a != 1.0;
            return (binary_scalar_kernel(
                op, id, self, scaled ? at::Scalar(s->toDouble() * a) : *s
            ));
        }

        // A Python number on the left (rsub, 2 / x, ...): rewritten around the tensor.
        if (auto wrapped = wrapped_scalar(self)) {
            const double s = wrapped->toDouble();
            auto dtype = at::result_type(self, other);
            if (id == B::Div && at::isIntegralType(dtype, true)) {
                dtype = at::typeMetaToScalarType(at::get_default_dtype());
            }
            auto b = propagate_as(op, other, dtype);
            bridge::HandlePtr h;
            switch (id) {
                case B::Add:
                case B::Mul:
                    h = (bridge::binary_scalar(
                        id, a == 1.0 ? b : bridge::binary_scalar(B::Mul, b, a), s
                    ));
                    break;
                case B::Sub:
                    h = (bridge::binary_scalar(
                        B::Add, bridge::binary_scalar(B::Mul, b, -a), s
                    ));
                    break;
                case B::Div:
                    h = (bridge::binary_scalar(
                        B::Mul, bridge::unary(bridge::UnaryOp::Reciprocal, b), s
                    ));
                    break;
                default:
                    unsupported(
                        op, "a Python number as the first operand is not supported"
                    );
            }
            return wrap_output(h, other.sizes(), dtype);
        }

        // Two tensors: broadcast (expanded views), promote, apply alpha to the second.
        KTT_REQUIRE(
            self.is_privateuseone() && other.is_privateuseone(),
            op,
            "operands on ",
            self.device(),
            " and ",
            other.device()
        );
        auto shape = broadcast_shape(self, other);
        auto dtype = at::result_type(self, other);
        if (id == B::Div && at::isIntegralType(dtype, true)) {
            dtype = at::typeMetaToScalarType(at::get_default_dtype());
        }
        auto out_dtype = is_compare(id) ? at::kBool : dtype;

        // A mesh-placed operand: per chip on the shards. Broadcasting operands need the
        // output's device rank (TT-NN broadcasts their size-1 dims).
        const bool same_shape =
            (self.sizes() == c10::IntArrayRef(shape) &&
             other.sizes() == c10::IntArrayRef(shape));
        if (same_shape || native_broadcast_op(id, shape.size())) {
            // Operands at the output's rank (leading size-1 dims), as views.
            auto aligned = [&](const at::Tensor& t) {
                if (t.dim() == static_cast<int64_t>(shape.size())) {
                    return t;
                }
                std::vector<int64_t> sizes(shape.size() - t.dim(), 1);
                sizes.insert(sizes.end(), t.sizes().begin(), t.sizes().end());
                return t.reshape(sizes);
            };
            const auto sa = aligned(self), oa = aligned(other);
            auto xs = mesh::placed_base(sa);
            auto ys = mesh::placed_base(oa);
            auto dims = bridge::device_sizes_of(shape);
            auto in_x = xs ? xs : (xs || ys ? dense(op, sa) : nullptr);
            auto in_y = ys ? ys : (xs || ys ? dense(op, oa) : nullptr);
            const bool ranks =
                (in_x && bridge::device_sizes(in_x).size() == dims.size() &&
                 bridge::device_sizes(in_y).size() == dims.size());
            if ((xs || ys) && ranks) {
                const auto cast = to_bridge(dtype);
                // A partial sum plus a replicated tensor: the sum's collective adds it.
                if (id == B::Add && a == 1.0 && same_shape) {
                    for (const bool swap : {false, true}) {
                        const auto& sum = swap ? in_y : in_x;
                        const auto& rest = swap ? in_x : in_y;
                        if (!bridge::placement(sum).has_partial() ||
                            bridge::dtype(sum) != cast || bridge::dtype(rest) != cast) {
                            continue;
                        }
                        if (auto out = bridge::reduce_add(sum, rest)) {
                            if (mesh::recording()) {
                                const auto at = bridge::placement(sum);
                                mesh::record(
                                    op,
                                    {bridge::placement(in_x), bridge::placement(in_y)},
                                    bridge::Placement{},
                                    {{swap ? size_t{1} : size_t{0}, at, {}, false}},
                                    "partial + replicated: the all-reduce adds the "
                                    "other input"
                                );
                            }
                            return wrap_output(out, shape, out_dtype);
                        }
                    }
                }
                auto f = mesh::broadcast(
                    {bridge::device_sizes(in_x), bridge::device_sizes(in_y)}, dims
                );
                if (id == B::Add || id == B::Sub) {
                    f.partial = mesh::Factors::Partial::All;
                } else if (id == B::Mul) {
                    f.partial = mesh::Factors::Partial::One;
                }
                f.dtype = cast;
                auto out = mesh::run_placed(
                    op, f, {in_x, in_y}, dims, [&](const auto& in) {
                        auto x =
                            (bridge::dtype(in[0]) == cast
                                 ? in[0]
                                 : bridge::typecast(in[0], cast));
                        auto y =
                            (bridge::dtype(in[1]) == cast
                                 ? in[1]
                                 : bridge::typecast(in[1], cast));
                        if (a != 1.0) {
                            y = bridge::binary_scalar(B::Mul, y, a);
                        }
                        return bridge::binary(id, x, y);
                    }
                );
                if (out) {
                    return wrap_output(*out, shape, out_dtype);
                }
            }
        }

        // A whole Flat, head-major (or, under KTT_NHWC=1, Nhwc) operand of the result's
        // sizes: the op as stored.
        bridge::layout_reader(op);
        if (auto p = stored_operands(
                op, self, other, shape, dtype, native_broadcast_op(id, shape.size())
            )) {
            auto [x, y] = *p;
            if (a != 1.0) {
                y = bridge::binary_scalar(B::Mul, y, a);
            }
            return wrap_output(bridge::binary(id, x, y), shape, out_dtype);
        }

        // Native broadcast (TT-NN binary_ng ops, rank <= 6): operands at the result's
        // rank with their size-1 dims.
        const bool native_broadcast = native_broadcast_op(id, shape.size());
        if (native_broadcast &&
            (self.sizes() != c10::IntArrayRef(shape) ||
             other.sizes() != c10::IntArrayRef(shape))) {
            auto aligned = [&](const at::Tensor& t) {
                std::vector<int64_t> sizes(shape.size() - t.dim(), 1);
                sizes.insert(sizes.end(), t.sizes().begin(), t.sizes().end());
                return t.reshape(sizes);
            };
            auto xb = dense_as(op, aligned(self), dtype);
            auto yb = dense_as(op, aligned(other), dtype);
            if (a != 1.0) {
                yb = bridge::binary_scalar(B::Mul, yb, a);
            }
            return wrap_output(bridge::binary(id, xb, yb), shape, out_dtype);
        }
        auto x = dense_as(op, self.expand(shape), dtype);
        auto y = dense_as(op, other.expand(shape), dtype);
        if (a != 1.0) {
            y = bridge::binary_scalar(B::Mul, y, a);
        }
        return (wrap_output(
            bridge::binary(id, x, y), shape, is_compare(id) ? at::kBool : dtype
        ));
    }

    namespace {

        std::mutex g_fallback_mutex;
        std::map<std::string, int64_t> g_fallback_counts;

        bool env_flag(const char* name, bool default_value) {
            const char* value = std::getenv(name);
            return value ? std::string(value) != "0" : default_value;
        }

        // Boxed fallback for overloads without a tt kernel: raises, naming the
        // arguments; with KTT_CPU_FALLBACK=1 runs the op on CPU instead (counted,
        // warned once per op unless KTT_FALLBACK_WARN=0). Ops returning aliases of
        // their inputs are never run on CPU.
        void no_kernel(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
            const auto& schema = op.schema();

            // _foreach_* ops: ATen's per-tensor loop (the CPU kernel) calling tt
            // kernels.
            if (schema.operator_name().name.rfind("aten::_foreach_", 0) == 0 &&
                op.hasKernelForDispatchKey(c10::DispatchKey::CPU)) {
                op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CPU), stack);
                return;
            }

            std::ostringstream args;
            auto arguments = torch::jit::last(*stack, schema.arguments().size());
            for (const auto& value : arguments) {
                if (value.isTensor() && value.toTensor().defined()) {
                    const auto& t = value.toTensor();
                    args << " " << t.device() << ":" << t.scalar_type() << t.sizes();
                }
            }
            std::string name = schema.operator_name().name;
            if (!schema.overload_name().empty()) {
                name += "." + schema.overload_name();
            }
            if (!env_flag("KTT_CPU_FALLBACK", false)) {
                unsupported(name, "no tt kernel; tensor args:" + args.str());
            }
            for (const auto& ret : schema.returns()) {
                if (ret.alias_info() && !ret.alias_info()->isWrite()) {
                    unsupported(
                        name, "returns a view; CPU fallback cannot keep the aliasing"
                    );
                }
            }

            bool first = false;
            {
                std::lock_guard<std::mutex> lock(g_fallback_mutex);
                first = g_fallback_counts[name]++ == 0;
            }
            if (first && env_flag("KTT_FALLBACK_WARN", true)) {
                TORCH_WARN(
                    "KohakuTT CPU FALLBACK: ",
                    name,
                    " has no tt kernel and runs on CPU (tensors copied both ways; "
                    "counts in "
                    "kohakutt_native.fallback_counts()). tensor args:",
                    args.str()
                );
            }
            at::native::cpu_fallback(op, stack);
        }

    }  // namespace

    std::vector<std::string> fallback_counts() {
        std::lock_guard<std::mutex> lock(g_fallback_mutex);
        std::vector<std::string> out;
        for (const auto& [name, count] : g_fallback_counts) {
            out.push_back(name + "=" + std::to_string(count));
        }
        return out;
    }

    void reset_fallback_counts() {
        std::lock_guard<std::mutex> lock(g_fallback_mutex);
        g_fallback_counts.clear();
    }

    at::Tensor exact_select(const at::Tensor& a, const at::Tensor& b, bool data_left) {
        const FidelityFloor exact(3);
        const auto& data = data_left ? a : b;
        const auto& select = data_left ? b : a;
        auto product = [&](const at::Tensor& part) {
            return data_left ? at::matmul(part, select) : at::matmul(select, part);
        };
        if (data.scalar_type() != at::kFloat) {
            return product(data.to(at::kFloat));
        }
        // fp32 = hi + mid + lo, each part exact in bf16 (8 significant bits).
        auto hi = data.to(at::kBFloat16).to(at::kFloat);
        auto rest = data - hi;
        auto mid = rest.to(at::kBFloat16).to(at::kFloat);
        auto lo = rest - mid;
        return product(hi) + (product(mid) + product(lo));
    }

    TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
        m.fallback(torch::CppFunction::makeFromBoxedFunction<&no_kernel>());
    }

}  // namespace ktt
