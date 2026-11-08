// Kernel contract v1 (docs/design/kernel-contract.md): validate with KTT_REQUIRE, read
// inputs with dense(), compute through the bridge, return wrap_output().
#pragma once

#include <algorithm>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <ATen/ATen.h>

#include "core/torch/tt_tensor.hpp"
#include "core/device/tt_bridge.hpp"

namespace ktt {

    constexpr int kContractVersion = 1;

    // Raises KohakuTTUnsupportedOperator (a NotImplementedError in Python) for `op`.
    [[noreturn]] void unsupported(const std::string& op, const std::string& reason);

// Raises KohakuTTUnsupportedOperator for `op` unless `cond` holds.
#define KTT_REQUIRE(cond, op, ...)                         \
    do {                                                   \
        if (!(cond)) {                                     \
            ::ktt::unsupported(op, c10::str(__VA_ARGS__)); \
        }                                                  \
    } while (false)

    // The dense device handle of tt tensor `t` (views materialized).
    bridge::HandlePtr dense(const std::string& op, const at::Tensor& t);

    // A new tt tensor of `sizes` and `dtype` holding the kernel's output handle.
    at::Tensor wrap_output(
        bridge::HandlePtr out, c10::IntArrayRef sizes, at::ScalarType dtype
    );

    // Broadcast shape of a and b (PyTorch rules).
    std::vector<int64_t> broadcast_shape(const at::Tensor& a, const at::Tensor& b);

    // PyTorch type promotion of a and b.
    at::ScalarType result_type(const at::Tensor& a, const at::Tensor& b);

    // CPU fallback calls per overload ("name=count") since load or the last reset.
    std::vector<std::string> fallback_counts();
    void reset_fallback_counts();

    // Writes functional result `result` into `dst` (in-place and out= variants).
    at::Tensor& write_into(
        const std::string& op, at::Tensor& dst, const at::Tensor& result
    );

    // out= form: resizes `out` to `result`'s sizes if they differ, casts `result` to
    // out's dtype and writes it in.
    at::Tensor& into(const char* op, at::Tensor& out, const at::Tensor& result);

    // out= form without the cast: resizes `out` to `result`'s sizes if they differ,
    // then write_into.
    at::Tensor& write_out(
        const std::string& op, at::Tensor& out, const at::Tensor& result
    );

    // dense(), then cast on device to `dtype` when its physical dtype differs.
    bridge::HandlePtr dense_as(
        const std::string& op, const at::Tensor& t, at::ScalarType dtype
    );

    // Generic table kernels: one row per aten overload in the family files. They follow
    // PyTorch broadcasting and type promotion, and accept wrapped Python numbers.
    at::Tensor unary_kernel(const char* op, bridge::UnaryOp id, const at::Tensor& self);
    // An elementwise op of same-shape `operands` (sizes `shape`) with a mesh-placed
    // operand: `run` (local handles in `dtype` -> local result) per chip at the
    // operands' cheapest common placement, the result in `dtype`; nullopt otherwise.
    std::optional<at::Tensor> placed_pointwise(
        const char* op,
        const std::vector<at::Tensor>& operands,
        at::IntArrayRef shape,
        at::ScalarType dtype,
        const std::function<bridge::HandlePtr(const std::vector<bridge::HandlePtr>&)>&
            run
    );

    at::Tensor binary_scalar_kernel(
        const char* op,
        bridge::BinaryOp id,
        const at::Tensor& self,
        const at::Scalar& scalar
    );
    at::Tensor binary_kernel(
        const char* op,
        bridge::BinaryOp id,
        const at::Tensor& self,
        const at::Tensor& other,
        const at::Scalar& alpha
    );

    // a @ b where one operand is a selection matrix (0 / 1 entries, at most one 1 per
    // output) and the other data, with every value copied exactly: HiFi4, and fp32
    // data split into three bf16-exact parts (the FPU keeps about 19 bits of an fp32
    // operand). The result is fp32.
    at::Tensor exact_select(const at::Tensor& a, const at::Tensor& b, bool data_left);

    // For its lifetime, the calling thread's numerics scope with fidelity at least
    // rank `floor` (0..3: LoFi .. HiFi4); a strict scope is left as it is.
    class FidelityFloor {
    public:
        explicit FidelityFloor(int floor) {
            const auto n = bridge::numerics_current();
            const int fidelity = n.strict ? n.fidelity : std::max(n.fidelity, floor);
            bridge::numerics_push(fidelity, n.approx, n.strict, n.attention);
        }

        ~FidelityFloor() {
            bridge::numerics_pop();
        }

        FidelityFloor(const FidelityFloor&) = delete;
        FidelityFloor& operator=(const FidelityFloor&) = delete;
    };

}  // namespace ktt
