// aten range kernel for tt: arange (cached integer ranges; float ranges from them).

#include <torch/library.h>

#include <algorithm>
#include <cmath>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        at::Tensor& tt_arange_out(
            const at::Scalar& start,
            const at::Scalar& end,
            const at::Scalar& step,
            at::Tensor& out
        ) {
            const char* op = "arange.start_out";
            const bool integral =
                (start.isIntegral(false) && end.isIntegral(false) &&
                 step.isIntegral(false));
            if (!integral) {
                // start + i * step for i < ceil((end - start) / step), in fp32.
                const double a = start.toDouble();
                const double b = end.toDouble();
                const double s = step.toDouble();
                KTT_REQUIRE(s != 0.0, op, "step must be nonzero");
                const auto n = (static_cast<int64_t>(
                    std::max(0.0, std::ceil((b - a) / s))
                ));
                if (out.numel() != n) {
                    out.resize_({n});
                }
                if (n == 0) {
                    return out;
                }
                auto i = at::empty({n}, out.options().dtype(at::kFloat));
                tt_arange_out(0, n, 1, i);
                return into(op, out, i * s + a);
            }
            const int64_t a = start.toLong(), b = end.toLong(), s = step.toLong();
            KTT_REQUIRE(s != 0, op, "step must be nonzero");
            const int64_t n = std::max<int64_t>(0, (b - a + s - (s > 0 ? 1 : -1)) / s);
            if (out.numel() != n) {
                out.resize_({n});
            }
            if (n == 0) {
                return out;
            }
            auto dtype = out.scalar_type();
            auto h = bridge::arange_range(a, b, s, to_bridge(dtype));
            commit(op, out, h);
            return out;
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("arange.start_out", &tt_arange_out);
    }

}  // namespace ktt
