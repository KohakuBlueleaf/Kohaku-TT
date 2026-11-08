// aten logical reductions for tt: any / all (whole tensor and per dim).

#include <torch/library.h>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        // any / all as bool: any = sum(x != 0) > 0, all = sum(x == 0) == 0 (empty:
        // any false, all true).
        at::Tensor logical_reduce(
            const at::Tensor& self, at::OptionalIntArrayRef dims, bool keep, bool all
        ) {
            auto flags = (all ? at::eq(self, 0) : at::ne(self, 0)).to(at::kFloat);
            std::vector<int64_t> d = dims ? to_vector(*dims) : std::vector<int64_t>{};
            auto m = flags.sum(d, keep);
            return all ? at::eq(m, 0) : at::gt(m, 0);
        }

        at::Tensor& tt_any_all_out(const at::Tensor& self, at::Tensor& out) {
            return (
                into(
                    "any.all_out", out, logical_reduce(self, std::nullopt, false, false)
                )
            );
        }

        at::Tensor& tt_all_all_out(const at::Tensor& self, at::Tensor& out) {
            return (
                into(
                    "all.all_out", out, logical_reduce(self, std::nullopt, false, true)
                )
            );
        }

        at::Tensor& tt_any_out(
            const at::Tensor& self, int64_t dim, bool keep, at::Tensor& out
        ) {
            return into(
                "any.out",
                out,
                logical_reduce(self, std::vector<int64_t>{dim}, keep, false)
            );
        }

        at::Tensor& tt_all_out(
            const at::Tensor& self, int64_t dim, bool keep, at::Tensor& out
        ) {
            return into(
                "all.out",
                out,
                logical_reduce(self, std::vector<int64_t>{dim}, keep, true)
            );
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("any.all_out", &tt_any_all_out);
        m.impl("all.all_out", &tt_all_all_out);
        m.impl("any.out", &tt_any_out);
        m.impl("all.out", &tt_all_out);
    }

}  // namespace ktt
