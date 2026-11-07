// User programs on tt tensors: torch.ops.ktt.run_program (kohakutt.kernels
// descriptions) and ktt::extension::run (programs of C++ extension libraries).
#include <ATen/ATen.h>
#include <torch/library.h>

#include "core/device/tt_bridge.hpp"
#include "core/torch/contract.hpp"
#include "core/torch/tt_tensor.hpp"
#include "ktt/extension/torch.hpp"

namespace ktt {
    namespace {

        // `launch(handles, shapes, dtypes)` over the dense handles of `inputs` and the
        // device form of `outputs`; returns the outputs as tt tensors.
        template <typename Launch>
        std::vector<at::Tensor> run_user(
            const char* op,
            at::TensorList inputs,
            const std::vector<extension::Output>& outputs,
            Launch launch
        ) {
            std::vector<bridge::HandlePtr> handles;
            for (const auto& t : inputs) {
                TORCH_CHECK(t.is_privateuseone(), op, ": inputs must be tt tensors");
                handles.push_back(dense(op, t));
            }
            std::vector<std::vector<int64_t>> shapes;
            std::vector<bridge::DType> dtypes;
            for (const auto& o : outputs) {
                shapes.push_back(o.sizes);
                dtypes.push_back(to_bridge(o.dtype));
            }
            const auto outs = launch(handles, shapes, dtypes);
            std::vector<at::Tensor> result;
            for (size_t i = 0; i < outs.size(); ++i) {
                result.push_back(
                    wrap_output(outs[i], outputs[i].sizes, outputs[i].dtype)
                );
            }
            return result;
        }

        // Runs `spec`; output i has the sizes out_sizes[sum(out_ranks[:i]) :
        // +out_ranks[i]] and ScalarType out_dtypes[i].
        std::vector<at::Tensor> run_program(
            at::TensorList inputs,
            const std::string& spec,
            c10::IntArrayRef out_ranks,
            c10::IntArrayRef out_sizes,
            c10::IntArrayRef out_dtypes
        ) {
            TORCH_CHECK(
                out_ranks.size() == out_dtypes.size(),
                "ktt::run_program: out_ranks and out_dtypes differ in length"
            );
            std::vector<extension::Output> outputs;
            size_t offset = 0;
            for (size_t i = 0; i < out_ranks.size(); ++i) {
                const auto rank = static_cast<size_t>(out_ranks[i]);
                TORCH_CHECK(
                    offset + rank <= out_sizes.size(),
                    "ktt::run_program: out_sizes too short"
                );
                outputs.push_back({
                    {out_sizes.begin() + offset, out_sizes.begin() + offset + rank},
                    static_cast<at::ScalarType>(out_dtypes[i])
                });
                offset += rank;
            }
            return run_user(
                "ktt::run_program",
                inputs,
                outputs,
                [&](const auto& handles, const auto& shapes, const auto& dtypes) {
                    return bridge::run_program(spec, handles, shapes, dtypes);
                }
            );
        }

    }  // namespace

    namespace extension {

        std::vector<at::Tensor> run(
            const std::string& name,
            at::TensorList inputs,
            const std::vector<Output>& outputs
        ) {
            return run_user(
                "ktt::extension::run",
                inputs,
                outputs,
                [&](const auto& handles, const auto& shapes, const auto& dtypes) {
                    return bridge::run_registered(name, handles, shapes, dtypes);
                }
            );
        }

    }  // namespace extension

    TORCH_LIBRARY_FRAGMENT(ktt, m) {
        m.def(
            "run_program(Tensor[] inputs, str spec, int[] out_ranks, int[] out_sizes, "
            "int[] out_dtypes) -> Tensor[]",
            &run_program
        );
    }

}  // namespace ktt
