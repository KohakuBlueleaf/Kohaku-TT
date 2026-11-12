// User programs on tt tensors: torch.ops.ktt.run_program, program ops with
// remembered launches (ktt.program_op) and ktt::extension::run (C++ extensions).
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "core/device/tt_bridge.hpp"
#include "core/torch/contract.hpp"
#include "core/torch/tt_tensor.hpp"
#include "ktt/extension/torch.hpp"

namespace ktt {
    namespace {

        // The tensors the running program-op call writes in place (its schema's `(a!)`
        // arguments), on this thread.
        std::vector<const c10::TensorImpl*>& written() {
            thread_local std::vector<const c10::TensorImpl*> w;
            return w;
        }

        // The dense handle of input `t` of `op`; for a tensor the call writes in place,
        // its own base with the write recorded (written_base).
        bridge::HandlePtr input_handle(const std::string& op, const at::Tensor& t) {
            TORCH_CHECK(t.is_privateuseone(), op, ": inputs must be tt tensors");
            const auto& w = written();
            if (std::find(w.begin(), w.end(), t.unsafeGetTensorImpl()) != w.end()) {
                auto h = written_base(t);
                TORCH_CHECK(
                    h,
                    op,
                    ": an input written in place must be a whole contiguous bf16, fp32 "
                    "or int32 tt tensor"
                );
                return h;
            }
            return dense(op, t);
        }

        // `launch(handles, shapes, dtypes)` over the dense handles of `inputs` and the
        // device form of `outputs`; returns the outputs as tt tensors.
        template <typename Launch>
        std::vector<at::Tensor> run_user(
            const std::string& op,
            at::TensorList inputs,
            const std::vector<extension::Output>& outputs,
            Launch launch
        ) {
            std::vector<bridge::HandlePtr> handles;
            for (const auto& t : inputs) {
                handles.push_back(input_handle(op, t));
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

        // One remembered call of a program op: the program, the op tensors it reads
        // (indices), its outputs and its per-call values.
        struct Launch {
            bridge::UserProgramPtr program;
            std::vector<size_t> positions;
            std::vector<extension::Output> outputs;
            std::vector<std::vector<int64_t>> shapes;
            std::vector<bridge::DType> dtypes;
            std::vector<int64_t> values;
        };

        // Launches by call key; emptied beyond kMaxLaunches.
        struct Launches {
            std::mutex mutex;
            std::unordered_map<std::string, std::shared_ptr<const Launch>> map;
        };

        constexpr size_t kMaxLaunches = 4096;

        Launches& launches() {
            static Launches all;
            return all;
        }

        // The program-op call running its first-call op on this thread: its key and
        // tensors (for the ktt.run_program_first inside to remember), until taken.
        struct Pending {
            bool active = false;
            std::string key;
            std::vector<const c10::TensorImpl*> tensors;
            // Outputs the op returns; 0: one list of any length.
            size_t returns = 0;
        };

        Pending& pending() {
            thread_local Pending p;
            return p;
        }

        void append(std::string& key, char tag, int64_t value) {
            key.push_back(tag);
            key.append(reinterpret_cast<const char*>(&value), sizeof(value));
        }

        void append(std::string& key, const at::Tensor& t) {
            append(key, 'T', static_cast<int64_t>(t.scalar_type()));
            append(key, 'r', t.dim());
            for (const int64_t s : t.sizes()) {
                append(key, 's', s);
            }
        }

        // Appends argument `v` to `key` (values, and the dtype and sizes of tensors)
        // and its tensors to `tensors`; false for a type that has no key.
        bool append(
            std::string& key, const c10::IValue& v, std::vector<at::Tensor>& tensors
        ) {
            if (v.isTensor()) {
                const auto& t = v.toTensor();
                if (!t.defined()) {
                    return false;
                }
                append(key, t);
                tensors.push_back(t);
            } else if (v.isTensorList()) {
                const auto list = v.toTensorList();
                append(key, 'L', static_cast<int64_t>(list.size()));
                for (size_t i = 0; i < list.size(); ++i) {
                    at::Tensor t = list.get(i);
                    if (!t.defined()) {
                        return false;
                    }
                    append(key, t);
                    tensors.push_back(std::move(t));
                }
            } else if (v.isInt()) {
                append(key, 'i', v.toInt());
            } else if (v.isBool()) {
                append(key, 'b', v.toBool());
            } else if (v.isDouble()) {
                int64_t bits = 0;
                const double d = v.toDouble();
                std::memcpy(&bits, &d, sizeof(bits));
                append(key, 'd', bits);
            } else if (v.isString()) {
                const auto& s = v.toStringRef();
                append(key, 'S', static_cast<int64_t>(s.size()));
                key.append(s);
            } else if (v.isIntList()) {
                const auto list = v.toIntList();
                append(key, 'I', static_cast<int64_t>(list.size()));
                for (size_t i = 0; i < list.size(); ++i) {
                    append(key, 'i', list.get(i));
                }
            } else if (v.isNone()) {
                key.push_back('n');
            } else {
                return false;
            }
            return true;
        }

        // Runs `launch` over the op tensors; returns its outputs.
        std::vector<at::Tensor> run(
            const Launch& launch, const std::vector<at::Tensor>& tensors
        ) {
            static const std::string op = "program op";
            std::vector<bridge::HandlePtr> handles;
            handles.reserve(launch.positions.size());
            for (const size_t p : launch.positions) {
                handles.push_back(input_handle(op, tensors[p]));
            }
            const auto outs = bridge::run_program(
                launch.program, handles, launch.shapes, launch.dtypes, launch.values
            );
            std::vector<at::Tensor> result;
            result.reserve(outs.size());
            for (size_t i = 0; i < outs.size(); ++i) {
                result.push_back(wrap_output(
                    outs[i], launch.outputs[i].sizes, launch.outputs[i].dtype
                ));
            }
            return result;
        }

        // The tt kernel of a program op: runs the launch remembered for the call's
        // values, tensor dtypes and sizes and numerics scope, else the first-call op.
        class ProgramOp final : public c10::OperatorKernel {
        public:
            ProgramOp(int64_t id, c10::OperatorHandle first)
                : id_(id), first_(std::move(first)) {
            }

            void operator()(
                const c10::OperatorHandle& op,
                c10::DispatchKeySet,
                torch::jit::Stack* stack
            ) {
                const auto& schema = op.schema();
                const size_t count = schema.arguments().size();
                // Arguments written in place: their own bases, the writes recorded, the
                // version counters bumped after the launch.
                std::vector<at::Tensor> writes;
                {
                    const auto args = torch::jit::last(*stack, count);
                    for (size_t i = 0; i < count; ++i) {
                        const auto* alias = schema.arguments()[i].alias_info();
                        if (alias && alias->isWrite() && args[i].isTensor()) {
                            writes.push_back(args[i].toTensor());
                        }
                    }
                }

                struct Writing {
                    std::vector<const c10::TensorImpl*> saved;

                    explicit Writing(const std::vector<at::Tensor>& ts)
                        : saved(written()) {
                        written().clear();
                        for (const auto& t : ts) {
                            written().push_back(t.unsafeGetTensorImpl());
                        }
                    }

                    ~Writing() {
                        written() = saved;
                    }
                } writing(writes);

                auto bump = [&] {
                    for (const auto& t : writes) {
                        if (!t.is_inference()) {
                            t.unsafeGetTensorImpl()->bump_version();
                        }
                    }
                };
                const auto n = bridge::numerics_current();
                std::string key;
                key.reserve(160);
                append(key, 'o', id_);
                append(
                    key,
                    'n',
                    n.fidelity | (n.attention << 8) | (int64_t{n.approx} << 16) |
                        (int64_t{n.strict} << 17)
                );
                std::vector<at::Tensor> tensors;
                bool keyed = true;
                for (const auto& v : torch::jit::last(*stack, count)) {
                    keyed = keyed && append(key, v, tensors);
                }
                std::shared_ptr<const Launch> hit;
                if (keyed) {
                    auto& all = launches();
                    const std::lock_guard lock(all.mutex);
                    if (const auto it = all.map.find(key); it != all.map.end()) {
                        hit = it->second;
                    }
                }
                const auto& returns = schema.returns();
                const bool list = returns.size() == 1 &&
                    returns[0].type()->kind() == c10::TypeKind::ListType;
                if (hit) {
                    auto outs = run(*hit, tensors);
                    bump();
                    torch::jit::drop(*stack, count);
                    if (list) {
                        stack->emplace_back(
                            c10::List<at::Tensor>(c10::ArrayRef<at::Tensor>(outs))
                        );
                    } else {
                        for (auto& out : outs) {
                            stack->emplace_back(std::move(out));
                        }
                    }
                    return;
                }
                auto& p = pending();
                p.active = keyed;
                p.key = std::move(key);
                p.returns = list ? 0 : returns.size();
                p.tensors.clear();
                for (const auto& t : tensors) {
                    p.tensors.push_back(t.unsafeGetTensorImpl());
                }

                struct Done {
                    ~Done() {
                        pending().active = false;
                    }
                } done;

                first_.callBoxed(stack);
                bump();
            }

        private:
            int64_t id_;
            c10::OperatorHandle first_;
        };

        // Makes op `name` ("ns::op") a program op on tt tensors; `first` (same schema)
        // serves its calls without a remembered launch (ktt.run_program_first).
        void program_op(const std::string& name, const std::string& first) {
            static std::atomic<int64_t> next_id{0};
            const auto split = name.find("::");
            TORCH_CHECK(split != std::string::npos, "ktt::program_op: bad name ", name);
            auto handle = c10::Dispatcher::singleton().findSchemaOrThrow(
                first.c_str(), ""
            );
            // Lives as long as the process: the registration is never withdrawn.
            auto* lib = new torch::Library(
                torch::Library::FRAGMENT,
                name.substr(0, split),
                std::nullopt,
                __FILE__,
                __LINE__
            );
            lib->impl(
                name.substr(split + 2).c_str(),
                c10::DispatchKey::PrivateUse1,
                torch::CppFunction::makeFromBoxedFunctor(
                    std::make_unique<ProgramOp>(next_id++, std::move(handle))
                )
            );
        }

        // Output i has the sizes out_sizes[sum(out_ranks[:i]) : +out_ranks[i]] and
        // ScalarType out_dtypes[i].
        std::vector<extension::Output> outputs_of(
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
                outputs.push_back(
                    {{out_sizes.begin() + offset, out_sizes.begin() + offset + rank},
                     static_cast<at::ScalarType>(out_dtypes[i])}
                );
                offset += rank;
            }
            return outputs;
        }

        // Remembers the launch for the pending program-op call when every input is
        // one of its tensors and the outputs are what the op returns.
        void remember(
            at::TensorList inputs,
            const std::string& spec,
            const std::vector<extension::Output>& outputs,
            c10::IntArrayRef values
        ) {
            auto& p = pending();
            if (!p.active) {
                return;
            }
            p.active = false;
            if (p.returns != 0 && p.returns != outputs.size()) {
                return;
            }
            auto launch = std::make_shared<Launch>();
            for (const auto& t : inputs) {
                const auto at = std::find(
                    p.tensors.begin(), p.tensors.end(), t.unsafeGetTensorImpl()
                );
                if (at == p.tensors.end()) {
                    return;
                }
                launch->positions.push_back(at - p.tensors.begin());
            }
            launch->outputs = outputs;
            for (const auto& o : outputs) {
                launch->shapes.push_back(o.sizes);
                launch->dtypes.push_back(to_bridge(o.dtype));
            }
            launch->values = values.vec();
            launch->program = bridge::user_program(spec);
            auto& all = launches();
            const std::lock_guard lock(all.mutex);
            if (all.map.size() >= kMaxLaunches) {
                all.map.clear();
            }
            all.map[p.key] = std::move(launch);
        }

        // Runs `spec` with per-call `values`; outputs as outputs_of().
        std::vector<at::Tensor> run_program(
            at::TensorList inputs,
            const std::string& spec,
            c10::IntArrayRef out_ranks,
            c10::IntArrayRef out_sizes,
            c10::IntArrayRef out_dtypes,
            c10::IntArrayRef values
        ) {
            static const std::string op = "ktt::run_program";
            return run_user(
                op,
                inputs,
                outputs_of(out_ranks, out_sizes, out_dtypes),
                [&](const auto& handles, const auto& shapes, const auto& dtypes) {
                    return bridge::run_program(
                        spec, handles, shapes, dtypes, values.vec()
                    );
                }
            );
        }

        // run_program() as the launch of the program op whose first-call op runs on
        // this thread: remembered for that op.
        std::vector<at::Tensor> run_program_first(
            at::TensorList inputs,
            const std::string& spec,
            c10::IntArrayRef out_ranks,
            c10::IntArrayRef out_sizes,
            c10::IntArrayRef out_dtypes,
            c10::IntArrayRef values
        ) {
            auto result = run_program(
                inputs, spec, out_ranks, out_sizes, out_dtypes, values
            );
            remember(
                inputs, spec, outputs_of(out_ranks, out_sizes, out_dtypes), values
            );
            return result;
        }

    }  // namespace

    namespace extension {

        std::vector<at::Tensor> run(
            const std::string& name,
            at::TensorList inputs,
            const std::vector<Output>& outputs
        ) {
            static const std::string op = "ktt::extension::run";
            return run_user(
                op,
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
            "int[] out_dtypes, int[] values=[]) -> Tensor[]",
            &run_program
        );
        m.def(
            "run_program_first(Tensor[] inputs, str spec, int[] out_ranks, "
            "int[] out_sizes, int[] out_dtypes, int[] values=[]) -> Tensor[]",
            &run_program_first
        );
        m.def("program_op(str name, str first) -> ()", &program_op);
    }

}  // namespace ktt
