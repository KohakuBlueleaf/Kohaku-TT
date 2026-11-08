// torch.ops.ktt.*: mesh open/close, synchronize, counters and memory statistics.
#include <mutex>

#include <ATen/ATen.h>
#include <torch/library.h>

#include "core/torch/constants.hpp"
#include "core/torch/contract.hpp"
#include "core/device/tt_bridge.hpp"

namespace ktt {
    namespace {

        void close_mesh() {
            clear_device_constants();
            bridge::close();
        }

        void open_mesh(std::vector<int64_t> device_ids, int64_t rows, int64_t cols) {
            std::vector<int> ids(device_ids.begin(), device_ids.end());
            bridge::open(ids, static_cast<int>(rows), static_cast<int>(cols));
        }

        void mark_exiting() {
            bridge::mark_exiting();
        }

        int64_t mesh_size() {
            return bridge::mesh_size();
        }

        std::vector<int64_t> compute_grid() {
            return bridge::compute_grid();
        }

        void synchronize() {
            bridge::synchronize();
        }

        std::vector<int64_t> counters() {
            auto c = bridge::counters();
            return {
                c.uploads,
                c.downloads,
                c.ops,
                c.op_ns,
                c.programs,
                c.desc_hits,
                c.alloc_ns,
                c.args_ns,
                c.desc_ns,
                c.enqueue_ns
            };
        }

        std::vector<int64_t> memory_stats() {
            auto m = bridge::memory_stats();
            return {m.allocated, m.peak};
        }

        void reset_peak_memory() {
            bridge::reset_peak_memory();
        }

        // Numerics scope of the calling thread (fidelity rank 0..3, approx, strict,
        // attention mode 0..2).
        void numerics_push(
            int64_t fidelity, bool approx, bool strict, int64_t attention
        ) {
            TORCH_CHECK(
                fidelity >= 0 && fidelity <= 3, "ktt::numerics_push: fidelity rank"
            );
            TORCH_CHECK(
                attention >= 0 && attention <= 2, "ktt::numerics_push: attention mode"
            );
            bridge::numerics_push(
                static_cast<int>(fidelity), approx, strict, static_cast<int>(attention)
            );
        }

        void numerics_pop() {
            bridge::numerics_pop();
        }

        std::vector<int64_t> numerics_current() {
            const auto n = bridge::numerics_current();
            return {n.fidelity, n.approx, n.strict, n.attention};
        }

        // The default tt generator (hooks.cpp), locked for the caller's scope.
        at::Generator& default_generator() {
            static auto generator = (
                at::globalContext().defaultGenerator(
                    c10::Device(c10::DeviceType::PrivateUse1, 0)
                )
            );
            return generator;
        }

        void manual_seed(int64_t seed) {
            auto& gen = default_generator();
            std::lock_guard<std::mutex> lock(gen.mutex());
            gen.set_current_seed(static_cast<uint64_t>(seed));
        }

        int64_t initial_seed() {
            return static_cast<int64_t>(default_generator().current_seed());
        }

        at::Tensor get_rng_state() {
            auto& gen = default_generator();
            std::lock_guard<std::mutex> lock(gen.mutex());
            return gen.get_state();
        }

        void set_rng_state(const at::Tensor& state) {
            auto& gen = default_generator();
            std::lock_guard<std::mutex> lock(gen.mutex());
            gen.set_state(state);
        }

    }  // namespace

    TORCH_LIBRARY(ktt, m) {
        m.def("open(int[] device_ids, int rows, int cols) -> ()", &open_mesh);
        m.def("close() -> ()", &close_mesh);
        m.def("mark_exiting() -> ()", &mark_exiting);
        m.def("mesh_size() -> int", &mesh_size);
        m.def("compute_grid() -> int[]", &compute_grid);
        m.def("synchronize() -> ()", &synchronize);
        m.def("read_device_profiler() -> ()", &bridge::read_device_profiler);
        m.def("clear_program_cache() -> ()", &bridge::clear_program_cache);
        m.def("counters() -> int[]", &counters);
        m.def("memory_stats() -> int[]", &memory_stats);
        m.def("reset_peak_memory() -> ()", &reset_peak_memory);
        m.def(
            "numerics_push(int fidelity, bool approx, bool strict, int attention) -> "
            "()",
            &numerics_push
        );
        m.def("numerics_pop() -> ()", &numerics_pop);
        m.def("numerics_current() -> int[]", &numerics_current);
        m.def("manual_seed(int seed) -> ()", &manual_seed);
        m.def("initial_seed() -> int", &initial_seed);
        m.def("get_rng_state() -> Tensor", &get_rng_state);
        m.def("set_rng_state(Tensor state) -> ()", &set_rng_state);
        m.def("fallback_counts() -> str[]", &fallback_counts);
        m.def("reset_fallback_counts() -> ()", &reset_fallback_counts);
    }

}  // namespace ktt
