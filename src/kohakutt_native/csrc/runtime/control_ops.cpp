// torch.ops.ktt.*: mesh open/close, synchronize, counters and memory statistics.
#include <algorithm>
#include <mutex>
#include <sstream>
#include <string>

#include <ATen/ATen.h>
#include <torch/library.h>

#include "core/torch/constants.hpp"
#include "core/torch/contract.hpp"
#include "core/torch/mesh_rules.hpp"
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

        // Parses "R", "S(d)", "P(sum|max|min)" per mesh axis, comma separated; S dims
        // are torch dims of a rank-`rank` tensor.
        bridge::Placement parse_placement(const std::string& spec, int64_t rank) {
            bridge::Placement p;
            std::string item;
            std::stringstream in(spec);
            const int64_t lift = rank < 2 ? 2 - rank : 0;
            while (std::getline(in, item, ',')) {
                item.erase(std::remove(item.begin(), item.end(), ' '), item.end());
                if (item == "R") {
                    p.axes.push_back(bridge::AxisPlacement::replicate());
                } else if (item.rfind("S(", 0) == 0 && item.back() == ')') {
                    int64_t d = std::stoll(item.substr(2, item.size() - 3));
                    d = d < 0 ? d + rank : d;
                    TORCH_CHECK(
                        d >= 0 && d < std::max<int64_t>(rank, 1),
                        "ktt: shard dim of ",
                        spec
                    );
                    p.axes.push_back(
                        bridge::AxisPlacement::shard(static_cast<int>(d + lift))
                    );
                } else if (item == "P(sum)" || item == "P") {
                    p.axes.push_back(
                        bridge::AxisPlacement::partial(bridge::ReduceKind::Sum)
                    );
                } else if (item == "P(max)") {
                    p.axes.push_back(
                        bridge::AxisPlacement::partial(bridge::ReduceKind::Max)
                    );
                } else if (item == "P(min)") {
                    p.axes.push_back(
                        bridge::AxisPlacement::partial(bridge::ReduceKind::Min)
                    );
                } else {
                    TORCH_CHECK(
                        false, "ktt: bad placement item '", item, "' in '", spec, "'"
                    );
                }
            }
            return p;
        }

        // torch.ops.ktt.distribute: a copy of tt tensor `self` placed by `spec`.
        at::Tensor distribute(const at::Tensor& self, const std::string& spec) {
            // A whole, contiguous storage converts from its base as placed.
            auto& st = storage_of(self);
            const bool whole =
                (st.base && self.is_contiguous() && self.storage_offset() == 0 &&
                 self.numel() * self.element_size() ==
                     static_cast<int64_t>(self.storage().nbytes()) &&
                 !bridge::is_block(bridge::dtype(st.base)));
            auto h = whole ? st.base : dense("ktt::distribute", self);
            auto placed = bridge::distribute(h, parse_placement(spec, self.dim()));
            return wrap_output(placed, self.sizes(), self.scalar_type());
        }

        // torch.ops.ktt.placement: the placement `self` is read at, e.g. "[R, S(1)]"
        // (device dims); "[]" for a view of placed storage no placement maps to.
        std::string placement_of(const at::Tensor& self) {
            if (auto h = mesh::placed_base(self)) {
                return bridge::placement(h).str();
            }
            auto& st = storage_of(self);
            if (!st.base) {
                return "[]";
            }
            const bool block = bridge::is_block(bridge::dtype(st.base));
            return block ? bridge::placement(st.base).str() : "[]";
        }

        // torch.ops.ktt.local_shard: chip `chip`'s local values of `self`'s storage
        // base as a CPU tensor of the local device shape.
        at::Tensor local_shard(const at::Tensor& self, int64_t chip) {
            auto& st = storage_of(self);
            TORCH_CHECK(st.base, "ktt::local_shard: tensor has no device base");
            const auto dt = physical(st.dtype);
            auto host = (
                at::empty(bridge::local_sizes(st.base), at::TensorOptions().dtype(dt))
            );
            bridge::download_local(st.base, host.data_ptr(), static_cast<int>(chip));
            return host;
        }

        std::vector<int64_t> mesh_shape() {
            auto s = bridge::mesh_shape();
            return std::vector<int64_t>(s.begin(), s.end());
        }

        std::vector<int64_t> mem_get_info() {
            auto m = bridge::mem_info();
            return {m.free, m.total};
        }

        std::vector<int64_t> mesh_counters() {
            auto c = bridge::mesh_counters();
            return {
                c.gathers,
                c.gather_bytes,
                c.all_gathers,
                c.all_reduces,
                c.reduce_scatters,
                c.partitions,
                c.host_conversions,
                c.ccl_failures
            };
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
        m.def("distribute(Tensor self, str placement) -> Tensor", &distribute);
        m.def("placement(Tensor self) -> str", &placement_of);
        m.def("local_shard(Tensor self, int chip) -> Tensor", &local_shard);
        m.def("mesh_counters() -> int[]", &mesh_counters);
        m.def("mem_get_info() -> int[]", &mem_get_info);
        m.def("mesh_fallback_ops() -> str[]", &mesh::fallback_ops);
        m.def("reset_mesh_fallback_ops() -> ()", &mesh::reset_fallback_ops);
        m.def("mesh_shape() -> int[]", &mesh_shape);
        m.def("reset_mesh_counters() -> ()", &bridge::reset_mesh_counters);
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
