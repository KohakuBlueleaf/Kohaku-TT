// Native trace capture on the torch side: storages rebound while capturing are written
// back into their original buffers inside the trace, so every replay reads and writes
// the live state.
#include <torch/library.h>

#include "core/device/trace.hpp"
#include "core/torch/contract.hpp"
#include "core/torch/mesh_rules.hpp"

namespace ktt {
    namespace {

        // Storages rebound during the active capture: context, original base,
        // keep-alive.
        struct Rebind {
            TtStorage* storage;
            bridge::HandlePtr original;
            c10::Storage keep;
        };

        std::optional<std::vector<Rebind>> g_rebinds;

        // Prepare pass: log rebinds of one eager call (no trace).
        void prepare_begin() {
            TORCH_CHECK(!g_rebinds, "tt: a capture or prepare pass is already active");
            g_rebinds.emplace();
            bridge::trace_placement(true);
            bridge::set_tracing(true);
        }

        // Copies `src` into `dst` in place, converted to dst's mesh placement first
        // when the two differ.
        void write_back(const bridge::HandlePtr& dst, const bridge::HandlePtr& src) {
            const auto p = bridge::placement(dst);
            bridge::copy_into(
                dst, bridge::placement(src) == p ? src : bridge::distribute(src, p)
            );
        }

        // Runs, in the capture's state, the write-backs a capture of the same call
        // runs: every storage the eager call rebound is written back into a scratch
        // buffer like its original base (an intermediate's original in the capture)
        // and like its new base (a persistent storage's original in the capture). The
        // call's own effects stand; the pass ends also when a copy raises.
        void prepare_end() {
            TORCH_CHECK(g_rebinds, "tt: no prepare pass is active");
            auto rebinds = std::move(*g_rebinds);
            g_rebinds.reset();
            try {
                for (auto& r : rebinds) {
                    const auto& base = r.storage->base;
                    if (!r.original || !base) {
                        continue;
                    }
                    write_back(bridge::empty_like(r.original), base);
                    write_back(bridge::empty_like(base), base);
                }
            } catch (...) {
                bridge::trace_placement(false);
                bridge::set_tracing(false);
                throw;
            }
            bridge::trace_placement(false);
            bridge::set_tracing(false);
        }

        int64_t capture_begin() {
            bridge::trace_placement(true);
            bridge::set_tracing(true);
            int64_t id = 0;
            try {
                id = bridge::trace_begin();
            } catch (...) {
                bridge::set_tracing(false);
                bridge::trace_placement(false);
                throw;
            }
            g_rebinds.emplace();
            return id;
        }

        // Ends the capture: write-backs for rebound storages, then the trace itself.
        // Returns the number of device tensors the trace holds. When an op of the
        // capture failed, the trace is released and its error raised.
        int64_t capture_end() {
            TORCH_CHECK(g_rebinds.has_value(), "tt: no capture is active");
            auto rebinds = std::move(*g_rebinds);
            g_rebinds.reset();
            int64_t held = 0;
            try {
                for (auto& r : rebinds) {
                    if (r.original && r.storage->base != r.original) {
                        write_back(r.original, r.storage->base);
                        r.storage->base = r.original;
                    }
                }
            } catch (...) {
                // Rebound storages keep the original buffers; the trace is dropped.
                for (auto& r : rebinds) {
                    if (r.original) {
                        r.storage->base = r.original;
                    }
                }
                bridge::set_tracing(false);
                try {
                    bridge::trace_abort();
                } catch (...) {
                    // The first error is the one raised.
                }
                bridge::trace_placement(false);
                throw;
            }
            bridge::set_tracing(false);
            try {
                held = bridge::trace_end();
            } catch (...) {
                bridge::trace_placement(false);
                throw;
            }
            bridge::trace_placement(false);
            return held;
        }

        // Copies tt tensor `src` into whole-storage tt tensor `dst` in place (trace
        // inputs); a mesh-placed `dst` takes `src` converted to its placement, per
        // chip.
        void copy_into(const at::Tensor& dst, const at::Tensor& src) {
            const char* op = "ktt::copy_into";
            KTT_REQUIRE(
                dst.is_privateuseone() && src.is_privateuseone(),
                op,
                "needs two tt tensors"
            );
            KTT_REQUIRE(dst.sizes() == src.sizes(), op, "sizes differ");
            KTT_REQUIRE(dst.scalar_type() == src.scalar_type(), op, "dtypes differ");
            auto& st = storage_of(dst);
            KTT_REQUIRE(
                dst.storage_offset() == 0 && dst.is_contiguous() && st.base &&
                    dst.sizes() == c10::IntArrayRef(st.sizes),
                op,
                "destination must be a whole, contiguous tt tensor"
            );
            const auto p = bridge::placement(st.base);
            if (p.is_replicated()) {
                // The source as stored: bridge::copy_into converts only when formats
                // differ.
                bridge::copy_into(st.base, dense_any_layout(op, src));
                return;
            }
            auto from = mesh::placed_base(src);
            if (!from) {
                from = dense(op, src);
            }
            if (!(bridge::placement(from) == p)) {
                from = bridge::distribute(from, p);
            }
            bridge::copy_into(bridge::local_view(st.base), bridge::local_view(from));
        }

        void replay(int64_t id) {
            bridge::trace_execute(id);
        }

        int64_t unsafe_buffers(int64_t id) {
            return bridge::unsafe_buffers(id);
        }

        std::vector<int64_t> trace_footprint(int64_t id) {
            return bridge::trace_footprint(id);
        }

        std::vector<int64_t> debug_buffer_range(const at::Tensor& self) {
            KTT_REQUIRE(
                self.is_privateuseone() && storage_of(self).base,
                "ktt::_debug_buffer_range",
                "needs a tt tensor"
            );
            return bridge::buffer_range(storage_of(self).base);
        }

        int64_t debug_fence(int64_t begin, int64_t end) {
            return bridge::try_fence(begin, end);
        }

        void release(int64_t id) {
            bridge::trace_release(id);
        }

        std::vector<int64_t> debug_trace_l1(int64_t id) {
            return bridge::trace_l1_state(id);
        }

    }  // namespace

    void note_rebind(TtStorage& st, const c10::Storage& storage) {
        if (!g_rebinds) {
            return;
        }
        for (const auto& r : *g_rebinds) {
            if (r.storage == &st) {
                return;
            }
        }
        g_rebinds->push_back({&st, st.base, storage});
    }

    TORCH_LIBRARY_FRAGMENT(ktt, m) {
        m.def("prepare_begin() -> ()", &prepare_begin);
        m.def("prepare_end() -> ()", &prepare_end);
        m.def("capture_begin() -> int", &capture_begin);
        m.def("capture_end() -> int", &capture_end);
        m.def("copy_into(Tensor dst, Tensor src) -> ()", &copy_into);
        m.def("replay(int id) -> ()", &replay);
        m.def("unsafe_buffers(int id) -> int", &unsafe_buffers);
        m.def("trace_footprint(int id) -> int[]", &trace_footprint);
        m.def("release(int id) -> ()", &release);
        m.def("_debug_trace_l1(int id) -> int[]", &debug_trace_l1);
        m.def("_debug_buffer_range(Tensor self) -> int[]", &debug_buffer_range);
        m.def("_debug_fence(int begin, int end) -> int", &debug_fence);
    }

}  // namespace ktt
