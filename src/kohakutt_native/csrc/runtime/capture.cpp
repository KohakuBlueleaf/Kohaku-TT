// Native trace capture on the torch side: storages rebound while capturing are written
// back into their original buffers inside the trace, so every replay reads and writes
// the live state.
#include <torch/library.h>

#include "core/torch/contract.hpp"

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
        }

        // Compiles the write-back copy of every storage the eager call rebound (one
        // copy into a scratch buffer each); the call's own effects stand.
        void prepare_end() {
            TORCH_CHECK(g_rebinds, "tt: no prepare pass is active");
            for (auto& r : *g_rebinds) {
                if (r.original && r.storage->base) {
                    auto scratch = (
                        bridge::empty(r.storage->sizes, bridge::dtype(r.storage->base))
                    );
                    bridge::copy_into(scratch, r.storage->base);
                }
            }
            g_rebinds.reset();
        }

        int64_t capture_begin() {
            auto id = bridge::trace_begin();
            g_rebinds.emplace();
            return id;
        }

        // Ends the capture: write-backs for rebound storages, then the trace itself.
        // Returns the number of device tensors the trace holds.
        int64_t capture_end() {
            TORCH_CHECK(g_rebinds.has_value(), "tt: no capture is active");
            for (auto& r : *g_rebinds) {
                if (r.original && r.storage->base != r.original) {
                    bridge::copy_into(r.original, r.storage->base);
                    r.storage->base = r.original;
                }
            }
            g_rebinds.reset();
            return bridge::trace_end();
        }

        // Copies tt tensor `src` into whole-storage tt tensor `dst` in place (trace
        // inputs).
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
            bridge::copy_into(st.base, dense(op, src));
        }

        void replay(int64_t id) {
            bridge::trace_execute(id);
        }

        int64_t unsafe_buffers(int64_t id) {
            return bridge::unsafe_buffers(id);
        }

        int64_t misplaced_fences(int64_t id) {
            return bridge::trace_misplaced_fences(id);
        }

        void release(int64_t id) {
            bridge::trace_release(id);
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
        m.def("misplaced_fences(int id) -> int", &misplaced_fences);
        m.def("release(int id) -> ()", &release);
    }

}  // namespace ktt
