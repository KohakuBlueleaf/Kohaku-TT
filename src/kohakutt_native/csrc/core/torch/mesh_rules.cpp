// Mesh placement rules on tt tensors: placed views and persistent-tensor migration.
#include "core/torch/mesh_rules.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <map>

#include "core/torch/tt_tensor.hpp"

namespace ktt::mesh {
    namespace {

        using bridge::Placement;

        // Whole-storage persistent tensors whose placed bases ops read recently: a
        // parameter (requires_grad) or a tensor written in place.
        struct Seen {
            std::weak_ptr<bridge::Handle> base;
            c10::weak_intrusive_ptr<c10::TensorImpl> owner{
                c10::intrusive_ptr<c10::TensorImpl>()
            };
        };

        constexpr size_t kSeen = 64;
        thread_local std::array<Seen, kSeen> g_seen;
        thread_local size_t g_seen_next = 0;
        std::atomic<int64_t> g_migrations{0};

        bool persistent(const at::Tensor& t) {
            return t.requires_grad() || (!t.is_inference() && t._version() > 0);
        }

        void note_base(const at::Tensor& t, const bridge::HandlePtr& base) {
            if (!persistent(t)) {
                return;
            }
            for (auto& s : g_seen) {
                if (s.base.lock() == base) {
                    return;
                }
            }
            g_seen[g_seen_next++ % kSeen] = Seen{
                base, c10::weak_intrusive_ptr<c10::TensorImpl>(t.getIntrusivePtr())
            };
        }

        // The live persistent tensor whose storage base is `h`, if one was read
        // recently; undefined otherwise.
        at::Tensor persistent_owner(const bridge::HandlePtr& h) {
            for (auto& s : g_seen) {
                if (s.base.lock() != h) {
                    continue;
                }
                auto impl = s.owner.lock();
                if (!impl) {
                    continue;
                }
                at::Tensor t(std::move(impl));
                if (storage_of(t).base == h) {
                    return t;
                }
            }
            return at::Tensor();
        }

        // True under KTT_MESH_MIGRATE=1 (not faster in every measured case yet).
        bool migration_enabled() {
            static const bool on = [] {
                const char* v = std::getenv("KTT_MESH_MIGRATE");
                return v && std::string(v) == "1";
            }();
            return on;
        }

        // Records that a consumer of `st`'s base wanted `want` (the base is at `now`);
        // true when one placement other than `now` holds a majority of the last 8
        // wants, at least 2 of them (hysteresis).
        bool want_moves(TtStorage& st, const Placement& want, const Placement& now) {
            constexpr size_t kWindow = 8;
            st.wanted.push_back(want.str());
            if (st.wanted.size() > kWindow) {
                st.wanted.erase(st.wanted.begin());
            }
            std::map<std::string, size_t> counts;
            for (const auto& w : st.wanted) {
                ++counts[w];
            }
            const auto top = std::max_element(
                counts.begin(), counts.end(), [](const auto& x, const auto& y) {
                    return x.second < y.second;
                }
            );
            return (
                top->first != now.str() && top->second >= 2 &&
                top->second * 2 > st.wanted.size()
            );
        }

        // Migration: a persistent input's conversion priced as amortized over the
        // window; the placement its consumers keep wanting becomes permanent.
        bool migrate(
            const Factors& f,
            std::vector<Operand>& ops,
            std::vector<bridge::HandlePtr>& in,
            const Hints& hints
        ) {
            if (!migration_enabled() || bridge::traces_alive()) {
                return false;
            }
            std::vector<at::Tensor> owners(in.size());
            auto amortized_ops = ops;
            bool any = false;
            for (size_t i = 0; i < in.size(); ++i) {
                owners[i] = persistent_owner(in[i]);
                if (owners[i].defined()) {
                    amortized_ops[i].weight = 1.0 / 8;
                    any = true;
                }
            }
            auto amortized = any ? decide(f, amortized_ops, hints) : std::nullopt;
            bool moved = false;
            for (size_t i = 0; amortized && i < in.size(); ++i) {
                if (!owners[i].defined()) {
                    continue;
                }
                const auto& want = amortized->inputs[i];
                auto& st = storage_of(owners[i]);
                if (want.is_replicated() || !want_moves(st, want, ops[i].at)) {
                    continue;
                }
                auto moved_to = bridge::distribute(in[i], want);
                st.base = moved_to;
                st.heads_of.reset();
                st.heads.clear();
                st.wanted.clear();
                in[i] = moved_to;
                ops[i].at = want;
                ops[i].ready.clear();
                moved = true;
                g_migrations++;
            }
            return moved;
        }

        void warn(const std::string& message) {
            TORCH_WARN(message);
        }

        // Fallback warnings as torch warnings; migration from the tensors' storages.
        [[maybe_unused]] const bool g_installed = [] {
            set_warning_sink(&warn);
            set_migration_hook(&migrate);
            return true;
        }();

    }  // namespace

    int64_t migrations() {
        return g_migrations.load();
    }

    bridge::HandlePtr placed_base(const at::Tensor& t) {
        if (!t.is_privateuseone() || t.numel() == 0) {
            return nullptr;
        }
        auto& st = storage_of(t);
        if (!st.base || st.dtype != t.scalar_type() ||
            bridge::is_block(bridge::dtype(st.base)) ||
            bridge::placement(st.base).is_replicated()) {
            return nullptr;
        }
        const auto& b = st.sizes;
        const auto v = to_vector(t.sizes());
        int64_t numel = 1;
        for (const int64_t d : b) {
            numel *= d;
        }
        const bool whole =
            (t.is_contiguous() && t.storage_offset() == 0 && numel == t.numel());
        if (whole && v == b) {
            note_base(t, st.base);
            return st.base;
        }
        return view_of(
            st.base, b, v, to_vector(t.strides()), t.storage_offset(), whole
        );
    }

}  // namespace ktt::mesh
