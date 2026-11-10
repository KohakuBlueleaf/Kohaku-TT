// View resolution and in-place commit: maps (sizes, strides, offset) over a storage's
// base tensor onto reshape / slice / permute / repeat device ops.
#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>

#include "core/torch/constants.hpp"
#include "core/torch/contract.hpp"
#include "core/torch/mesh_rules.hpp"

namespace ktt {
    namespace {

        using Dims = std::vector<int64_t>;

        Dims contiguous_strides(const Dims& sizes) {
            Dims strides(sizes.size(), 1);
            for (int64_t i = static_cast<int64_t>(sizes.size()) - 2; i >= 0; --i) {
                strides[i] = strides[i + 1] * std::max<int64_t>(sizes[i + 1], 1);
            }
            return strides;
        }

        int64_t product(const Dims& sizes) {
            return std::accumulate(
                sizes.begin(), sizes.end(), int64_t{1}, std::multiplies<>()
            );
        }

        // Device-dim indices for per-logical-dim values of a rank-r tensor (rank < 2
        // pads).
        Dims to_device(const Dims& values, int64_t pad_value) {
            Dims out;
            for (size_t i = values.size(); i < 2; ++i) {
                out.push_back(pad_value);
            }
            out.insert(out.end(), values.begin(), values.end());
            return out;
        }

        bool is_contiguous(const Dims& sizes, const Dims& strides) {
            auto expected = contiguous_strides(sizes);
            for (size_t i = 0; i < sizes.size(); ++i) {
                if (sizes[i] != 1 && strides[i] != expected[i]) {
                    return false;
                }
            }
            return true;
        }

        // The view without expanded (stride 0) dims, or false when nothing is expanded.
        bool unexpand(Dims& sizes, Dims& strides, Dims& repeats) {
            bool expanded = false;
            repeats.assign(sizes.size(), 1);
            for (size_t i = 0; i < sizes.size(); ++i) {
                if (strides[i] == 0 && sizes[i] > 1) {
                    repeats[i] = sizes[i];
                    sizes[i] = 1;
                    expanded = true;
                }
            }
            return expanded;
        }

        // Where a view reads its base: per base dim begin, extent, step; view-dim
        // order.
        struct Plan {
            Dims begins, ext, steps, order;
            std::vector<bool> used;
        };

        bridge::HandlePtr map_dims(
            const std::function<bridge::HandlePtr()>& base,
            const Dims& bs,
            const Dims& sizes,
            const Dims& strides,
            int64_t off
        );

        // Contiguous sizes for a base of base_n elements whose strides are the view's
        // distinct non-unit strides (outermost first); nullopt when they do not nest.
        std::optional<Dims> rebase_sizes(
            int64_t base_n, const Dims& sizes, const Dims& strides
        ) {
            Dims nz;
            for (size_t i = 0; i < sizes.size(); ++i) {
                if (sizes[i] > 1) {
                    nz.push_back(strides[i]);
                }
            }
            std::sort(nz.rbegin(), nz.rend());
            nz.erase(std::unique(nz.begin(), nz.end()), nz.end());
            if (nz.empty() || nz[0] <= 0 || base_n % nz[0] != 0) {
                return std::nullopt;
            }
            Dims rebased{base_n / nz[0]};
            for (size_t k = 0; k + 1 < nz.size(); ++k) {
                if (nz[k + 1] <= 0 || nz[k] % nz[k + 1] != 0) {
                    return std::nullopt;
                }
                rebased.push_back(nz[k] / nz[k + 1]);
            }
            if (nz.back() > 1) {
                rebased.push_back(nz.back());
            }
            return rebased;
        }

        // Dense handle of view (sizes, strides, offset) over `st`, or nullptr if
        // unsupported.
        bridge::HandlePtr resolve_view(
            const TtStorage& st, Dims sizes, Dims strides, int64_t off
        ) {
            const Dims& bs = st.sizes;
            const int64_t n = product(sizes);
            const int64_t base_n = product(bs);

            // Expanded dims: resolve the unexpanded view, then repeat.
            Dims repeats;
            if (unexpand(sizes, strides, repeats)) {
                auto inner = resolve_view(st, sizes, strides, off);
                return inner ? bridge::repeat(inner, to_device(repeats, 1)) : nullptr;
            }

            // Identity and contiguous reshape of the whole base.
            bool contiguous = is_contiguous(sizes, strides);
            if (off == 0 && contiguous && n == base_n) {
                return sizes == bs ? st.base : bridge::reshape(st.base, sizes);
            }

            // Permuted contiguous view: resolve it in stride order, then permute back.
            if (sizes.size() >= 2) {
                Dims order(sizes.size());
                std::iota(order.begin(), order.end(), 0);
                std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
                    return strides[a] > strides[b];
                });
                Dims ps, pst;
                for (int64_t d : order) {
                    ps.push_back(sizes[d]);
                    pst.push_back(strides[d]);
                }
                if (!std::is_sorted(order.begin(), order.end()) &&
                    is_contiguous(ps, pst)) {
                    auto inner = resolve_view(st, ps, pst, off);
                    if (inner) {
                        Dims back(order.size());
                        for (size_t k = 0; k < order.size(); ++k) {
                            back[order[k]] = static_cast<int64_t>(k);
                        }
                        return bridge::permute(inner, back);
                    }
                }
            }

            // Map view dims onto the base dims, or onto a reshape of the base whose
            // contiguous strides contain the view's strides.
            if (auto h = map_dims(
                    [&] {
                        return st.base;
                    },
                    bs,
                    sizes,
                    strides,
                    off
                )) {
                return h;
            }

            // Adjacent view dims with nesting strides merged: resolve the coarser view,
            // then reshape back (e.g. a sliced [B, H*W, C] base read as [B, H, W, C]).
            if (sizes.size() >= 2) {
                Dims cs{sizes[0]}, cst{strides[0]};
                for (size_t i = 1; i < sizes.size(); ++i) {
                    if (cst.back() == sizes[i] * strides[i]) {
                        cs.back() *= sizes[i];
                        cst.back() = strides[i];
                    } else {
                        cs.push_back(sizes[i]);
                        cst.push_back(strides[i]);
                    }
                }
                if (cs.size() < sizes.size()) {
                    if (auto h = resolve_view(st, cs, cst, off)) {
                        return bridge::reshape(h, sizes);
                    }
                }
            }

            // A contiguous range no base dim maps (e.g. crossing rows): flatten, slice,
            // reshape.
            if (contiguous && off + n <= base_n) {
                auto flat = bridge::reshape(st.base, {base_n});
                auto part = bridge::slice(flat, {0, off}, {1, off + n}, {1, 1});
                return bridge::reshape(part, sizes);
            }
            auto rebased = rebase_sizes(base_n, sizes, strides);
            if (!rebased || *rebased == bs) {
                return nullptr;
            }
            return map_dims(
                [&] {
                    return bridge::reshape(st.base, *rebased);
                },
                *rebased,
                sizes,
                strides,
                off
            );
        }

        // Each non-unit view dim mapped onto a dim of a contiguous base of sizes `bs`;
        // nullopt when no mapping exists.
        std::optional<Plan> plan_dims(
            const Dims& bs, const Dims& sizes, const Dims& strides, int64_t off
        ) {
            const auto bst = contiguous_strides(bs);
            const size_t r = bs.size();
            Plan p{Dims(r, 0), Dims(r, 1), Dims(r, 1), {}, std::vector<bool>(r, false)};
            int64_t rem = off;
            for (size_t j = 0; j < r; ++j) {
                p.begins[j] = bst[j] ? rem / bst[j] : 0;
                rem -= p.begins[j] * bst[j];
                if (p.begins[j] >= std::max<int64_t>(bs[j], 1)) {
                    return std::nullopt;
                }
            }
            if (rem != 0) {
                return std::nullopt;
            }
            for (size_t i = 0; i < sizes.size(); ++i) {
                if (sizes[i] == 1) {
                    continue;
                }
                int64_t best = -1;
                for (size_t j = 0; j < r; ++j) {
                    if (p.used[j] || strides[i] % bst[j] != 0) {
                        continue;
                    }
                    int64_t step = strides[i] / bst[j];
                    if (step > 0 && p.begins[j] + (sizes[i] - 1) * step < bs[j] &&
                        (best < 0 || bst[j] > bst[best])) {
                        best = static_cast<int64_t>(j);
                    }
                }
                if (best < 0) {
                    return std::nullopt;
                }
                p.used[best] = true;
                p.ext[best] = sizes[i];
                p.steps[best] = strides[i] / bst[best];
                p.order.push_back(best);
            }
            return p;
        }

        // Ends (exclusive) of the base region a plan reads.
        Dims plan_ends(const Plan& p) {
            Dims ends(p.begins.size());
            for (size_t j = 0; j < ends.size(); ++j) {
                ends[j] = p.begins[j] + (p.ext[j] - 1) * p.steps[j] + 1;
            }
            return ends;
        }

        // The view read through plan `p` from a base of sizes `bs`: slice, permute,
        // reshape. `base` is called only when the mapping exists.
        bridge::HandlePtr map_dims(
            const std::function<bridge::HandlePtr()>& base,
            const Dims& bs,
            const Dims& sizes,
            const Dims& strides,
            int64_t off
        ) {
            auto plan = plan_dims(bs, sizes, strides, off);
            if (!plan) {
                return nullptr;
            }
            const auto& p = *plan;
            const size_t r = bs.size();
            auto h = base();
            if (p.begins != Dims(r, 0) || p.ext != bs || p.steps != Dims(r, 1)) {
                h = (bridge::slice(
                    h,
                    to_device(p.begins, 0),
                    to_device(plan_ends(p), 1),
                    to_device(p.steps, 1)
                ));
            }
            Dims shape = p.ext;
            if (!std::is_sorted(p.order.begin(), p.order.end()) && r >= 2) {
                // Base dims the view does not use (extent 1) lead, in base order.
                Dims perm;
                for (size_t j = 0; j < r; ++j) {
                    if (!p.used[j]) {
                        perm.push_back(static_cast<int64_t>(j));
                    }
                }
                perm.insert(perm.end(), p.order.begin(), p.order.end());
                h = bridge::permute(h, perm);
                for (size_t k = 0; k < r; ++k) {
                    shape[k] = p.ext[perm[k]];
                }
            }
            return shape == sizes ? h : bridge::reshape(h, sizes);
        }

        // `base` (sizes bs) with the box [begins, ends) replaced by `value` (box
        // sizes), built by slicing and concatenating along each partially covered dim.
        bridge::HandlePtr splice(
            const bridge::HandlePtr& base,
            Dims bs,
            const bridge::HandlePtr& value,
            const Dims& begins,
            const Dims& ends,
            size_t from = 0
        ) {
            const size_t r = bs.size();
            const int64_t pad = std::max<int64_t>(0, 2 - static_cast<int64_t>(r));
            for (size_t d = from; d < r; ++d) {
                if (begins[d] == 0 && ends[d] == bs[d]) {
                    continue;
                }

                // before | middle (recursive on later dims) | after, along dim d.
                auto range = [&](int64_t lo, int64_t hi) {
                    Dims b(r, 0), e = bs;
                    b[d] = lo;
                    e[d] = hi;
                    return (bridge::slice(
                        base, to_device(b, 0), to_device(e, 1), Dims(r + pad, 1)
                    ));
                };
                Dims mid_sizes = bs;
                mid_sizes[d] = ends[d] - begins[d];
                Dims mid_begins = begins, mid_ends = ends;
                mid_begins[d] = 0;
                mid_ends[d] = mid_sizes[d];
                std::vector<bridge::HandlePtr> parts;
                if (begins[d] > 0) {
                    parts.push_back(range(0, begins[d]));
                }
                parts.push_back(splice(
                    range(begins[d], ends[d]),
                    mid_sizes,
                    value,
                    mid_begins,
                    mid_ends,
                    d + 1
                ));
                if (ends[d] < bs[d]) {
                    parts.push_back(range(ends[d], bs[d]));
                }
                return parts.size() == 1 ? parts[0] : bridge::concat(parts, d + pad);
            }
            return value;
        }

        // Any view as a gather from the flattened base at flat element indices computed
        // on host (uploads the index, so not used while capturing a trace).
        bridge::HandlePtr gather_view(const TtStorage& st, const at::Tensor& t) {
            const int64_t base_n = product(st.sizes);
            const int64_t n = t.numel();
            if (base_n > std::numeric_limits<int32_t>::max()) {
                return nullptr;
            }
            std::vector<int32_t> index(n);
            const auto sizes = to_vector(t.sizes());
            const auto strides = to_vector(t.strides());
            Dims pos(sizes.size(), 0);
            for (int64_t e = 0; e < n; ++e) {
                int64_t flat = t.storage_offset();
                for (size_t k = 0; k < sizes.size(); ++k) {
                    flat += pos[k] * strides[k];
                }
                if (flat < 0 || flat >= base_n) {
                    return nullptr;
                }
                index[e] = static_cast<int32_t>(flat);
                for (int64_t k = static_cast<int64_t>(sizes.size()) - 1; k >= 0; --k) {
                    if (++pos[k] < sizes[k]) {
                        break;
                    }
                    pos[k] = 0;
                }
            }
            // The index is a device constant per (base, view) geometry: built on the
            // first (eager) read, reused without upload inside trace captures.
            auto key = (c10::str(
                "view|",
                c10::IntArrayRef(st.sizes),
                "|",
                t.sizes(),
                "|",
                t.strides(),
                "|",
                t.storage_offset()
            ));
            auto idx_tensor = device_constant(key, [&] {
                return at::tensor(index, at::kInt).reshape({1, n});
            });
            auto flat_base = bridge::reshape(st.base, {1, base_n});
            auto picked = (bridge::gather(
                flat_base, 1, resolve("gather_view", idx_tensor)
            ));
            return bridge::reshape(picked, sizes);
        }

        // The [B, C / d, T, d] head-major copy of st's [B, T, C] base, cached per write
        // epoch (not while traces are alive).
        bridge::HandlePtr major_copy(TtStorage& st, int64_t d) {
            const int64_t b = st.sizes[0], seq = st.sizes[1], heads = st.sizes[2] / d;
            const int64_t epoch = bridge::write_epoch();
            const bool caching = !bridge::traces_alive();
            if (caching && st.major_of == st.base && st.major_d == d &&
                st.major_epoch == epoch) {
                return st.major;
            }
            bridge::HandlePtr major;
            if (heads == 1) {
                major = bridge::reshape(st.base, {b, 1, seq, d});
            } else if (seq % 32 == 0 && d % 32 == 0) {
                major = bridge::split_heads(st.base, heads);
            } else {
                major = bridge::permute(
                    bridge::reshape(st.base, {b, seq, heads, d}), {0, 2, 1, 3}
                );
            }
            if (caching) {
                st.major = major;
                st.major_of = st.base;
                st.major_d = d;
                st.major_epoch = epoch;
                st.major_view.reset();
            }
            return major;
        }

        // View (sizes, strides, off) of a contiguous base of sizes `bs` as strides and
        // offset over that base with dims -3 and -2 swapped; nullopt when unmapped.
        std::optional<std::pair<Dims, int64_t>> swapped_view(
            const Dims& bs, const Dims& sizes, const Dims& strides, int64_t off
        ) {
            const size_t r = bs.size(), n = sizes.size();
            Dims plain = sizes;
            for (size_t i = 0; i < n; ++i) {
                if (strides[i] == 0) {
                    plain[i] = 1;
                }
            }
            const auto plan = plan_dims(bs, plain, strides, off);
            if (!plan) {
                return std::nullopt;
            }
            Dims ss = bs;
            std::swap(ss[r - 3], ss[r - 2]);
            const auto sst = contiguous_strides(ss);
            auto stored_stride = [&](int64_t j) {
                const size_t d = static_cast<size_t>(j);
                return sst[d == r - 3 ? r - 2 : d == r - 2 ? r - 3 : d];
            };
            int64_t moved = 0;
            for (size_t j = 0; j < r; ++j) {
                moved += plan->begins[j] * stored_stride(j);
            }
            Dims out(n, 0);
            size_t k = 0;
            for (size_t i = 0; i < n; ++i) {
                if (plain[i] != 1) {
                    const int64_t j = plan->order[k++];
                    out[i] = plan->steps[j] * stored_stride(j);
                }
            }
            // Size-1 dims take the stride a contiguous dim would have there.
            int64_t next = 1;
            for (size_t i = n; i-- > 0;) {
                if (sizes[i] == 1) {
                    out[i] = next;
                } else if (out[i] != 0) {
                    next = out[i] * sizes[i];
                }
            }
            return std::make_pair(out, moved);
        }

        // True when the dims of size > 1 have strides in descending order.
        bool in_order(const Dims& sizes, const Dims& strides) {
            int64_t last = std::numeric_limits<int64_t>::max();
            for (size_t i = 0; i < sizes.size(); ++i) {
                if (sizes[i] > 1 && strides[i] != 0) {
                    if (strides[i] > last) {
                        return false;
                    }
                    last = strides[i];
                }
            }
            return true;
        }

        // A storage whose base is head-major `st`'s stored tensor [.., H, T, D].
        TtStorage stored_storage(const TtStorage& st) {
            TtStorage plain;
            plain.base = bridge::as_stored(st.base);
            plain.sizes = st.sizes;
            const size_t r = plain.sizes.size();
            std::swap(plain.sizes[r - 3], plain.sizes[r - 2]);
            plain.dtype = st.dtype;
            return plain;
        }

        // Head-major `h` [B, T, H, D] as [B, T, H D] from one concat-heads program,
        // else null.
        bridge::HandlePtr merge_major(const bridge::HandlePtr& h) {
            const auto s = bridge::stored_sizes_of(
                bridge::device_sizes(h), bridge::Format::HeadMajor
            );
            const int64_t tile_bytes = bridge::dtype(h) == bridge::DType::BFloat16
                ? 2048
                : 4096;
            if (s.size() != 4 || s[3] % 32 != 0 || bridge::traces_alive() ||
                2 * s[1] * (s[3] / 32) * tile_bytes > kMaxMergedHeadBytes) {
                return nullptr;
            }
            return bridge::concat_heads(bridge::as_stored(h));
        }

        // View `t` of head-major storage `st` as an Nchw handle read from the stored
        // tensor; null when the view needs the storage's Nchw copy.
        bridge::HandlePtr resolve_major(TtStorage& st, const at::Tensor& t) {
            const auto sizes = to_vector(t.sizes());
            const auto strides = to_vector(t.strides());
            const int64_t off = t.storage_offset();
            const size_t r = st.sizes.size();
            const bool whole = t.is_contiguous() && off == 0 &&
                t.numel() == product(st.sizes);
            if (whole && sizes == st.sizes) {
                return bridge::to_nchw(st.base);
            }
            // The heads merged back: [.., T, H D].
            if (whole && sizes.size() + 1 == r &&
                sizes.back() == st.sizes[r - 2] * st.sizes[r - 1] &&
                std::equal(sizes.begin(), sizes.end() - 1, st.sizes.begin())) {
                if (auto merged = merge_major(st.base)) {
                    return merged;
                }
            }
            if (auto moved = swapped_view(st.sizes, sizes, strides, off)) {
                const auto plain = stored_storage(st);
                if (auto h = resolve_view(plain, sizes, moved->first, moved->second)) {
                    if (!in_order(sizes, moved->first)) {
                        bridge::count_head_major_read();
                    }
                    return h;
                }
            }
            return nullptr;
        }

        // Head view [B, H, T, d], strides [T C, D, C, 1], of a [B, T, C] base: a slice
        // of the base's cached [B, C / D, T, D] head-major copy, else nullptr.
        bridge::HandlePtr head_major(TtStorage& st, const at::Tensor& t) {
            if (st.sizes.size() != 3 || t.dim() != 4 || t.scalar_type() == at::kInt ||
                bridge::traces_alive()) {
                return nullptr;
            }
            const int64_t b = st.sizes[0], seq = st.sizes[1], c = st.sizes[2];
            const int64_t d_full = t.stride(1), off = t.storage_offset();
            // Strides of size-1 dims are not checked.
            if (d_full <= 0 || c % d_full != 0 || d_full % 32 != 0 || t.size(0) != b ||
                t.size(2) != seq || (b > 1 && t.stride(0) != seq * c) ||
                (seq > 1 && t.stride(2) != c) || t.stride(3) != 1 || t.size(1) < 2 ||
                off >= c) {
                return nullptr;
            }
            const int64_t h0 = off / d_full, inner = off % d_full;
            const int64_t heads = t.size(1), d = t.size(3);
            if (inner % 32 != 0 || d % 32 != 0 || inner + d > d_full ||
                h0 + heads > c / d_full) {
                return nullptr;
            }
            const auto major = major_copy(st, d_full);
            if (heads == c / d_full && d == d_full) {
                return major;
            }
            return bridge::slice(
                major,
                to_device({0, h0, 0, inner}, 0),
                to_device({b, h0 + heads, seq, inner + d}, 1),
                to_device({1, 1, 1, 1}, 1)
            );
        }

        // q / k / v of a fused projection: view [B, H, T, D] with strides [T C, D, C,
        // 1] at column offset i H D of a [B, T, C = 3 H D] base, served from one cached
        // head split; other head views from head_major().
        bridge::HandlePtr head_split(TtStorage& st, const at::Tensor& t) {
            if (st.sizes.size() != 3 || t.dim() != 4) {
                return nullptr;
            }
            const int64_t b = st.sizes[0], seq = st.sizes[1], c = st.sizes[2];
            const int64_t heads = t.size(1), d = t.size(3);
            const int64_t off = t.storage_offset();
            const bool shaped =
                (t.size(0) == b && t.size(2) == seq && c == 3 * heads * d &&
                 t.stride(0) == seq * c && t.stride(1) == d && t.stride(2) == c &&
                 t.stride(3) == 1 && off % (heads * d) == 0 && off < c);
            if (!shaped || d % 32 != 0 || t.scalar_type() == at::kInt) {
                return head_major(st, t);
            }
            // While traces are alive a base is written in place, so its split is not
            // cached.
            if (bridge::traces_alive()) {
                return bridge::split_qkv_heads(st.base, heads)[off / (heads * d)];
            }
            const int64_t epoch = bridge::write_epoch();
            if (st.heads_of != st.base || st.heads_count != heads ||
                st.heads_epoch != epoch) {
                st.heads = bridge::split_qkv_heads(st.base, heads);
                st.heads_of = st.base;
                st.heads_count = heads;
                st.heads_epoch = epoch;
            }
            return st.heads[off / (heads * d)];
        }

        // resolve() of a non-empty view `t` over storage `st` (bf16 / fp32 / int32
        // base).
        bridge::HandlePtr resolve_in(
            const std::string& op, TtStorage& st, const at::Tensor& t
        ) {
            auto h = head_split(st, t);
            if (!h) {
                h = (resolve_view(
                    st, to_vector(t.sizes()), to_vector(t.strides()), t.storage_offset()
                ));
            }
            if (!h) {
                h = gather_view(st, t);
            }
            KTT_REQUIRE(
                h != nullptr,
                op,
                "unsupported view: sizes ",
                t.sizes(),
                " strides ",
                t.strides(),
                " offset ",
                t.storage_offset(),
                " over base ",
                c10::IntArrayRef(st.sizes)
            );
            return h;
        }

    }  // namespace

    bridge::HandlePtr head_stored(const at::Tensor& t) {
        if (!t.is_privateuseone() || t.dim() < 4 || t.numel() == 0) {
            return nullptr;
        }
        auto& st = storage_of(t);
        if (!st.base || st.dtype != t.scalar_type()) {
            return nullptr;
        }
        const auto format = bridge::format(st.base);
        const bool view = format == bridge::Format::Nchw && t.dim() == 4 &&
            st.sizes.size() == 3;
        if ((format != bridge::Format::HeadMajor && !view) ||
            !bridge::placement(st.base).is_replicated()) {
            return nullptr;
        }
        const bool whole = t.is_contiguous() && t.storage_offset() == 0 &&
            t.numel() == product(st.sizes);
        if (!view && whole && t.sizes() == c10::IntArrayRef(st.sizes)) {
            return st.base;
        }
        const auto sizes = to_vector(t.sizes());
        const auto dims = bridge::device_sizes_of(sizes);
        if (!view) {
            // A view that keeps the order of the stored dims: read as stored.
            if (sizes.size() != st.sizes.size()) {
                return nullptr;
            }
            auto moved = (swapped_view(
                st.sizes, sizes, to_vector(t.strides()), t.storage_offset()
            ));
            if (!moved) {
                return nullptr;
            }
            const size_t r = sizes.size();
            Dims vs = sizes, vst = moved->first;
            std::swap(vs[r - 3], vs[r - 2]);
            std::swap(vst[r - 3], vst[r - 2]);
            if (!in_order(vs, vst)) {
                return nullptr;
            }
            const auto plain = stored_storage(st);
            auto h = resolve_view(plain, vs, vst, moved->second);
            return h ? bridge::view_as(h, dims, bridge::Format::HeadMajor) : nullptr;
        }
        // A head view [B, T, H, D] of a whole [B, T, H D] storage.
        if (!whole || st.sizes[0] != sizes[0] || st.sizes[1] != sizes[1] ||
            !bridge::head_major_rule(dims, bridge::dtype(st.base))) {
            return nullptr;
        }
        const auto major = major_copy(st, sizes[3]);
        if (major != st.major || bridge::traces_alive()) {
            return bridge::view_as(major, dims, bridge::Format::HeadMajor);
        }
        if (!st.major_view) {
            st.major_view = bridge::view_as(major, dims, bridge::Format::HeadMajor);
        }
        return st.major_view;
    }

    bridge::HandlePtr resolve(const std::string& op, const at::Tensor& t) {
        KTT_REQUIRE(t.is_privateuseone(), op, "expected a tt tensor, got ", t.device());
        bridge::layout_reader(op);
        auto& st = storage_of(t);
        ensure_base(st, t.scalar_type(), t.storage().nbytes());
        KTT_REQUIRE(
            st.dtype == t.scalar_type(),
            op,
            "a tt storage of ",
            st.dtype,
            " viewed as ",
            t.scalar_type(),
            " is not supported"
        );
        if (t.numel() == 0) {
            return bridge::empty(to_vector(t.sizes()), to_bridge(t.scalar_type()));
        }

        // The whole of a Flat base read at its stored rows [d0, d1 ... dk] (e.g.
        // weight.reshape(O, -1)): the stored tensor itself.
        auto stored_rows = [&](const bridge::HandlePtr& base) -> bridge::HandlePtr {
            if (bridge::format(base) != bridge::Format::Flat || !t.is_contiguous() ||
                t.storage_offset() != 0 || t.numel() != product(st.sizes)) {
                return nullptr;
            }
            const auto rows = bridge::stored_sizes_of(
                bridge::device_sizes(base), bridge::Format::Flat
            );
            const auto dims = bridge::device_sizes_of(to_vector(t.sizes()));
            return dims == rows ? bridge::as_stored(base) : nullptr;
        };

        // Block-float storage: views resolve against its bf16 decode (as Nchw).
        if (bridge::is_block(bridge::dtype(st.base))) {
            TtStorage plain;
            plain.base = bridge::typecast(st.base, bridge::DType::BFloat16);
            if (auto h = stored_rows(plain.base)) {
                return h;
            }
            plain.base = bridge::to_nchw(plain.base);
            plain.sizes = st.sizes;
            plain.dtype = st.dtype;
            return resolve_in(op, plain, t);
        }

        // Mesh-placed storage: views resolve against its replicated copy (the gather
        // fallback of ops without placement rules).
        if (!bridge::placement(st.base).is_replicated()) {
            mesh::fallback(
                op,
                st.base,
                "the op has no placement rule for it, or reads a view its shards do "
                "not follow"
            );
            TtStorage plain;
            plain.base = bridge::replicated(st.base);
            plain.sizes = st.sizes;
            plain.dtype = st.dtype;
            return resolve_in(op, plain, t);
        }

        // Nhwc or Flat storage: views resolve against its Nchw copy.
        if (bridge::format(st.base) != bridge::Format::Nchw) {
            if (auto h = stored_rows(st.base)) {
                return h;
            }
            if (bridge::format(st.base) == bridge::Format::HeadMajor) {
                if (auto h = resolve_major(st, t)) {
                    return h;
                }
            }
            TtStorage plain;
            plain.base = bridge::to_nchw(st.base);
            plain.sizes = st.sizes;
            plain.dtype = st.dtype;
            return resolve_in(op, plain, t);
        }
        return resolve_in(op, st, t);
    }

    namespace {

        // Binds `new_base` as the storage's base (no copy). While traces are alive it
        // is copied into the existing buffer instead: a buffer allocated now can sit in
        // a trace's temporaries (docs/concepts/trace.md), so persistent storages keep
        // their buffers then.
        void store(TtStorage& st, bridge::HandlePtr new_base) {
            st.heads_of.reset();
            st.heads.clear();
            if (bridge::traces_alive() && st.base && st.base != new_base &&
                bridge::dtype(st.base) == bridge::dtype(new_base) &&
                bridge::placement(st.base) == bridge::placement(new_base)) {
                bridge::copy_into(st.base, new_base);
            } else {
                // A value written over an existing base is persistent.
                if (st.base && st.base != new_base) {
                    bridge::keep_in_dram(new_base);
                }
                st.base = std::move(new_base);
            }
        }

    }  // namespace

    void commit(const std::string& op, const at::Tensor& dst, bridge::HandlePtr value) {
        auto& st = storage_of(dst);
        if (dst.numel() == 0) {
            return;
        }
        st.blank = false;
        ensure_base(st, dst.scalar_type(), dst.storage().nbytes());
        note_rebind(st, dst.storage());
        st.scalar.reset();
        bridge::layout_reader(op);

        // A write into block-float storage stores bf16 from then on.
        if (bridge::is_block(bridge::dtype(st.base))) {
            st.base = bridge::typecast(st.base, bridge::DType::BFloat16);
        }
        const auto sizes = to_vector(dst.sizes());
        const auto strides = to_vector(dst.strides());
        const int64_t off = dst.storage_offset();
        const int64_t base_n = product(st.sizes);

        // The whole storage (an Nhwc value of the base's sizes is stored as it is).
        if (off == 0 && dst.is_contiguous() && dst.numel() == base_n) {
            // A head-major value written through a head view: its heads merged.
            if (bridge::format(value) == bridge::Format::HeadMajor &&
                sizes.size() == 4 &&
                st.sizes == Dims{sizes[0], sizes[1], sizes[2] * sizes[3]} &&
                bridge::placement(st.base).is_replicated()) {
                if (auto merged = merge_major(value)) {
                    store(st, merged);
                    st.dtype = dst.scalar_type();
                    return;
                }
            }
            if (sizes != st.sizes) {
                value = bridge::to_nchw(value);
            }
            store(st, mesh::reshape_placed(op, value, sizes, st.sizes));
            st.dtype = dst.scalar_type();
            return;
        }

        // A partial write works on replicated handles: a mesh-placed base is rebound to
        // its replicated copy, a placed value is converted.
        const char* part = "a write into part of a tensor works on replicated tensors";
        if (!bridge::placement(st.base).is_replicated()) {
            mesh::fallback(op, st.base, part);
            st.base = bridge::replicated(st.base);
        }
        if (!bridge::placement(value).is_replicated()) {
            mesh::fallback(op, value, part);
            value = bridge::replicated(value);
        }
        // ... and Nchw: an Nhwc base is read through its Nchw copy, the result stored
        // as Nchw (copied back into the Nhwc buffer while traces are alive).
        value = bridge::to_nchw(value);
        const auto nchw = bridge::to_nchw(st.base);

        // A box (steps 1, any dim order) of the base, or of a reshape of it (rebase):
        // bring the value into base dim order, splice it in, reshape back.
        auto box_write = [&](const Dims& bs) {
            const size_t r = bs.size();
            auto plan = plan_dims(bs, sizes, strides, off);
            if (!plan || plan->steps != Dims(r, 1)) {
                return false;
            }
            Dims perm = plan->order;
            for (size_t j = 0; j < r; ++j) {
                if (!plan->used[j]) {
                    perm.push_back(static_cast<int64_t>(j));
                }
            }
            Dims view_order_shape(r);
            for (size_t k = 0; k < r; ++k) {
                view_order_shape[k] = plan->ext[perm[k]];
            }
            auto box =
                (view_order_shape == sizes ? value
                                           : bridge::reshape(value, view_order_shape));
            if (!std::is_sorted(perm.begin(), perm.end())) {
                Dims inverse(r);
                for (size_t k = 0; k < r; ++k) {
                    inverse[perm[k]] = static_cast<int64_t>(k);
                }
                box = bridge::permute(box, inverse);
            }
            auto base = bs == st.sizes ? nchw : bridge::reshape(nchw, bs);
            auto spliced = splice(base, bs, box, plan->begins, plan_ends(*plan));
            store(st, bs == st.sizes ? spliced : bridge::reshape(spliced, st.sizes));
            return true;
        };
        if (box_write(st.sizes)) {
            return;
        }
        auto rebased = rebase_sizes(base_n, sizes, strides);
        if (rebased && *rebased != st.sizes && box_write(*rebased)) {
            return;
        }

        // A contiguous range: splice into the flattened base.
        if (dst.is_contiguous() && off + dst.numel() <= base_n) {
            auto flat = bridge::reshape(nchw, {base_n});
            auto spliced = splice(
                flat,
                {base_n},
                bridge::reshape(value, {dst.numel()}),
                {off},
                {off + dst.numel()}
            );
            store(st, bridge::reshape(spliced, st.sizes));
            return;
        }
        unsupported(
            op,
            c10::str(
                "in-place write into view sizes ",
                dst.sizes(),
                " strides ",
                dst.strides(),
                " offset ",
                off,
                " is not supported yet"
            )
        );
    }

}  // namespace ktt
