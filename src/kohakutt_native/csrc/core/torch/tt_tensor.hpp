// tt tensors on the torch side: storage-backed TensorImpls whose storage context holds
// a base device tensor; views are ordinary PyTorch views of that storage.
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <ATen/ATen.h>

#include "core/device/tt_bridge.hpp"

namespace ktt {

    // The storage context: the base device tensor, its logical sizes and dtype. Bridge
    // handles are immutable; an in-place write replaces `base` for every alias. A
    // storage filled with raw bytes (deserialization) keeps them in `pending` until
    // first typed use.
    struct TtStorage {
        bridge::HandlePtr base;
        std::vector<int64_t> sizes;
        at::ScalarType dtype = at::kByte;
        at::Tensor pending;
        // True from allocation by torch.empty until the first write (commit()).
        bool blank = false;

        // q / k / v heads split from `heads_of` (a fused [B, T, 3 H D] base), reused by
        // the three views of one attention (core/torch/views.cpp).
        bridge::HandlePtr heads_of;
        int64_t heads_count = 0, heads_epoch = -1;
        std::vector<bridge::HandlePtr> heads;

        // [B, C / D, T, D] head-major copy of `major_of` (a [B, T, C] base, head dim
        // `major_d`), reused by its head views (core/torch/views.cpp).
        bridge::HandlePtr major_of, major;
        int64_t major_d = 0, major_epoch = -1;
        // `major` relabeled as the head-major storage of the [B, T, H, D] view.
        bridge::HandlePtr major_view;

        // Placements the last consumers of a persistent mesh-placed base wanted
        // (core/torch/mesh_rules.cpp, migration).
        std::vector<std::string> wanted;

        // The value of a one-element storage when the host knows it
        // (set_known_scalar(); every other write clears it).
        std::optional<double> scalar;

        // A base this storage alone holds (a private copy made for in-place writes);
        // kernels may write it in place while it is still `base`.
        std::weak_ptr<bridge::Handle> owned;
    };

    // The host-known value of one-element tt tensor `t` covering its storage; nullopt
    // when unknown.
    std::optional<double> known_scalar(const at::Tensor& t);

    // Records `value` as the value one-element tt tensor `t` now holds on the device.
    void set_known_scalar(const at::Tensor& t, double value);

    // Allocates the base of `st` for typed use as `dtype`: uploads pending bytes, or an
    // empty 1-D base for a storage that has never been written.
    void ensure_base(TtStorage& st, at::ScalarType dtype, size_t nbytes);

    // The raw bytes [offset, offset + count) of a storage, as a uint8 CPU tensor.
    at::Tensor storage_bytes(TtStorage& st, int64_t offset, int64_t count);

    // Replaces a storage's contents with raw host bytes (whole-storage byte writes).
    void set_storage_bytes(TtStorage& st, const at::Tensor& bytes);

    // The tt allocator: storages created by size alone (UntypedStorage on tt).
    c10::Allocator* tt_allocator();

    // The physical torch dtype a logical dtype is stored as on device: bf16, f32,
    // int32; int64 -> int32 (range-checked on upload), bool -> bf16 (0/1). Others raise
    // unsupported.
    at::ScalarType physical(at::ScalarType dtype);

    // The bridge dtype of a logical torch dtype (via physical()).
    bridge::DType to_bridge(at::ScalarType dtype);

    std::vector<int64_t> to_vector(c10::IntArrayRef sizes);

    // A new contiguous tt tensor (fresh storage) of `sizes` and `dtype` holding
    // `handle`.
    at::Tensor make_tt(
        bridge::HandlePtr handle, c10::IntArrayRef sizes, at::ScalarType dtype
    );

    // The storage context of tt tensor `t`.
    TtStorage& storage_of(const at::Tensor& t);

    // The device tensor holding exactly the elements of view `t` (views.cpp); raises
    // unsupported for stride patterns it cannot express.
    bridge::HandlePtr resolve(const std::string& op, const at::Tensor& t);

    // True when 4-D tt tensor `t` has channels_last strides over its whole storage
    // (offset 0, every storage element): its storage holds the values in NHWC order.
    bool whole_channels_last(const at::Tensor& t);

    // The whole channels_last view `t` as an Nhwc handle of its NCHW sizes without a
    // copy (views.cpp): a bf16 / fp32 [N, H, W, C] tile base with W, C % 32 == 0.
    bridge::HandlePtr channels_last_nhwc(const at::Tensor& t);

    // `t` as a head-major handle without a conversion, else null (views.cpp): a
    // head-major storage or an in-order view of one, or a head view under the rule.
    bridge::HandlePtr head_stored(const at::Tensor& t);

    // `t` as a head view [B, H, S, D] (strides [S C, D, C, 1]) of a whole bf16 Nchw
    // [B, S, C] storage from column t.storage_offset() (views.cpp), else nullopt.
    std::optional<bridge::PitchedView> head_view(const at::Tensor& t);

    // Most bytes of the two circular buffers of one concat-heads program: each holds
    // heads x D / 32 tiles.
    constexpr int64_t kMaxMergedHeadBytes = 1024 * 1024;

    // The block-float base of 2-D `t` when t is its whole storage, or (`transposed`
    // set) the transpose of it; null otherwise (matmul operands, aten/matmul).
    bridge::HandlePtr block_base(const at::Tensor& t, bool& transposed);

    // Records `st`'s current base before its first rebinding in an active capture
    // (capture.cpp); `storage` keeps the context alive.
    void note_rebind(TtStorage& st, const c10::Storage& storage);

    // Empties the head caches filled in the pass that just ended (views.cpp), so they
    // do not keep its buffers alive; called by prepare_end() and capture_end().
    void drop_pass_caches();

    // The base of `t` for a kernel that writes it in place: `t` a whole contiguous
    // bf16 / fp32 / int32 storage (Nchw, replicated), its base made private outside
    // traces, its caches dropped and the write recorded (note_in_place); else null.
    bridge::HandlePtr written_base(const at::Tensor& t);

    // Writes `value` (dense, sizes of `dst`) into view `dst` (views.cpp).
    void commit(const std::string& op, const at::Tensor& dst, bridge::HandlePtr value);

}  // namespace ktt
