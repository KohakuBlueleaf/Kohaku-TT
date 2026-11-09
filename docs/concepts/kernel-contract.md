---
title: Kernel contract (v1)
summary: "How an aten kernel for the tt device is written. Version: ktt::kContractVersion in csrc/core/torch/contract.hpp; a change that breaks kernels bumps it and po..."
tags:
  - concepts
  - kernel-contract
---

# Kernel contract (v1)

How an aten kernel for the `tt` device is written. Version: `ktt::kContractVersion` in
`csrc/core/torch/contract.hpp`; a change that breaks kernels bumps it and ports the
template.

## Where code goes

| Piece | File | Sees |
|---|---|---|
Everything for an ATen op lives in its family folder `csrc/aten/<family>/` (families:
eltwise, matmul, linalg, attention, nn, reduce, index, tensor, spectral; `matmul` holds the
matmul family (mm, bmm, linear, scaled mm and their kernels), `linalg` the `torch.linalg`
decompositions and solves):

| Piece | File | Sees |
|---|---|---|
| registration + validation | `aten_<name>.cpp` | torch only |
| implementation (TT-NN calls, composites) | `device_<name>.cpp` (+ declaration in `core/device/tt_bridge.hpp`) | TT-NN only |
| own Metalium program, host side | `ktt_<name>/program.{hpp,cpp}`, helpers in `core/device/program.hpp` | TT-NN / Metalium |
| own Metalium program, device kernels | `ktt_<name>/{reader,compute,writer}.cpp` | Metalium device APIs |
| OpInfo names covered | `TARGETS` in `src/kohakutt/tools/coverage.py` | Python |

CMake globs the folders by file name; a new file needs no build edit. Templates:
`csrc/aten/README.md`.

## Writing a kernel

1. **Validate** every argument with `KTT_REQUIRE(cond, "<overload>", msg...)`. A variant the
   kernel cannot do (dtype, rank, broadcasting, alpha, layout) must raise here as
   `KohakuTTUnsupportedOperator`, never reach TT-NN and fail there.
2. **Read inputs** with `dense("<overload>", t)`: a device tensor holding exactly the
   elements of `t`. Views (any PyTorch sizes/strides/offset over a tt storage) are
   resolved to reshape / slice / permute / repeat device ops; patterns that cannot be
   expressed raise unsupported.
3. **Compute** through a bridge function (one TT-NN call or a Metalium `generic_op`).
4. **Return** `wrap_output(handle, sizes, dtype)` with the logical (unpadded) sizes.
5. **In-place and out=** variants compute the functional result and call
   `write_into(op, dst, result)`; it writes through `commit()`, which replaces the
   storage's base tensor (all aliases see the write). Writes into a partial view raise
   unsupported for now.

Table families use rows instead of hand-written kernels; each row generates the
functional, in-place and out= kernels (`tt_<op>`, `tt_<op>_`, `tt_<op>_out`):

```cpp
KTT_UNARY(silu, "silu", bridge::UnaryOp::Silu)               // aten/elementwise/unary.cpp
KTT_BINARY_ALPHA(add, "add", bridge::BinaryOp::Add)          // aten/elementwise/binary.cpp
```

plus the enum value and its `switch` case in `aten/eltwise/device_eltwise.cpp`, and the three
`m.impl(...)` lines.

## Tensor model

- A tt tensor is a regular `TensorImpl` over a `StorageImpl` whose context
  (`TtStorage`) holds the base device tensor and its logical sizes. The data pointer is
  never dereferenced as host memory.
- View ops are PyTorch's own (`view`, `as_strided`, `_reshape_alias` registered to
  `at::native`), so aliasing, version counters and autograd view tracking are standard.
- Ops never mutate their inputs: every op returns a new handle, and in-place writes swap
  the storage's base. Buffers held by a trace (and trace input buffers) are the
  exception: replays overwrite them in place. Therefore copies (`copy_`, `clone`,
  `_to_copy`) always copy data (`private_copy`) and never share a handle between two
  storages.
- Logical rank 0 and 1 tensors are stored as device rank 2 ((1, 1) and (1, n)).

## Rules

- Inputs and outputs are logical PyTorch tensors: sizes as PyTorch sees them, padding and
  tiling are the bridge's business.
- Bridge calls are asynchronous: they predict the output's device shape, enqueue the
  TT-NN call on the default stream and return. Kernels never read device data on the
  host; only ops whose output depends on data (e.g. `nonzero`) synchronize.
- No host round-trips inside a kernel (bridge counters must show 0 uploads/downloads).
- No silent dtype casts; mixed dtypes either follow PyTorch promotion explicitly
  (`result_type`) or raise.
- Matmul-class ops use the bridge numerics (fp32 accumulation; HiFi3 for bf16 /
  block-float operands, HiFi4 with an fp32 operand; `KTT_MATMUL_FIDELITY` overrides).
- An own Metalium program replaces a TT-NN op only where it is measured faster with
  equal error statistics; the bridge op keeps the TT-NN path (`KTT_CUSTOM_KERNELS=0`)
  and chooses per call (e.g. by size).
- An overload without a kernel raises from the boxed fallback, naming its tensor args.

## Checking a kernel

1. `python -m kohakutt.tools.coverage --ops <opinfo names>`: per-sample outcome
   `ok` / `unsupported` / `error`. `error` is a bug (the kernel let a variant through).
2. Numerics are judged as statistics, never thresholds: the tt result and a CPU baseline
   at the same precision (same rounded inputs) are both measured against a CPU fp64
   reference, and their error statistics (max / mean / median of absolute and relative
   error) are compared. Matching statistics mean a correct implementation; a gap is a
   bug to examine. Records go to `docs/reference/coverage_stats.json`.
3. Data-movement ops are checked for bit-exactness against CPU.
4. In a real model, `kohakutt.tools.kernel_audit.KernelAudit` checks every aten call on
   its actual inputs the same way (fp64 ruler, same-precision CPU simulation), or times
   each call device-synchronized.
5. `python -m kohakutt.tools.support` regenerates `docs/reference/aten_support.md`.
