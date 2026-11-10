# kohakutt_native

The C++ layer behind the `tt` device, loaded by `kohakutt`: the ATen ops PyTorch
dispatches to `tt`, and the device runtime they share. Custom (non-ATen) kernels are
user ops: Python `kohakutt.kernels` descriptions, or extension libraries built on
`include/ktt/extension/` with `cmake/KohakuTT.cmake`.

```
include/ktt/extension/  public C++ API of extension libraries: device.hpp (program
                        builders), torch.hpp (running them on tt tensors)
cmake/KohakuTT.cmake    TT-NN / torch build flags; kohakutt_add_extension()
csrc/
  aten/<family>/     every ATen op tt implements, one folder per op family:
                     aten_*.cpp (registration + PyTorch semantics), device_*.cpp
                     (implementation), ktt_<name>/ (Kohaku-TT's own Metalium
                     programs: program.cpp + reader / compute / writer.cpp)
  core/torch/        torch side: tt tensor model, views, the kernel contract helpers
  core/device/       device side: device, default stream, handles, trace capture,
                     tt_bridge.hpp (the interface between the two sides)
  runtime/           torch-side runtime: device guard, hooks, capture, ktt:: ops
```

**Where is `aten::X` implemented?** In `csrc/aten/<family>/`: its registration in
`aten_*.cpp`, its implementation in `device_*.cpp` (and its own Metalium program, if
any, in a `ktt_<name>/` folder). `docs/reference/aten_support.md` lists every
overload and its status. Templates: [csrc/aten/README.md](csrc/aten/README.md); a
standalone kernel walkthrough: [examples/kernels/metalium](../../examples/kernels/metalium/).

Torch-side files (`core/torch/`, `runtime/`, `aten_*.cpp`) include torch headers only;
device-side files (`core/device/`, `device_*.cpp`, `ktt_*/program.cpp`) include TT-NN /
Metalium headers only. They meet in `core/device/tt_bridge.hpp`, which carries neither.

Design notes: [kernel contract](../../docs/concepts/kernel-contract.md),
[tensor layout](../../docs/concepts/layout.md), [trace](../../docs/concepts/trace.md).
Format C / C++ with `python -m kohakutt.tools.cxx_format`.
