# Writing your own kernels

Three ways to run your own program on `tt` tensors as a torch op. All three build the
same thing (a reader, a compute kernel and a writer over circular buffers) and run on
the default stream, in order with every other tt op, in eager mode and inside traces.

| folder | you write | pick it when |
|---|---|---|
| [`metalium/`](metalium/) | the three kernels as TT-Metalium C++ files, the program in Python with `kohakutt.kernels` | you want full control of the kernel source, or are porting an existing Metalium kernel |
| [`dsl_metal/`](dsl_metal/) | the whole program in Python with `kohakutt.dsl.metal`; kernel sources are generated and checked | you want a kernel that compiles once for every shape and is verified against the hardware rules before it runs |
| [`cpp_extension/`](cpp_extension/) | the program and the op in C++, built as an extension library | the host side must be C++ (no Python on the call path) |

The same two programs appear in each Python folder, so the files can be read side by
side:

| program | math | `metalium/` | `dsl_metal/` |
|---|---|---|---|
| SwiGLU | `silu(a) * b` | `swiglu.py` + `kernels/{reader,compute,writer}.cpp` | `swiglu_km.py` |
| RoPE | rotate column pairs by a `(cos, sin)` table, in SFPU instructions | `rope.py` + `kernels/rope_{reader,compute}.cpp` | `rope_km.py` |

```bash
python examples/kernels/metalium/swiglu.py --rows 1024 --cols 4096
python examples/kernels/dsl_metal/swiglu_km.py --rows 1024 --cols 4096
python examples/kernels/dsl_metal/rope_km.py --batch 9 --heads 16 --tokens 576 --dim 64
```

The `dsl_metal` examples print whether their output equals the hand-written program's
(it does, bit for bit) and both times.

Start with [`metalium/README.md`](metalium/README.md): it explains the anatomy of a
program and the conventions every kernel follows. The design of the DSL is in
[docs/concepts/dsl.md](../../docs/concepts/dsl.md).
