---
title: Kernel DSL
summary: "Kernel programs written in Python: what the metal level describes, what never recompiles, what the verifier checks."
tags:
  - concepts
  - dsl
---

# Kernel DSL

`kohakutt.dsl` describes a kernel program in Python and turns it into TT-Metalium
kernel sources. A program written this way is a torch op on `tt` tensors like any
hand-written one: it runs on the default stream, in eager mode and inside traces.

The package has one level today, the **metal level** (`kohakutt.dsl.metal`, imported as
`km`). It names exactly what a Metalium program has: circular buffers, the two
data-movement kernels, the compute kernel, NoC reads and writes, register slots. It adds
no abstraction over the hardware; it adds binding times and checks.

Working examples: [`examples/kernels/dsl_metal/`](../../examples/kernels/dsl_metal/)
(SwiGLU and RoPE, each beside its hand-written twin in `examples/kernels/metalium/`).

## Why a language

Hand-written kernels share one structure, and the effort is not in the math:

| Part | Content | Where the effort goes |
|---|---|---|
| reader | tile ranges per core; DRAM -> L1 circular buffers | data movement |
| compute | per-tile unpack, math, pack over circular buffers | numerics choices |
| writer | circular buffer -> DRAM by tile range | little |
| host | program description: buffers, kernel sources, arguments, core split | boilerplate, error-prone |

The rules that make a kernel correct on the hardware (a push size divides its buffer, a
push follows its read barrier, register slots fit the mode) are not checked by any
compiler: breaking one gives a hang or wrong values. The DSL holds the program as a
model so those rules are checked before a kernel is built.

## A program

```python
import kohakutt.dsl.metal as km


@km.program("swiglu")
def swiglu_program(p, a: km.TensorRef, b: km.TensorRef, out: km.Out):
    start, count = p.split(p.runtime("tiles"))
    cb_a, cb_b, cb_out = p.cb(0, like=a), p.cb(1, like=b), p.cb(16, like=out)

    @p.dm0
    def reader(k):
        for i in k.range(count):
            a_slot, b_slot = cb_a.reserve(1), cb_b.reserve(1)
            t = k.let("t", start + i)
            k.read(a.page(t), a_slot)
            k.read(b.page(t), b_slot)
            k.reads_done()
            cb_a.push(1)
            cb_b.push(1)

    ...  # @p.compute and @p.dm1
```

The function body runs once, in Python, and records the program; the loops and
arithmetic on values become C++ in the generated kernels. `km.op(schema, program)`
registers a torch op whose builder returns a `km.Call` (the tensors and the runtime
values of this call). The builder runs once per argument sizes, dtypes, non-tensor
values and `kohakutt.numerics` scope; the backend remembers the launch and serves
later such calls without running Python.

## Binding times

Every value in a program has one of these binding times, and the binding time decides
what a change of that value costs:

| Binding time | Examples | A new value costs |
|---|---|---|
| static | dtype, the op body, buffer page counts, declared `km.Static` parameters | a kernel compile |
| size class | the number of cores: the smallest of 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, the whole grid that covers the split's total (cores past it take no work) | one TT-Metal program build, no kernel compile |
| runtime | tile counts, tensor addresses, `p.runtime(name)` values | nothing |
| runtime, from a fixed list | `p.runtime("block", choices=(1, 2, 3))`: a value that may also be a push / pop count, a loop step or a register slot; the verifier checks every choice | nothing |
| runtime, a declared multiple | `p.runtime("nr", multiple_of="block")`: checked at every launch, so a loop over it in steps of `block` is whole | nothing |
| per core | the core's index, from which it derives its own range (`p.split`) or its items in turn (`p.stride`) | nothing |
| data | a word loaded from L1 on the device (`slot.load32(i)`), such as a row index; it is never a count or a loop bound, and it selects what a NoC read fetches only with a declared range (`.bounded(low, high)`) | nothing |

The range of a data value is the launch's contract: nothing checks it on the device, so
an index outside its table reads other memory.

So a new batch size, sequence length or image size compiles nothing: sizes are runtime
arguments and each core derives its share of the work on the device. Only what the
author declares static (a block size, a dtype) can cause a compile.

Measured on a Blackhole p150 with the two example programs, over 14 RoPE shapes and 10
SwiGLU shapes: no kernel compiled after the first call, 9 and 7 program builds (one per
core count used), output equal bit for bit to the hand-written kernels, warm time
within noise of them.

## The verifier

`km.verify(program)` runs before anything is emitted; `km.VerifyError` names the rule.
`km.RULES` lists them. Each rule comes from a failure seen on the hardware:

| Group | Checks |
|---|---|
| binding | a value is used only where its binding time allows (a runtime value is never a buffer size, a loaded value never a count); a stepped loop runs over a multiple of its step |
| circular buffers | push and pop sizes divide the buffer; pushes equal pops across the kernels (none inside a condition); a kernel writes and loads no further than it reserved; the buffers fit L1 |
| NoC | reader and writer use different NoCs; a sub-page read keeps source and destination aligned alike, proved from what the offsets are multiples of; a push, a load or a write follows its read barrier; a page is flushed before it is popped or read into again |
| compute | register slots fit the mode (8 bf16 or 4 fp32 tiles per half); math sits between acquire and commit, packs between commit and release; a changed vector-unit constant is restored at exit |
| inline C++ | the buffers, semaphores and registers it touches are declared |

Inline C++ (`k.inline`, `km.SfpuFn`) is the way to write anything the typed operations
do not cover; the verifier checks its declared effects, not its text.

## Generated sources

Sources are written under `KOHAKUTT_CACHE/km` (default:
`~/.cache/kohakutt/km`, or under `XDG_CACHE_HOME`), named by content. `km.source(program, role)` returns the text of one kernel,
which reads like a hand-written Metalium kernel.

## Relation to `kohakutt.kernels` and tt-lang

`kohakutt.kernels` runs kernel files you wrote; the DSL writes them. Both produce the
same program description and share the launch path, so they can be mixed and compared.

[tt-lang](https://github.com/tenstorrent/tt-lang) has a similar shape (compute and
data-movement functions over dataflow buffers). It brings its own compiler and runtime,
which cannot share a process with TT-NN's; this DSL lowers in process to the program
descriptions every other Kohaku-TT op uses.
