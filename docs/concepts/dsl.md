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

## Building a program by code

The decorated form above is written by a person. A compiler builds the same model
through the builder underneath it, with no traced function and no torch op:

```python
p = km.Program("scale", km.Machine(cols, rows))
x, out = p.input("x", "bfloat16"), p.output("out", "bfloat16")
start, count = p.split(p.runtime("tiles"))
cb_x = p.cb(0, like=x)

k = p.kernel("dm0")  # a builder; several may be open at once
with k.loop(count) as i:
    slot = k.reserve(cb_x, 1)
    k.read(x.page(start + i), slot)
    k.reads_done()
    k.push(cb_x, 1)
...  # p.kernel("compute"), p.kernel("dm1")

run = km.Launcher(p)  # verifies, emits; a description per core count and sizes
(y,) = run({"x": t}, outputs={"out": (list(t.shape), t.dtype)}, runtime={"tiles": n})
```

| Decorated form | By code |
|---|---|
| tensors from the function's annotations | `p.input(name, dtype)`, `p.output(name, dtype)` |
| `@p.dm0` / `@p.compute` / `@p.dm1` over a function | `p.kernel(role, ...)` returns the builder |
| `cb.reserve(n)`, `cb.push(n)`, `cb.wait(n)`, `cb.pop(n)` | `k.reserve(cb, n)`, `k.push(cb, n)`, `k.wait(cb, n)`, `k.pop(cb, n)` |
| `for i in k.range(n)`, `for done, n in k.blocks(count, b)` | `with k.loop(n) as i`, `with k.block_loop(count, b) as (done, n)` |
| `with c.registers():` | the same, or `c.acquire()` ... `c.commit()` ... `c.release()` |
| `km.op(schema, template)` | `km.Launcher(program)` |

Both forms record the same model: `km.dump(program)` prints it (declarations, then
each kernel's statements as a tree), and `km.fingerprint(program)` hashes that text
and the generated sources: equal fingerprints emit the same kernels (`km.op` still
keys its kernel classes on dtypes and statics). A value has a host
side too: `value.eval({"tiles": 96, "core": 3, "cores": 16})` computes what the device
would, which is how per-core arguments written as values of `p.core` are filled.

## The compute kernel

A compute body names what each unit of the core does:

| Unit | Statements |
|---|---|
| unpacker | `c.copy(cb, i, slot)`, `c.copy_block(cb, i, slot, n)`, `c.transpose(cb, i, slot)`, `c.bcast(cb, i, slot, "row" \| "col")`; `c.format(srca=, srcb=)` switches its formats |
| matrix unit | `m = c.matmul_init(a, b, transpose, ct, rt, kt)`, then `c.matmul(m, ia, ib, slot)`: the registers sum over calls; `t = c.tiles_init(op, a, b, bcast=, acc=)`, then `c.tiles(t, ia, ib, slot)`: whole-tile `"add"`, `"sub"`, `"mul"` of a tile of `a` and one of `b` (b's first row, first column or first element repeated with `bcast="row" \| "col" \| "scalar"`; `acc=True` adds the result onto the slot); `r = c.reduce_init(kind, dim, cb, scaler, out)`, then `c.reduce(r, i, si, slot)`: `"sum"` or `"max"` of tile `i` of `cb` along `"row"` (into column 0), `"col"` (into row 0) or `"scalar"` (into [0, 0]), times tile `si` of `scaler` (ones for a plain sum), summed (or maxed) over calls; `c.reduce_done()` clears the packer masks a reduction sets |
| vector unit | `c.op(name, slots..., value=)` for the ops of the op table (`km.ops.OPS`: unary functions, scalar arithmetic, binary arithmetic, comparisons, min / max, fill; on int32 registers `add_int32`, `sub_int32`, `mul_int32` (wrapping), `eq_int32` .. `ge_int32` (1 or 0), `and_int32`, `or_int32`, `xor_int32`, `shl_int32`, `shr_int32` (arithmetic), and `int32_to_fp32` / `fp32_to_int32` (toward zero) in place; int32 buffers are unpacked into fp32 registers, `fp32_dest=True` with the buffers in `unpack_fp32`); `c.sfpu(fn, a, b, out, mode=, args=)` for an own function; `c.run(fn, slots..., out=)` for a function generated from an expression |
| packer | `c.pack(slot, cb)`, `c.pack(slot, cb, at=i)` (page `i` of the reserve), `c.pack_format(cb)`, `c.pack_accumulate(1, cb)` ... `c.pack_accumulate(0)` (a pack adds to its page in L1); `c.pack_accumulate("after_first", cb, after=j)` (off on loop `j`'s first iteration, on after it: one page reserved before the loop sums what each iteration packs) |

A way of unpacking and a table op each take their setup first (`c.init("copy", cb)`,
`c.init("rsqrt")`); the matmul's setup is `c.matmul_init`. The op table is data: one
row (call, setup call, header, operand counts) adds an op.

### A vector-unit function from an expression

`km.sfpu_fn(name, expr)` writes one vector-unit function for an expression over up to
four register operands, so a chain of elementwise steps is one pass over the registers
instead of one pass per op:

```python
f = km.sfpu
a, b, c, d = f.operands(4)
gate = km.sfpu_fn("gate", f.sigmoid(a * b + c) * d)
...
c.init(gate.init)  # the setup the function runs under
c.run(gate, 0, 1, 2, 3, out=0)  # operand slots, result slot
```

| Expression | Meaning |
|---|---|
| `a + b`, `a - b`, `a * b`, `-a`, `abs(a)`, numbers | each rounds to fp32 on its own, as the tile op of that name does |
| `f.mad(a, b, c)` | `a * b + c` in one instruction with one rounding (the product keeps more than fp32 precision) |
| `f.minimum`, `f.maximum`, `f.relu`, `f.square` | lanewise |
| `f.select(a < b, x, y)` | `x` where the comparison (`<`, `<=`, `>`, `>=`, `==`, `!=`) holds, `y` elsewhere |
| `f.sigmoid`, `f.silu`, `f.recip`, `f.exp`, `f.tanh`, `f.sqrt`, `f.rsqrt` | TT-Metal's own code for that function (`km.sfpu.FUNCTIONS`); functions that need different setups do not mix in one function |
| `f.flash_exp(x, a, b)`, `f.flash_exp2(u)` | the backend flash attention's softmax exp, bit for bit: 2^(x a + b - 639) (`u` = x a + b, one multiply-add), the fraction of u through a degree-2 polynomial (relative error up to 1.75e-3), 0 for u below 512; for exp(s - m) take a = scale log2 e and b = 639 - a m. As a tile op: `c.run(km.sfpu_fn("p", f.flash_exp(s, a, b)), ...)` |
| `a.even`, `a.odd`, `{"even": x, "odd": y}` | a function of column pairs: operands and results by even and odd columns (a complex multiply, a rotation) |
| `f.scalar(i)` | fp32 scalar argument `i` (0 .. 3), the same in every lane: `c.run(fn, slots..., out=, values=(v, ...))` passes it as its bits (a float, or a kernel value holding the bits such as a runtime value), so a runtime scalar takes no register slot and no fill |
| `f.bf16(x)` | `x` rounded to bf16 (to nearest, ties to even) inside the chain, as an eager model's bf16 intermediate is |

`c.run(fn, *slots, out=, thread="pack")` runs a generated function on the pack thread,
the backend flash attention's placement: after the commit and before the section's
first pack, it waits for the math thread's section, holds the vector unit's mutex, and
the packs wait for it, so the vector work overlaps the math thread's next section. In
such a kernel every math-thread vector-unit call holds the same mutex. SFPU-3 refuses
a pack-thread run before the commit or after a pack of its section.

An operand is whatever its slot holds, so a row or column vector is an operand once
`c.bcast` has filled the slot. On bf16 registers the result is rounded to bf16 once, at
the store. The function is C++ for the kernel compiler (`fn.code`), which places the
no-op a multiply-add needs before an instruction that the hardware does not hold back.

Measured on a Blackhole p150: `silu`, `silu(a) * b` and a complex multiply written this
way equal the tile ops and the hand-written instruction code bit for bit, and so does a
chain of six ops over four operands (one of them a broadcast row).

A branch on a value that is fixed for the launch may hold buffer calls; both kernels of
a buffer then branch on the same value:

```python
with c.when(flag.ne(0)):
    c.init("mul")
    c.op("mul", 0, 2, 0)
with c.otherwise():
    c.init("sub_scalar")
    c.op("sub_scalar", 0, value=1.0)
```

The row softmax example (`examples/kernels/dsl_metal/softmax_km.py`) is written with
the matrix unit's statements: a row maximum and a row sum by `c.reduce`, `x - max`
and the product with `1 / sum` by column-broadcast `c.tiles`, the exp and the
reciprocal by table ops on the reduced slot.

### Statistics over a row

Exact (fp32, vector-unit) row statistics use TT-Metal's two-pass statistics registers.
A tile is unpacked transposed (`c.transpose`), so each lane holds one row of the
original tile and the calls add over its 32 columns (`first`, `count`: a ragged last
tile's valid columns):

```python
c.init("stats")                      # the setup every c.stats_* call runs under
c.stats_clear()
c.stats_save(2)                      # spill to slots 2 and 3
with c.loop(wt, name="j") as j:
    c.transpose(cb_x, 0, 0)
    ...                              # any vector-unit work on the chunk
    c.init("stats")
    c.stats_restore(2)
    c.stats_sum(0, count=n)          # anchor=True on the first chunk of a row
    c.stats_save(2)
c.stats_restore(2)
c.stats_mean(inv_n)                  # the sums become the mean (fp32 bits of 1 / n)
c.stats_finalize(2, inv_n)           # mean in row 0 of slot 2, variance of slot 3
c.init("transpose_dest")
c.op("transpose_dest", 2)            # the row of means becomes column 0
```

`c.stats_m2(slot)` adds squared differences from the mean (from 0 after a clear: the
sum of squares an RMS norm takes). Without other vector-unit work between chunks the
save and restore are left out, and `dual=True` uses two accumulators (faster; a
restored state has one). The statistics live in vector-unit registers that every other
vector-unit function overwrites, so the verifier (STAT-1) refuses other vector-unit
work while they are there, a restore that no save of the same register scope matches,
and a write to the saved slots before the restore.

## Binding times

Every value in a program has one of these binding times, and the binding time decides
what a change of that value costs:

| Binding time | Examples | A new value costs |
|---|---|---|
| static | dtype, the op body, buffer page counts, declared `km.Static` parameters | a kernel compile |
| size class | the number of cores: the smallest of 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, the whole grid that covers the split's total (cores past it take no work) | one TT-Metal program build, no kernel compile |
| sized | `p.sized("w_pages", of=kt * nt, limit=128)`: computed on the host from runtime values and rounded up to a class (1, 2, 3, 4, 6, 8, 12, ..., in steps of at most 1.5x); it may be a buffer's page count. `exact=True` takes the value itself (one build per distinct value) | one TT-Metal program build, no kernel compile |
| runtime | tile counts, tensor addresses, `p.runtime(name)` values | nothing |
| runtime, from a fixed list | `p.runtime("block", choices=(1, 2, 3))`: a value that may also be a push / pop count, a loop step or a register slot; the verifier checks every choice | nothing |
| runtime, a declared multiple | `p.runtime("nr", multiple_of="block")`: checked at every launch, so a loop over it in steps of `block` is whole | nothing |
| per core | the core's index, from which it derives its own range (`p.split`), its items in turn (`p.stride`), or its place in a layout (`p.define("n0", p.core // per_group * nt)`) | nothing |
| data | a word loaded from L1 on the device (`slot.load32(i)`), such as a row index; it is never a count or a loop bound, and it selects what a NoC read fetches only with a declared range (`.bounded(low, high)`) | nothing |

The range of a data value is the launch's contract: nothing checks it on the device, so
an index outside its table reads other memory.

A buffer sized from a runtime value is declared through a sized value, so the buffer's
size is part of the program description and not of any kernel:

```python
a_pages = p.sized("a_pages", of=kc, limit=32)  # rounded up to a class
cb_a = p.cb(0, like=a, pages=2 * a_pages)
...
chunk = cb_a.reserve(a_pages)  # pushes count in the same value
for i in k.range(n):  # n <= kc <= a_pages tiles are read
    k.read(a.page(first + i), chunk[i])
```

The verifier checks a sized buffer for every value it can take, and takes a page
index as inside it when it is written so: a loop variable counting to at most the
buffer's pages, or a value declared `.bounded(0, pages)`. A read of `n` pages from an
index (a block copy, a matmul's `ct` tiles of b) needs the index declared
`.bounded(0, pages - n + 1)` with `n` provably at most `pages`, or static bounds that
fit. Indices are unsigned on the device: an index that can go below 0 (`t - 1` in a
loop from 0, `.bounded(-3, 1)`) is refused. A subtraction of two runtime values is
taken as not going below 0; that is the launch's contract, not checked.

Work that is not one even split is laid out by hand: `p.layout(cores=groups *
per_group)` states how many cores have work (the program runs on the smallest core
count class not below it, and every kernel returns at once on a core past it), and
each core computes its place from `p.core` with `p.define`.

So a new batch size, sequence length or image size compiles nothing: sizes are runtime
arguments and each core derives its share of the work on the device. Only what the
author declares static (a block size, a dtype) can cause a compile.

Measured on a Blackhole p150 with the two example programs, over 14 RoPE shapes and 10
SwiGLU shapes: no kernel compiled after the first call, 9 and 7 program builds (one per
core count used), output equal bit for bit to the hand-written kernels, warm time
within noise of them.

The matmul example (`examples/kernels/dsl_metal/matmul_km.py`: a layout of column
groups, two sized buffers) over 11 shapes: no kernel compiled after the first call, 9
program builds, output equal bit for bit to the backend's matmul program, warm time
within noise of it. It takes bf16 a and w only: like the backend's kernel it starts
the units in the regular source order, which is right for a matmul only when both
sources have one format (UNPACK-1 refuses a matmul of two formats in that order).

## Control words, semaphores and other cores

A kernel can hand values to the other kernels of its core through a buffer of 16-byte
pages, and loop until a value says stop:

```python
to_compute = p.words(24)  # control words, four pages
first, counter = p.core_ref(0), p.semaphore(0, name="next")

r = k.var("r", p.core)  # reader
with k.forever():
    k.put(to_compute, r)
    k.exit_when(r >= rows)
    ...  # read row r
    k.set(r, k.fetch_add(counter, at=first) + groups)

with c.forever():  # compute
    row = c.take(to_compute)
    c.exit_when(row >= rows)
    ...
```

`p.core_ref(i)` names core `i` of the program; its NoC coordinates are common
arguments the host fills when it builds the program. Core 0 always runs; another core
only when the layout's core count is a static value above `i`. On it,
`k.sem_inc(sem, at=ref)`, `k.fetch_add(sem, at=ref)` (the old value, through the
atomic's return word) and `k.send(slot, to=ref, size=n)` (a write of `n` bytes of the
reserve to the same L1 address there); on the own core `k.sem_wait_min(sem, n)` and
`k.sem_set(sem, v)`. Every increment returns its old value to one L1 word of the
issuing core: a kernel that fetches is the only one of its core that increments, and
waits for its earlier increments (`k.atomics_done()`) first.

The RMS norm example (`examples/kernels/dsl_metal/rmsnorm_km.py`: control words, a row
counter on core 0, the backend's two-pass statistics and vector-unit code as inline
C++) over 7 shapes (bf16 and fp32, with and without gamma, ragged rows and widths,
fewer and more tile rows than cores): output equal bit for bit to the backend's norm
forward with one core per tile row (`KTT_NORM_GROUP=1`), no kernel compiled for a new
shape of a variant already run.

## Variants, lanes and K-block passes

A data-movement role may run different kernels on different cores (rows `p.cores.rows`,
columns `p.cores.cols`, or both chained); together they cover the grid, each core
once, and compute has one kernel, named compute:

```python
top, below = p.cores.rows(0), p.cores.rows(1, p.machine.rows - 1)
send = p.kernel("dm0", cores=top, noc=0, name="w_send")
recv = p.kernel("dm0", cores=below, noc=0, name="w_recv")
```

A lane multicasts reserved pages from one sender row to the receiver rows of each
column, with the handshake of the backend's kernels written as statements:

```python
lane = p.lane(top, below, name="w")
lane.open(send)                 # the sender's valid flag, once, first
slot = send.reserve(cb_w, n)    # ... read into slot ...
lane.send(send, slot)           # wait for every receiver, multicast, flush, flag
send.push(cb_w, n)

recv.reserve(cb_w, n)
lane.receive(recv)              # clear the flag, ready to the sender, wait
recv.push(cb_w, n)
```

Turned 90 degrees, a lane runs from one sender column to receiver columns along each
row: `p.lane(p.cores.cols(0), p.cores.cols(1, p.machine.cols - 1))`. A 2-D block
matmul shares both operands with a row lane for a on one NoC and a column lane for w
on the other (tests/km_grid2d.py: the left column multicasts a's tiles along its row,
the top row w's tiles down its column, core (x, y) computes output tile (y, x)).

The rectangle and the sender's place are per-core arguments the host fills
(`p.core_ref(lambda x, y: (x, 0))` names a core relative to each core). `p.choose(names,
table)` declares runtime values chosen together (a subblock's rows and columns), so a
register check takes the largest product of a row of the table. A value is declared
once (again only with the same choices and multiple). `p.limit(value, low, high)`
bounds a runtime value, `high` possibly another one, at every launch.

`km.k_passes(k, nk, body)` runs the K blocks of a block matmul in the form every kernel
of it shares (one pass when `nk` is 1, else the first, the middle ones, the last), and
`km.matmul_passes(c, shape, cb_a, cb_w, part, cb_out)` is its compute side: per K block
the products of each subblock, summed into a partials buffer (`p.partials`, one output
block of fp32 pages) by packer accumulation onto the pages the pass before left, and
reloaded for the last K block. Both take their counts declared: `nk` at least 1,
`kb_last` at most `kb` (`p.limit`), the a and w blocks exact sized values of `bm * kb`
and `kb * bn`.

The block matmul example (`examples/kernels/dsl_metal/gemm_km.py`, the backend's block
matmul in its one-row-per-core layout) over 9 shapes (one, two and four K blocks, a short
last K block, ragged and batched sizes, w stored [N, K], 1 x 4 and 4 x 1 subblocks, the
latter with one, two and four K blocks):
output equal bit for bit to the backend's block matmul (`KTT_GEMM=1`) run with the
same plan.

## Tables: job lists per core

A table is a block of per-core arguments the host writes at every launch, for work
that differs per core and per call (a job list):

```python
jobs = p.table(                                # (src, dst, tiles) per row
    "jobs", width=3, rows=16, most={2: 4096}, pages={0: (x, 2), 1: (out, 2)}
)
with k.loop(jobs.used, name="j") as j, k.loop(jobs[j, 2], name="t") as t:
    ...  # read x.page(jobs[j, 0] + t)

launcher.run({"x": x}, {"out": ...}, tables={"jobs": [rows of core 0, ...]})
```

`rows` and `width` are static; each core reads its own block (`jobs.used` rows this
launch). `pages` {column: (tensor, count column)} declares the columns that index pages
of a tensor. The launch checks every word against its column's `most`, the row count
against `rows`, and, for each page column, that the column plus its count stays inside
the tensor's tile pages, before anything is queued; a new job list rewrites the
per-core arguments of the cached program and builds nothing. The table is read-only on
the device.

The job-list example (`examples/kernels/dsl_metal/segments_km.py`, packing ragged tile
ranges of x one after another) over 5 job lists (7 to 1600 segments, a segment the
size of x, two lists of one size on other jobs per core): output equal bit for bit to
the backend's `torch.cat` of the same slices and to the host's, no program built for
the second list of one size.

A table can also be a **stage table**: a column names what each row does, and every
kernel of the core switches on it, so one program runs a list of different stages per
core and a new list builds nothing:

```python
st = p.table("stages", width=4, rows=8, most={0: 1, 2: 2},
             pages={1: (x, 2), 3: (out, 2)})      # (stage, source, count, target)
with k.loop(st.used, name="j") as j:
    with k.when(st[j, 0].eq(0)):                  # stage 0
        ...
    with k.otherwise():                           # stage 1
        ...
```

The switch is an equality of a table word at the walk's row with a static value. Each
kernel that shares a buffer switches on the same word, so its pushes and pops are
counted per stage and must agree per stage (CB-2); a stage's own work between kernels
of other cores goes through `k.sync`.

## The verifier

`km.verify(program)` runs before anything is emitted; `km.VerifyError` names the rule.
`km.RULES` lists them. Each rule comes from a failure seen on the hardware:

| Group | Checks |
|---|---|
| binding | a value is used only where its binding time allows (a runtime value is never a buffer size, a loaded value never a count); a sized value and a layout's core count are computed from runtime values; a stepped loop runs over a multiple of its step |
| circular buffers | push and pop sizes divide the buffer; pushes equal pops across the kernels (inside a condition only when it is fixed for the launch and both kernels branch on it); a kernel writes, packs, loads and sends no further than it reserved (in-order packs are counted from their reserve unless a loop lies between them and it), and reads only pages it waited for; the buffers fit the L1 budget given (`km.verify(program, l1_budget=...)` or `km.Launcher(program, l1_budget=...)`; `km.L1_BUDGET` is the L1 an empty chip leaves) at every combination of sized values (past 50000 combinations, each buffer at its own most, a bound above the true most; the budget's 32 KiB margin covers the up to 63 bytes of padding before each buffer); at the first launch of each core count, numerics scope and sizes through `km.Launcher` or a `km.op`, before anything is queued, the buffers and scratch of the sized values in use, each placed at a multiple of 64 bytes, fit the L1 free on the device then (32 KiB below the lowest live L1 tensor, within `KTT_L1_BUDGET`), else the launch raises `ValueError` naming the bytes needed and free; at every launch (repeat launches, a `km.op` call the backend replays without Python, and launches inside a trace capture included), on the stream before it is enqueued, the buffers as the device places them end below the lowest live L1 buffer and the end of the L1 bank (TT-Metal's own check lets them reach into the `KTT_L1_SMALL` region above it), else nothing is launched and the next synchronizing call raises `RuntimeError` "user program NAME: ... take N bytes of L1 per core, M are free" (later launches run once the L1 is freed); a trace replay is checked by the trace's own L1 guard ([trace](trace.md)) |
| NoC | reader and writer use different NoCs; a sub-page read keeps source and destination aligned alike, proved from what the offsets are multiples of; a push, a load or a write follows its read barrier; a page is flushed before it is popped or read into again; a semaphore increment or fetch on another core, or a lane's flag, follows the flush of what this kernel sent before it; a kernel ends with its increments landed |
| tables | a table is read at a static row below its size or at the variable of a loop over its used rows (that loop, by identity); a column used as a count declares its most; a table word is otherwise used only as a NoC page index `t[j, c]` or `t[j, c] + v` of the tensor its column is declared for, with v the variable of a loop over the declared count column of the same row (step 1, from 0); no page count comes from a table; kernels sharing a buffer count it from the same table words (the pushes and pops of CB-2 match by word) |
| variants and lanes | variants of a role cover the grid, each core once; pushes and pops are compared between the kernels that share a core; a lane has one sending kernel on its sender row (column) and receiving kernels on every receiver core, its rectangle lies in the grid without the sender and every core of it runs (no layout or split), its semaphores are touched only by its own statements, it is opened once before it sends, and it works under counts and conditions equal on every core; a lane's data moves only inside a reserve on both sides and is read in before it is sent; the receivers reserve, push and receive in step with the sender (the same calls on its buffer and the same sends and receives of every lane carrying it, in order, under the same loops and conditions) with at most one send or receive per lane in a reserve, and do not touch the pages it fills; a core named relative to each core is a core the program runs on |
| control | an endless loop has one exit, on a control word it takes, or puts before it and leaves unchanged up to it, compared with a value fixed for the launch; a loop ending on taken words is fed by the one loop that puts them, one per iteration, before ending on the same comparison; pushes and pops in endless loops are counted per stream of words; a wait on a semaphore is accepted only when the other kernel of the core raises it by increments the verifier counts (no condition around them, static loop counts) to at least the static value waited for; a semaphore exists on the core it is signalled on; endless loops sit at the top of a kernel; a core named as a target runs at every launch; a fetch has no other increment of its core in flight; only data-movement kernels touch semaphores |
| compute | register slots fit the mode (8 bf16 or 4 fp32 tiles per half), a matmul block's `ct * rt` slots included; math sits between acquire and commit, packs between commit and release; a slot that a table op, a vector-unit binary, a generated function or a pack reads was written earlier in its register scope (an unpack, a matmul, a fill or an op's result; a loop is taken to run at least once, and after a hand-written vector function or inline code nothing more is checked in that scope), so a statistics slot kept across a loop is filled before it; a whole-tile op or a reduction runs under its own setup as the unpacker's last one; while a reduction's setup is in effect (until `c.reduce_done()`, which the kernel reaches before it ends) a pack goes into the reduction's output buffer and packs a slot a reduction along the same dim wrote last (the packer masks keep only that row or column of every tile), and no other way of unpacking is set up; a way of unpacking, a table op and a matmul each run under their own setup; an unpack reads a buffer of the format its source register was set to (`c.startup`, `c.format`; a matmul reads b through the first source, a whole-tile op a through the first and b through the second, a row sum its scaler through the first and a reduction otherwise its buffer through the first); a pack goes into a buffer of the packer's format; packer accumulation adds only to a page packed earlier in the same reserve (switched on after a loop's first iteration: the loop counts from 0 in steps of 1, the buffer is reserved outside it, and the pack sits in it at a static page under no condition inside it) (a push moves page `i` to `i - n` and restarts in-order packs; a page named by a `k.var` stops counting once the variable is set; a loop of packs at its variable over a static count counts each page) and is off at exit; a setup or format made only inside a loop that may run zero times does not count after it; a changed vector-unit constant is restored at exit |
| page indices (at launch, PAGE-1) | at the first launch of each core count, runtime values and tensor page counts through `km.Launcher` or a `km.op` (a `km.op` call the backend replays has the same values), before anything is queued: every NoC page index of a tensor and every `.bounded(low, high)` value computed from runtime values, sized values, the core index, loop variables and definitions is bounded on every core the kernel runs on by interval arithmetic (loops enumerated up to 16384 elements per kernel, the rest as the range of their variable; a `when` and its else, and an endless loop's exit for the statements after it, narrow the values they compare, a word from L1 or a buffer included, until the value is set, taken or loaded again); a page past the tensor's tile pages (or below 0) or a value outside its declared range that is reached for sure raises `ValueError` "PAGE-1: KERNEL: page index ... reaches N on core C; T has P pages"; one the intervals cannot exclude is warned once per program (`km.PageWarning`), and refused under `KTT_KM_PAGE_CHECK=strict`; a page index read from a table is left to the table words' check; repeat launches cost one dictionary lookup |
| rows | tilize reads 32 pages of one row of the block each and writes tiles, untilize the reverse with a block that fits the registers, both outside a register scope |
| statistics | a `c.stats_*` call runs under `c.init("stats")` on statistics in the registers (after a clear, or a restore of what a save of the same register scope spilled); no other vector-unit work runs while they are there, the saved slots are not written until the restore, a restored state takes one accumulator |
| debug | a profiler zone's name is plain text (letters, digits, space and `_ . : / -`) and nothing declared inside a zone is used after it; a watcher waypoint names 1 to 4 letters or digits |
| inline C++ | the buffers, semaphores and registers it touches are declared; in vector-unit instruction macros, the result of a multiply-add, multiply, add or table lookup is not read by the next instruction through a form the hardware does not hold back (a min / max swap, a shift, an integer add, a set-exponent / mantissa / sign, a partial load, a constant write) |

Inline C++ (`k.inline`, `km.SfpuFn`) is the way to write anything the typed operations
do not cover; the verifier checks its declared effects and the spacing of its
vector-unit instructions as written, not what the code computes.

### What the verifier does not check

Not checked:

- that a write sent to another core with `k.send` lands in pages that core reserved
  before it signalled ready (a lane's multicast is checked, NOC-5);
- that the pages `k.fetch` reads hold what the other core put there, and stay held
  until the read lands (a `k.sync` between the stages is the program's to place);
- that every core reaches a `k.sync` at all: a core blocked on a buffer that another
  core's progress past the barrier would free hangs the barrier;
- that a program cannot deadlock in general: its first waits forming no cycle, and a
  producer that runs ahead of its consumers by more than a buffer holds;
- waits that grow while pages stay unpopped (there is no typed form of them yet);
- a value declared with `.bounded(low, high)` that depends on a word loaded from L1
  (`slot.load32`) or taken from a buffer (`k.take`): the range is the launch's contract,
  known on the device only (PAGE-1 checks a page index built from it against the
  declared range, not the word itself); a matmul block's tile span is proved from such
  a declared index range and the declared multiples of the block sizes;
- a page index PAGE-1 cannot prove (warned, refused only under
  `KTT_KM_PAGE_CHECK=strict`), one read from a table word outside the declared form
  (refused by the verifier), and a word taken from a buffer or fetched from a counter
  that no condition bounds (warned as known on the device only); a tensor whose pages
  are not tiles of its last two dims is not covered;
- what inline C++ (`k.inline`, `km.SfpuFn`) does beyond its declared buffers,
  semaphores, constants and arguments, and the spacing rules of its vector-unit macros;
- what a program computes: rounding, fidelity and formats are checked for consistency
  with the setup, not against a reference;
- that semaphores start from their initial value on a cached relaunch (the native side
  writes them; the verifier assumes it);
- variants against the cores a size class launches: they are checked against the
  whole grid, so a variant is required on cores no class launches;
- the L1 at build time when no `l1_budget` is given (`km.Launcher` and `km.op` give
  none by default: `examples/kernels/dsl_metal/gemm_km.py` declares block sizes taking
  up to 5.3 MB per core and plans each launch below 1 MiB), so a program whose
  declared most never fits is refused only at a launch whose sizes do not fit;
- a repeat launch against `KTT_L1_BUDGET` and the 32 KiB margin: it is checked
  against the L1 actually free below the lowest live L1 buffer, so it runs where the
  first launch of the same sizes would have been refused;
- a trace replay when `KTT_TRACE_L1_GUARD=0`: the replay then writes its buffers over
  any L1 tensor allocated inside their region after the capture.

Refused until they can be checked: a semaphore wait raised by another core, a set or
inline code (outside a lane); an endless loop inside a loop or a condition; a table
row other than a static index or the variable of a loop over the table's used rows; a
table column used as a count without a declared most; a page count taken from a table;
a table word anywhere but a loop count, a declared page index of the form above or a
stage switch (through a variable, in an offset, an address, another condition or a
semaphore operation).

Refused by construction (the model has no form for them): a multicast outside a lane;
a lane of several sender rows or columns; compute variants, and a kernel named after another
role; two semaphores, core references, lanes, tables or buffers of one name, and a
value declared twice differently; a table sized by anything but a static int, and any
device write to a table; any call on a partials buffer outside `km.matmul_passes`,
inline code's included (its ring alignment, packs that add onto the pages the pass
before left, holds by construction and is not otherwise checked).

A loop variable is keyed by its name and the range it walks, so kernels match loops
that name their variables alike; a table word by its table, row and column.

## Data held in L1 across stages

A program can run in stages over data that stays in L1: every core holds its part in
a reserve, all cores meet at a barrier, and each then reads what another core holds:

```python
point = p.sync_point()                       # a barrier of every core of the program
right = p.core_ref(lambda x, y: ((x + 1) % cols, y))
mine = k.reserve(cb_mine, 1)
k.read(x.page(p.core), mine)
k.reads_done()
k.sync(point)                                # every core holds its tile
got = k.reserve(cb_got, 1)
k.fetch(mine, right, got, x.page_bytes)      # the neighbour's tile: same pages there
k.reads_done()
k.push(cb_got, 1)
k.sync(point)                                # the neighbours are done with mine
k.push(cb_mine, 1)
```

`k.sync(point)` waits for this kernel's writes and reads to land, adds one to an
arrival count on core 0 and polls it until every core running the program (the
layout's count, else the launch's) has arrived as often. `p.sync_point(group=g)`
makes the barrier per group of `g` consecutive cores (counted on each group's first
core), and `point.member(i)` names member `i` of the running core's group; the launch
refuses a core count that is not whole groups. An exchange of partial results inside
a group is then a sync and a fetch from each member:

```python
point = p.sync_point(group=4)
k.sync(point)                                   # the group holds its partials
every = k.reserve(cb_all, 4)
for i in range(4):
    k.fetch(mine, point.member(i), every[i], page_bytes)
k.reads_done()
k.push(cb_all, 4)
k.sync(point)                                   # the group has read mine
```
`k.fetch(src, at, dst, n)` reads `n` bytes from core `at` at the address of this
core's slot `src`, so it reads the pages that core holds in the same buffer. A slot
is a reserve, or the waited pages `mine = k.front(cb, n)` (a wait returning the front
pages) when the compute produced them: a partial statistic packed by the compute is
fetched from there, and popped after the second sync. Waited pages are a source only
(written out, sent, fetched from) and are used before the next pop of their buffer
(CB-3, CB-4). SYNC-1: a point's syncs sit in
one data-movement kernel that runs on every row, outside conditions and in loops whose
counts are one value for every core, and nothing else touches its semaphore.

## Row-major data

A tensor may be read or written by rows: `p.input("x", "bfloat16", rows=W)` (or
`p.output`) names a tensor whose last dim is the static `W`; page `i` is its row `i`
(`x.page_bytes` bytes). The launch hands the program a row-major copy of the input and
turns a row-major output back into tiles (one TT-NN layout conversion each), and
checks the tensor's last dim against `W`. A buffer of one-row pages
(`p.cb(..., page=x.page_bytes)`) holds the rows; the compute kernel converts 32 rows
of `block` tiles' width (W = 32 * block) to and from tiles outside a register scope:

```python
c.wait(cb_rows, 32)                 # 32 one-row pages
c.reserve(cb_tiles, block)
c.tilize(cb_rows, cb_tiles, block)  # tilize_init / tilize_block / tilize_uninit
c.push(cb_tiles, block)
...
c.reserve(cb_back, 32)
c.untilize(cb_tiles, cb_back, block)  # pack_untilize, block at most the registers
c.push(cb_back, 32)
```

Both leave the unpacker's and packer's setup and formats unset for what follows
(`c.format`, `c.pack_format`, `c.init` again). fp32 data keeps its bits only with
both buffers in the kernel's `unpack_fp32` (fp32 registers); otherwise it passes the
unpacker's 19-bit source registers. The verifier checks the page sizes and
the block (ROW-1) and counts the pages written like packs (CB-3).

## Profiler zones and watcher waypoints

Any kernel can name a stretch of its statements as a device-profiler zone and mark
places a hung core reports:

```python
with k.loop(count) as i:
    k.waypoint("RD")              # 1 to 4 letters or digits
    slot = k.reserve(cb_x, 1)
    with k.zone("read"):          # DeviceZoneScopedN("read") around the statements
        k.read(x.page(start + i), slot)
        k.reads_done()
    k.push(cb_x, 1)
```

A zone is a C++ block in the generated source, so a value, variable or reserved page
declared inside it is used only inside it (DBG-1). Both compile to nothing in a plain
run: zones are recorded with `TT_METAL_DEVICE_PROFILER=1` (the zone's name is the
`zone name` column of the device log), waypoints are written with `TT_METAL_WATCHER`
set and shown in its report for a core that hangs. In a compute kernel a zone or a
waypoint is recorded by each of the three compute threads.

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
