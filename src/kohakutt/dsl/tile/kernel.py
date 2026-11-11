"""`@kd.kernel`: a traced tile-level kernel as a callable on tt tensors. A kernel class
(dtypes, static values, how rows are kept) is traced, planned and lowered once; its
program runs as a torch op whose launch the backend remembers per sizes.
"""

import inspect
import itertools
import math
import struct

import kohakutt.dsl.metal as km
from kohakutt.dsl.metal.model import PAGE_CLASSES
from kohakutt.dsl.tile import lower as lowering
from kohakutt.dsl.tile import plan as planning
from kohakutt.dsl.tile.ir import (
    BF16,
    F32,
    KernelIR,
    PlanError,
    TensorParam,
    ValueParam,
    fail,
)
from kohakutt.dsl.tile.matmul import lower_matmul, matmul_plan
from kohakutt.dsl.tile.trace import (
    Auto,
    Out,
    Runtime,
    Static,
    Tensor,
    TensorProxy,
    Tracer,
    _Typed,
)

TORCH_DTYPES = {"bfloat16": BF16, "float32": F32}
# The torch op's value of a static left to its kd.auto candidates.
AUTO = -1
_OPS = {}


class Like:
    """A tensor described by sizes and dtype only, for plans without a device."""

    def __init__(self, shape, dtype="bfloat16"):
        self.shape, self.dtype = tuple(shape), dtype


def like(shape, dtype="bfloat16"):
    """A stand-in tensor of `shape` and `dtype` ("bfloat16" / "float32", or a torch
    dtype) for `kernel.plan`, `kernel.metal` and `kernel.ir` without a device."""
    return Like(shape, str(dtype).replace("torch.", ""))


def _format(tensor):
    name = str(tensor.dtype).replace("torch.", "")
    if name not in TORCH_DTYPES:
        fail(f"tensor dtype {tensor.dtype}: the tile level takes bfloat16 and float32")
    return TORCH_DTYPES[name]


def _fp32_bits(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _recip_bits(value):
    """The fp32 bits of 1 / value, divided in fp32 as torch would."""
    import torch

    one = torch.tensor(1.0, dtype=torch.float32)
    q = one / torch.tensor(float(value), dtype=torch.float32)
    return _fp32_bits(q.item())


def _annotation(fn, p):
    a = p.annotation
    if isinstance(a, str):
        a = eval(a, fn.__globals__)
    if a is Tensor or a is Out:
        return a, None
    if isinstance(a, _Typed):
        return a.kind, a.item
    if a is inspect.Parameter.empty and isinstance(p.default, Auto):
        return Static, int
    fail(
        f"parameter {p.name}: annotate it kd.Tensor, kd.Out, kd.Runtime[...] or "
        "kd.Static[...]"
    )


class _Variant:
    """One built program of a kernel class: its design, km program and values."""

    def __init__(self, ident, design, program, values, mode, depth, seg_limit):
        self.ident, self.design, self.program = ident, design, program
        self.values, self.mode, self.depth = values, mode, depth
        self.seg_limit = seg_limit


class Kernel:
    """A tile-level kernel: call it on tt tensors; `plan(...)`, `metal(...)`,
    `ir(...)` and `source(..., role)` show what a call becomes."""

    def __init__(self, fn, name=None, dest=None):
        if dest not in (None, "fp32", "bf16"):
            fail(f"dest is 'fp32', 'bf16' or None, not {dest!r}")
        self.fn, self.name, self.dest = fn, name or fn.__name__, dest
        self.inputs, self.outputs, self.values, self.statics = [], [], [], {}
        self.order = []
        for p in inspect.signature(fn).parameters.values():
            kind, item = _annotation(fn, p)
            if kind is Out:
                self.outputs.append(p.name)
                continue
            self.order.append(p.name)
            if kind is Tensor:
                self.inputs.append(p.name)
            elif kind is Runtime:
                self.values.append(
                    ValueParam(p.name, "int" if item is int else "float")
                )
            else:
                self.statics[p.name] = p.default
        if not self.inputs or not self.outputs:
            fail(f"{self.name}: a kernel takes a kd.Tensor and a kd.Out")
        self._classes, self._variants, self._op = {}, [], None

    # Tracing and planning per kernel class.
    def _trace(self, dtypes, statics):
        tensors = [TensorParam(n, False, dtypes[n]) for n in self.inputs]
        tensors += [TensorParam(n, True, dtypes[n]) for n in self.outputs]
        ir = KernelIR(self.name, tensors, list(self.values), dict(statics))
        args = {t.name: TensorProxy(t) for t in tensors}
        args.update({v.name: v for v in self.values})
        args.update(statics)
        with Tracer(ir):
            self.fn(**args)
        return ir

    def _class(self, dtypes, statics, machine):
        """The variants of kernel class (`dtypes`, `statics`): Auto statics take
        their first candidate whose program verifies within the L1 budget."""
        key = (tuple(sorted(dtypes.items())), tuple(statics.items()), machine)
        found = self._classes.get(key)
        if found is not None:
            return found
        autos = [n for n, v in statics.items() if isinstance(v, Auto)]
        error = None
        for pick in itertools.product(*(statics[n].candidates for n in autos)):
            chosen = {**statics, **dict(zip(autos, pick))}
            try:
                design = planning.design(self._trace(dtypes, chosen), self.dest)
                variants = self._build(design, machine)
            except (km.VerifyError, PlanError) as e:
                error = error or e
                continue
            self._classes[key] = variants
            return variants
        raise error

    def _build(self, design, machine):
        if design.form == "matmul":
            if design.share:
                from kohakutt.dsl.tile.shared import lower_grid2d, lower_shared

                both = design.a.node.attrs.get("share") == "row"
                lower = lower_grid2d if both else lower_shared
                program, values = lower(design, machine)
                # Its block plan keeps each call's buffers within 1 MB.
                km.verify(program)
            else:
                program, values = lower_matmul(design, machine)
                km.verify(program, l1_budget=km.L1_BUDGET)
            variant = _Variant(
                len(self._variants), design, program, values, "matmul", 1, 0
            )
            self._variants.append(variant)
            return [variant]
        held = any(op.held for op in design.operands)
        rows = design.form in ("row", "columns", "flat")
        lower = lowering.lower
        if design.form == "flat":
            from kohakutt.dsl.tile.flat import lower_flat

            def lower(design, machine, block, mode, depth, limit):
                return lower_flat(design, machine, limit)

            held, modes = True, [("hold", 1)]
        if design.form == "columns":
            from kohakutt.dsl.tile.columns import lower_columns

            def lower(design, machine, block, mode, depth, limit):
                return lower_columns(design, machine, mode, depth, limit)

        if design.form != "flat":
            modes = [("hold", 2), ("hold", 1), ("reread", 1)]
            modes = modes if held else [("stream", 1)]
        if held and any(op.random for op in design.operands):
            modes = modes[:2]
        out = []
        if design.ir.grid.group > 1:
            from kohakutt.dsl.tile.wide import lower_wide

            # Rows split over core groups first; a call takes them when they fit.
            for limit in reversed([c for c in PAGE_CLASSES if c <= planning.SEG_LIMIT]):
                program, values = lower_wide(design, machine, limit)
                if program.footprint() <= km.L1_BUDGET:
                    km.verify(program, l1_budget=km.L1_BUDGET)
                    variant = _Variant(
                        len(self._variants), design, program, values, "wide", 1, limit
                    )
                    self._variants.append(variant)
                    out.append(variant)
                    break
        for mode, depth in modes:
            limits = (
                [c for c in PAGE_CLASSES if c <= planning.SEG_LIMIT] if rows else [0]
            )
            for limit in reversed(limits):
                program, values = lower(
                    design, machine, design.ir.grid.block, mode, depth, limit
                )
                if program.footprint() <= km.L1_BUDGET:
                    km.verify(program, l1_budget=km.L1_BUDGET)
                    ident = len(self._variants)
                    variant = _Variant(
                        ident, design, program, values, mode, depth, limit
                    )
                    self._variants.append(variant)
                    out.append(variant)
                    break
            if not rows:
                break
        if not out:
            fail(f"{self.name}: no plan fits L1")
        return out

    # One call.
    def _bind(self, args, kwargs):
        names = list(self.order)
        if len(args) > len(names):
            fail(f"{self.name} takes {len(names)} arguments")
        bound = dict(zip(names, args))
        out_dtype = kwargs.pop("out_dtype", None)
        for name, value in kwargs.items():
            if name not in names or name in bound:
                fail(f"{self.name}: unexpected argument {name}")
            bound[name] = value
        for name, default in self.statics.items():
            if name not in bound:
                if default is inspect.Parameter.empty:
                    fail(f"{self.name}: static {name} is not given")
                bound[name] = default
        missing = [n for n in names if n not in bound]
        if missing:
            fail(f"{self.name}: missing arguments {missing}")
        return bound, out_dtype

    def _call(self, bound, out_dtype, machine, free=None):
        """(variant, km.Call, env) of one call with bound arguments."""
        first = bound[self.inputs[0]]
        dtypes = {n: _format(bound[n]) for n in self.inputs}
        out_format = TORCH_DTYPES.get(str(out_dtype).replace("torch.", ""))
        for n in self.outputs:
            dtypes[n] = out_format or dtypes[self.inputs[0]]
        statics = {n: bound[n] for n in self.statics}
        variants = self._class(dtypes, statics, machine)
        design = variants[0].design
        like = getattr(design, "out_like", {})
        wanted = {
            n: dtypes[src] if isinstance(src, str) else F32 for n, src in like.items()
        }
        if any(dtypes[n] != d for n, d in wanted.items()):
            # Column outputs take the dtype of the first column input (else fp32).
            dtypes.update(wanted)
            variants = self._class(dtypes, statics, machine)
            design = variants[0].design
        shape = list(first.shape)
        if design.form == "matmul":
            left = design.prologue.x if design.prologue else design.a.tensor
            shape = [*bound[left.name].shape[:-1]]
            shape.append(bound[design.w.tensor.name].shape[-1])
        env = {n: tuple(bound[n].shape) for n in self.inputs}
        env.update({n: tuple(shape) for n in self.outputs})
        for n, src in like.items():
            if isinstance(src, tuple):
                # Sums over rows: the last dim of the run's tensor.
                env[n] = (bound[src[1]].shape[-1],)
            else:
                env[n] = tuple(bound[src].shape) if src else (*shape[:-1], 1)
        torch_dtypes = {BF16: "bfloat16", F32: "float32"}
        env.update({v.name: bound[v.name] for v in self.values if v.kind == "int"})
        for check in design.checks:
            _check(check, env)
        variant = self._pick(variants, env, free, machine.cores)
        runtime = {name: int(e.eval(env)) for name, e in variant.values.host.items()}
        for name, fn in variant.values.extra.items():
            runtime[name] = int(fn(runtime, machine.cores))
        for name, s in variant.values.scalars.items():
            if isinstance(s.value, ValueParam):
                raw = bound[s.value.name]
            else:
                raw = s.value.eval(env)
            runtime[name] = _recip_bits(raw) if s.recip else _fp32_bits(raw)
        torch_out = out_dtype or first.dtype

        def out_type(n):
            if n not in like:
                return torch_out
            name = torch_dtypes[dtypes[n]]
            if isinstance(torch_out, str):
                return name
            import torch

            return getattr(torch, name)

        call = km.Call(
            inputs={r.name: bound[r.name] for r in variant.program.inputs},
            outputs={n: (list(env[n]), out_type(n)) for n in self.outputs},
            runtime=runtime,
            static={"variant": variant.ident},
        )
        return variant, call, env

    def _pick(self, variants, env, free, cores=0):
        if variants[0].mode in ("stream", "matmul"):
            return variants[0]
        d = variants[0].design
        row = getattr(d, "row_tensor", None)
        if d.form == "flat":
            wt = int(d.length.eval(env))
        else:
            out = row.name if row is not None else d.stores[0][0].view.tensor.name
            wt = -(-int(env[out][-1]) // 32)
        for v in variants:
            tiles = wt
            if v.mode == "wide":
                g = d.ir.grid.group
                if d.ir.grid.count.eval(env) * g > cores:
                    continue
                tiles = -(-wt // g)
            need = next((c for c in PAGE_CLASSES if c >= tiles), None)
            fits = need is not None and v.seg_limit >= need
            if fits and (free is None or _l1(v, tiles) <= free):
                return v
        fail(f"{self.name}: a row of {wt} tiles fits no plan's L1")

    def __call__(self, *args, **kwargs):
        bound, out_dtype = self._bind(args, kwargs)
        op = self._torch_op()
        code = {None: 0, "bfloat16": 1, "float32": 2}[
            None if out_dtype is None else str(out_dtype).replace("torch.", "")
        ]
        args = [AUTO if isinstance(bound[n], Auto) else bound[n] for n in self.order]
        return op(*args, code)

    def _torch_op(self):
        if self._op is not None:
            return self._op
        import torch

        from kohakutt.dsl.metal.bridge import l1_free

        kinds = {"float": "float", "int": "int"}
        params = []
        for name in self.order:
            if name in self.inputs:
                params.append(f"Tensor {name}")
            elif name in self.statics:
                default = self.statics[name]
                kind = "float" if isinstance(default, float) else "int"
                params.append(f"{kind} {name}")
            else:
                v = next(v for v in self.values if v.name == name)
                params.append(f"{kinds[v.kind]} {name}")
        params.append("int out_code")
        result = (
            "Tensor"
            if len(self.outputs) == 1
            else ("(" + ", ".join(["Tensor"] * len(self.outputs)) + ")")
        )
        qualified = f"kohakutt_tile::{self.name}"
        if qualified in _OPS:
            fail(f"two tile-level kernels named {self.name}")
        _OPS[qualified] = self
        schema = f"{qualified}({', '.join(params)}) -> {result}"
        dtypes = {0: None, 1: torch.bfloat16, 2: torch.float32}

        def builder(*args):
            bound = dict(zip(self.order, args[:-1]))
            for name, default in self.statics.items():
                if bound[name] == AUTO and isinstance(default, Auto):
                    bound[name] = default
            cols, rows = torch.ops.ktt.compute_grid()
            _, call, _ = self._call(
                bound, dtypes[int(args[-1])], km.Machine(cols, rows), l1_free()
            )
            return call

        km.op(schema, _Template(self))(builder)
        self._op = builder.op
        return self._op

    # What a call becomes, without running it.
    def _offline(self, args, kwargs):
        machine = kwargs.pop("machine", None)
        bound, out_dtype = self._bind(args, kwargs)
        machine = machine or _machine(bound, self.inputs)
        return self._call(bound, out_dtype, machine), machine

    def plan(self, *args, machine=None, **kwargs):
        """The Plan of a call with these arguments (tt tensors or kd.like)."""
        kwargs = dict(kwargs)
        if machine is not None:
            kwargs["machine"] = machine
        (variant, call, env), machine = self._offline(args, kwargs)
        return _plan(variant, call, env, machine)

    def metal(self, *args, **kwargs):
        """The metal-level program of a call with these arguments, as text."""
        (variant, _, _), _ = self._offline(args, dict(kwargs))
        return km.dump(variant.program)

    def program(self, *args, **kwargs):
        """The km.Program of a call with these arguments."""
        (variant, _, _), _ = self._offline(args, dict(kwargs))
        return variant.program

    def source(self, *args, role="compute", **kwargs):
        """The C++ source of kernel `role` ("dm0", "compute", "dm1") of a call."""
        return km.source(self.program(*args, **kwargs), role)

    def ir(self, *args, **kwargs):
        """The tile IR of a call's kernel class, as text."""
        (variant, _, _), _ = self._offline(args, dict(kwargs))
        return variant.design.ir.dump()


def _l1(variant, wt):
    """Bytes of L1 per core of `variant` for rows of `wt` tiles."""
    seg = next(c for c in PAGE_CLASSES if c >= wt)
    return variant.program.footprint({"seg": seg})


def _machine(bound, inputs):
    """The device's grid when the first input lives on a tt device, else 11 x 10."""
    first = bound[inputs[0]]
    if getattr(getattr(first, "device", None), "type", None) == "tt":
        import torch

        return km.Machine(*torch.ops.ktt.compute_grid())
    return km.Machine()


def _check(check, env):
    kind, a, b, line = check
    if kind == "multiple":
        value = a.eval(env)
        if value % b:
            fail(f"{a.text()} = {value} is not a multiple of {b}", line)
    elif kind == "rows":
        value = a.eval(env)
        if value > b and value % b:
            fail(
                f"{a.text()} = {value}: a run kernel takes heights <= 32 or whole tiles"
            )
    else:
        x, y = a.eval(env), b.eval(env)
        if x != y:
            fail(f"{a.text()} = {x} differs from {b.text()} = {y}", line)


class _Template:
    """The metal level's view of a kernel: `trace` returns the program of a variant
    (km.op keys its launchers by dtypes and the variant)."""

    def __init__(self, kernel):
        self.kernel = kernel

    def trace(self, machine, dtypes, variant):
        return self.kernel._variants[variant].program


def _plan(variant, call, env, machine):
    d = variant.design
    program = variant.program
    runtime = dict(call.runtime)
    if d.form == "matmul":
        return matmul_plan(d, program, runtime, env)
    units = d.ir.grid.count.eval(env)
    cores = program.core_count(runtime)
    read = written = 0
    sizes = program.sizes(runtime)
    for op in d.operands:
        t = op.tensor
        per = planning.TILE_BYTES[t.dtype]
        if op.resident:
            row = _tiles(env[t.name])[1] if op.row else 1
            # Split rows: each core reads its slice of a resident row.
            read += row * per * (units if variant.mode == "wide" else cores)
        elif d.form == "flat":
            from kohakutt.dsl.tile.flat import run_length

            read += units * run_length(op.view).eval(env) * per
        else:
            times = len(op.passes) if (op.row and not _held(variant, op)) else 1
            tiles = units * (_tiles(env[t.name])[1] if op.row else 1)
            read += tiles * per * max(1, times)
    for s, _ in d.stores:
        t = s.view.tensor
        written += math.prod(_tiles(env[t.name])) * planning.TILE_BYTES[t.dtype]
    for s, _ in getattr(d, "unit_stores", ()):
        written += units * planning.TILE_BYTES[s.view.tensor.dtype]
    tiles_out = sum(math.prod(_tiles(env[s.view.tensor.name])) for s, _ in d.stores)
    cycles = (
        sum(planning.vector_cost(c.chain) for _, cases in d.stores for c in cases[:1])
        * tiles_out
        / max(1, len(d.stores))
    )
    if d.form == "columns":
        row_tiles = _tiles(env[d.row_tensor.name])[1]
        for s, chain in d.col_stores:
            written += units * planning.TILE_BYTES[s.view.tensor.dtype]
            cycles += units * planning.vector_cost(chain)
        for value in d.values:
            if value.kind == "exact" and id(value) in d.fused:
                cycles += units * 32 * 4
            elif value.kind == "exact":
                per_tile = planning.vector_cost(value.chain) + 32 * 2
                cycles += units * row_tiles * per_tile
            elif value.kind == "chain":
                cycles += units * planning.vector_cost(value.chain)
    if d.stat is not None:
        reduced = math.prod(_tiles(env[d.reduce["operand"].tensor.name]))
        cycles += units * planning.vector_cost(d.stat)
        cycles += reduced * 32 * planning.COST["reduce"]
    from kohakutt.dsl.metal.bridge import l1_bytes, l1_free

    l1 = l1_bytes(program, sizes)
    try:
        free = l1_free()
    except RuntimeError:
        free = None
    return planning.Plan(
        d,
        env,
        units,
        cores,
        d.ir.grid.block,
        variant.mode,
        l1,
        free,
        read,
        written,
        cycles,
        program,
    )


def _held(variant, op):
    return op.row and (op.resident or (op.held and variant.mode in ("hold", "wide")))


def _tiles(shape):
    shape = [1, 1, *shape][-max(2, len(shape)) :]
    lead = math.prod(int(n) for n in shape[:-2])
    return lead * -(-int(shape[-2]) // 32), -(-int(shape[-1]) // 32)


def kernel(fn=None, *, name=None, dest=None):
    """Decorates a kernel function `fn(tensors..., out..., runtime..., static...)`:
    the result is a Kernel callable on tt tensors. `dest` ("fp32", "bf16") fixes the
    destination register format; by default it follows the kernel's operands."""
    if fn is None:
        return lambda f: Kernel(f, name, dest)
    return Kernel(fn, name, dest)
