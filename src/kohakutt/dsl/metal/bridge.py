"""Runs metal-level programs as torch ops on tt tensors: the program description for
`torch.ops.ktt.run_program` (free of data sizes) and the `op` registration.
"""

import json

import torch

from kohakutt.core.numerics import Requirements, current_numerics, resolve
from kohakutt.dsl.metal.binding import as_expr
from kohakutt.dsl.metal.emit import emit
from kohakutt.dsl.metal.machine import Machine, ranges
from kohakutt.dsl.metal.model import CoreArg, TableArg
from kohakutt.dsl.metal.verify import DEFAULT_NOC, _counts, cb_ops, verify
from kohakutt.kernels import _program_op

# torch dtype -> (storage format name, ScalarType code).
DTYPES = {
    torch.bfloat16: ("bfloat16", 15),
    torch.float32: ("float32", 6),
    torch.int32: ("int32", 3),
}
KINDS = {"dm0": "reader", "compute": "compute", "dm1": "writer"}
_LIBRARIES = {}


class Call:
    """One launch of a program: `inputs` {name: tt tensor}, `outputs` {name: (sizes,
    torch dtype)}, `runtime` {name: int}, `static` {name: value} and `tables` {name:
    rows per core}."""

    def __init__(self, inputs, outputs, runtime=None, static=None, tables=None):
        self.inputs, self.outputs = dict(inputs), dict(outputs)
        self.runtime, self.static = dict(runtime or {}), dict(static or {})
        self.tables = dict(tables or {})


def _page_counts(pages, sizes):
    """The page counts a buffer call of `pages` makes with sized values `sizes`: one
    value, or every choice of a runtime value from a list."""
    try:
        return (int(as_expr(pages).eval(sizes)),)
    except KeyError:
        return _counts(pages, "a page count")


def rings(program, sizes=None):
    """Per pushed buffer: its index, the page counts of its recurring calls and the
    most bytes a kernel writes from a reserved pointer, for sized values `sizes`."""
    out = {}
    for kernel in program.kernels.values():
        for kind, cb, pages, path, times, _stmt in cb_ops(kernel):
            ring = out.setdefault(
                cb.index, {"cb": cb.index, "pushes": [], "written": 0}
            )
            recurring = bool(path) or not (isinstance(times, int) and times == 1)
            counts = _page_counts(pages, sizes or {})
            for n in counts:
                if recurring and n not in ring["pushes"]:
                    ring["pushes"].append(n)
            if kind == "reserve":
                ring["written"] = max(ring["written"], max(counts) * cb.page_bytes)
    return [out[index] for index in sorted(out)]


def _compute_options(options, numerics):
    """The compute entries of the description for kernel `options` under `numerics`."""
    bounds = Requirements(
        options["min_fidelity"], options["max_fidelity"], options["exact"]
    )
    active = resolve(numerics, bounds)
    approx = active.approx if options["approx"] is None else options["approx"]
    return {
        "fp32_dest": options["fp32_dest"],
        "unpack_fp32": [cb.index for cb in options["unpack_fp32"]],
        "fidelity": options["fidelity"] or active.fidelity,
        "approx": approx,
    }


def _per_core(layout, index, count, xy, noc, tables):
    """The per-core argument items of core `index` (at `xy`) of `count` for
    `layout`, on NoC `noc`."""
    out = [index] if layout.core_index else []
    for arg in layout.per_core:
        if isinstance(arg, CoreArg):
            out.append(arg.item(xy, count, noc))
        elif isinstance(arg, TableArg):
            base = tables[id(arg.table)] + index * arg.size
            out += [{"value": base + k} for k in range(arg.size)]
        else:
            out.append(arg.at(index, count))
    return out


def table_bases(program, count):
    """{id(table): index of its first word in the launch values} on `count` cores:
    after the program's values, each table's block per core in turn."""
    out, at = {}, len(program.values)
    for table in program.tables:
        out[id(table)] = at
        at += count * table.size_words
    return out


def tile_pages(sizes):
    """Tile pages of a tensor of `sizes` (the last two dims padded to tiles)."""
    sizes = [1, 1, *sizes][-max(2, len(sizes)) :]
    lead = 1
    for n in sizes[:-2]:
        lead *= int(n)
    return lead * -(-int(sizes[-2]) // 32) * -(-int(sizes[-1]) // 32)


def table_words(program, count, tables, page_counts=None):
    """The launch values of the program's tables: `tables` {name: [rows of core 0,
    rows of core 1, ...]} (a core left out has no rows); `page_counts` {tensor name:
    pages} bounds the page columns."""
    if not isinstance(tables, dict):
        raise TypeError("kohakutt.dsl.metal: tables is {name: rows per core}")
    unknown = set(tables) - {t.name for t in program.tables}
    if unknown:
        raise ValueError(f"kohakutt.dsl.metal: {program.name} has no table {unknown}")
    out = []
    for table in program.tables:
        given = tables.get(table.name, [])
        if not isinstance(given, (list, tuple)):
            raise TypeError(
                f"kohakutt.dsl.metal: table {table.name} takes a list of rows per core"
            )
        given = list(given)
        if len(given) > count:
            raise ValueError(
                f"kohakutt.dsl.metal: table {table.name} has rows for {len(given)} "
                f"cores, the launch runs {count}"
            )
        given += [[]] * (count - len(given))
        for rows in given:
            out += table.words(rows, page_counts)
    return out


def _page_counts_of(inputs, outputs):
    """{tensor name: tile pages} of one launch's inputs and outputs."""
    out = {name: tile_pages(t.shape) for name, t in inputs.items()}
    out.update({name: tile_pages(sizes) for name, (sizes, _) in outputs.items()})
    return out


def _common(program, name):
    """The description item of value common argument `name`: a launch value, or a
    core reference's NoC coordinate."""
    for ref in program.core_refs:
        for coord in (ref.x, ref.y):
            if coord.name == name:
                return {"noc": [*coord.core, coord.axis]}
    return {"value": program.values.index(name)}


def spec(program, sources, count, numerics, sizes=None):
    """The JSON description of `program` on its first `count` cores with kernel
    sources `sources` ({kernel name: path}) and sized values `sizes`; it holds no
    data size beyond those. A variant with no core at `count` cores is left out."""
    cores = program.machine.core_list(count)
    every = ranges(cores)
    sizes = sizes or {}
    tensor = {id(t): i for i, t in enumerate(program.tensors)}
    out = {"interleaved": True, "cbs": [], "semaphores": [], "kernels": []}
    for cb in program.cbs:
        tiles = int(as_expr(cb.pages).eval(sizes))
        entry = {"index": cb.index, "tiles": tiles, "dtype": cb.dtype}
        if cb.page is not None:
            entry["page_bytes"] = cb.page
        out["cbs"].append({**entry, "cores": every})
    for s in program.scratches:
        size = int(as_expr(s.bytes).eval(sizes))
        out["cbs"].append({"index": s.index, "bytes": size, "cores": every})
    for sem in program.semaphores:
        entry = {"id": int(sem.id), "initial": int(sem.initial)}
        out["semaphores"].append({**entry, "cores": ranges(sem.cores.select(cores))})
    order, bases = {role: i for i, role in enumerate(KINDS)}, table_bases(
        program, count
    )
    for name, kernel in sorted(
        program.kernels.items(), key=lambda item: order[item[1].role]
    ):
        role, kind = kernel.role, KINDS[kernel.role]
        layout = kernel.layout()
        own = set(kernel.cores.select(cores))
        if not own and name != role:
            continue
        if not own:
            raise ValueError(f"kohakutt.dsl.metal: {role} has no core at {count} cores")
        noc = DEFAULT_NOC.get(role) if kernel.noc is None else int(kernel.noc)
        common = [{"addr": tensor[id(t)]} for t in layout.tensors]
        common += [_common(program, name) for name in layout.values]
        common += [count] if layout.core_count else []
        entry = {
            "source": str(sources[name]),
            "type": kind,
            "cores": ranges(own),
            "compile_args": [{"accessor": tensor[id(t)]} for t in layout.tensors],
            "common_args": common,
            "runtime_args": [
                [list(core), _per_core(layout, i, count, core, noc, bases)]
                for i, core in enumerate(cores)
                if core in own and layout.per_core_count
            ],
            "defines": {name: str(v) for name, v in kernel.defines.items()},
            "args": {"per_core": layout.per_core_count, "common": layout.common},
        }
        if kernel.noc is not None:
            entry["noc"] = int(kernel.noc)
        if role == "compute":
            entry.update(_compute_options(kernel.options, numerics))
        out["kernels"].append(entry)
    out["rings"] = rings(program, sizes)
    return json.dumps(out)


def l1_free():
    """Bytes of L1 per core a program launched now may give its buffers, or None
    when the native side cannot tell (no backend, or one built before this check)."""
    if not hasattr(torch.ops.ktt, "cb_budget"):
        return None
    return int(torch.ops.ktt.cb_budget())


def l1_check(program, sizes, free):
    """Raises ValueError when the buffers and scratch of `program` with sized values
    `sizes` take more than `free` bytes of L1 per core (None: nothing to check)."""
    need = program.footprint(sizes)
    if free is not None and need > free:
        raise ValueError(
            f"kohakutt.dsl.metal: {program.name}: this launch's circular buffers and "
            f"scratch take {need} bytes of L1 per core, {free} are free"
        )


def _format(dtype):
    if dtype not in DTYPES:
        raise TypeError(f"kohakutt.dsl.metal: unsupported tensor dtype {dtype}")
    return DTYPES[dtype]


class Launcher:
    """Runs a built Program on tt tensors with no torch op registered: `run(inputs,
    outputs, runtime)`. Sources are emitted once; a description is built once per
    core count, sized values and numerics scope. `check=False` skips the verifier;
    `l1_budget` (such as km.L1_BUDGET) bounds the buffers at every size."""

    def __init__(self, program, check=True, l1_budget=None):
        if check:
            verify(program, l1_budget=l1_budget)
        self.program, self.sources, self._specs = program, emit(program), {}

    def prepare(self, inputs, outputs, runtime=None, numerics=None, tables=None):
        """(input tensors, the other arguments of torch.ops.ktt.run_program) of one
        launch: `inputs` {name: tt tensor}, `outputs` {name: (sizes, torch dtype)},
        `runtime` {name: int}."""
        program, runtime = self.program, dict(runtime or {})
        numerics = current_numerics() if numerics is None else numerics
        for ref in program.tensors:
            given = outputs[ref.name][1] if ref.is_output else inputs[ref.name].dtype
            if _format(given)[0] != ref.dtype:
                raise TypeError(
                    f"kohakutt.dsl.metal: {program.name}: tensor {ref.name} is "
                    f"{given}, the program was built for {ref.dtype}"
                )
        count, sizes = program.core_count(runtime), program.sizes(runtime)
        key = (count, numerics, *sizes.values())
        text = self._specs.get(key)
        if text is None:
            l1_check(program, sizes, l1_free())
            text = spec(program, self.sources, count, numerics, sizes)
            self._specs[key] = text
        shapes = [outputs[t.name] for t in program.outputs]
        rest = (
            text,
            [len(sizes) for sizes, _ in shapes],
            [s for sizes, _ in shapes for s in sizes],
            [_format(dtype)[1] for _, dtype in shapes],
            program.value_list(runtime, count)
            + table_words(
                program, count, tables or {}, _page_counts_of(inputs, outputs)
            ),
        )
        return [inputs[t.name] for t in program.inputs], rest

    def run(self, inputs, outputs, runtime=None, tables=None):
        """Launches the program; returns its output tensors in declared order.
        `tables` {name: rows per core} fills the program's tables."""
        tensors, rest = self.prepare(inputs, outputs, runtime, tables=tables)
        return tuple(torch.ops.ktt.run_program(tensors, *rest))

    __call__ = run


def op(schema, template):
    """Registers the decorated builder as torch op `schema` ("ns::name(...) -> ...")
    for tt tensors. The builder takes the op's arguments and returns the Call of
    `template`: it runs once per argument sizes, dtypes and non-tensor values, and
    sources and descriptions are built once per kernel class."""
    qualified, _, signature = schema.partition("(")
    ns, _, name = qualified.strip().partition("::")
    single = schema.rsplit("->", 1)[1].strip() == "Tensor"

    def register(builder):
        if ns not in _LIBRARIES:
            _LIBRARIES[ns] = torch.library.Library(ns, "FRAGMENT")
        lib = _LIBRARIES[ns]
        classes = {}

        def launcher(call):
            """The Launcher of the call's kernel class (formats and statics)."""
            dtypes = {n: _format(t.dtype)[0] for n, t in call.inputs.items()}
            dtypes.update({n: _format(o[1])[0] for n, o in call.outputs.items()})
            key = (tuple(dtypes.items()), tuple(call.static.items()))
            entry = classes.get(key)
            if entry is None:
                cols, rows = torch.ops.ktt.compute_grid()
                program = template.trace(Machine(cols, rows), dtypes, **call.static)
                entry = classes[key] = Launcher(program, check=False)
            return entry

        def prepare(*args, **kwargs):
            call = builder(*args, **kwargs)
            return launcher(call).prepare(
                call.inputs, call.outputs, call.runtime, tables=call.tables
            )

        def fake(*args, **kwargs):
            call = builder(*args, **kwargs)
            device = next(iter(call.inputs.values())).device
            outs = [
                torch.empty(list(sizes), dtype=dtype, device=device)
                for sizes, dtype in call.outputs.values()
            ]
            return outs[0] if single else tuple(outs)

        _program_op(lib, ns, name, signature, prepare, single)
        lib._register_fake(name, fake)
        builder.op = getattr(getattr(torch.ops, ns), name)
        builder.template = template
        return builder

    return register
