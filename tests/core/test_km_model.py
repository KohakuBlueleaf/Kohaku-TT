"""The metal-level program model on the CPU: size classes, binding times, argument
layouts and the launch description, which must hold no data size."""

import json

import pytest
from km_programs import MACHINE, rope, swiglu

import kohakutt.dsl.metal as km
from kohakutt.core.numerics import Numerics
from kohakutt.dsl.metal import binding
from kohakutt.dsl.metal.bridge import spec
from kohakutt.dsl.metal.machine import ranges

# Candidate block sizes: 3 does not divide the 8-page buffers, 4 does.
BLOCKS = km.auto(3, 4)


def test_size_classes_are_bounded():
    classes = MACHINE.classes()
    assert classes == [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 110]
    assert all(2 * b <= 3 * a for a, b in zip(classes[1:], classes[2:]))
    assert [MACHINE.pick(n) for n in (0, 1, 3, 5, 63, 64, 109, 110, 10**6)] == [
        1,
        1,
        3,
        4,
        48,
        64,
        96,
        110,
        110,
    ]


def test_core_ranges_are_rectangles():
    assert ranges(MACHINE.core_list(110)) == [[0, 0, 10, 9]]
    assert ranges(MACHINE.core_list(16)) == [[0, 0, 10, 0], [0, 1, 4, 1]]
    assert ranges(MACHINE.core_list(1)) == [[0, 0, 0, 0]]


def test_core_sets():
    cores = km.CoreSet()
    row = cores.rows(0)
    assert cores.covers(row) and not row.covers(cores)
    assert row.covers(cores.rows(0)) and not row.overlaps(cores.rows(1, 2))
    assert row.select(MACHINE.core_list(16)) == MACHINE.core_list(11)


def test_binding_times():
    tiles = binding.Value("tiles")
    assert tiles.binding == binding.RUNTIME
    assert binding.CoreIndex().binding == binding.PER_CORE
    assert (tiles + 1).binding == binding.DERIVED
    assert (binding.Const(3) * 4).value() == 12
    assert binding.minimum(tiles, 4).upper() == 4
    assert (tiles % 8).upper() == 7


def test_runtime_value_cannot_steer_python():
    with pytest.raises(km.VerifyError) as error:
        bool(binding.Value("tiles"))
    assert error.value.rule == "BIND-1"


def test_rope_model():
    program = rope()
    assert [t.name for t in program.tensors] == ["x", "table", "out"]
    assert program.values == ["period", "tiles", "tiles_base", "tiles_extra"]
    assert {cb.name: (cb.index, cb.pages) for cb in program.cbs} == {
        "x": (0, 8),
        "table": (1, 8),
        "out": (16, 2),
    }
    assert program.footprint() == 8 * 2048 + 8 * 4096 + 2 * 2048
    reader = program.kernels["dm0"].layout()
    assert [t.name for t in reader.tensors] == ["x", "table"]
    assert reader.values == ["tiles_base", "tiles_extra", "period"]
    assert (reader.common, reader.per_core_count) == (5, 1)
    compute = program.kernels["compute"].layout()
    assert (compute.common, compute.per_core_count) == (2, 1)


def test_split_values():
    program = rope()
    runtime = {"tiles": 5184, "period": 36}
    assert program.core_count(runtime) == 110
    assert program.value_list(runtime, 110) == [36, 5184, 47, 14]
    assert program.core_count({"tiles": 4, "period": 1}) == 4
    assert program.value_list({"tiles": 4, "period": 1}, 4) == [1, 4, 1, 0]
    # 5 tiles on the next class, 6 cores: the last one takes no work.
    assert program.core_count({"tiles": 5, "period": 1}) == 6
    assert program.value_list({"tiles": 5, "period": 1}, 6) == [1, 5, 0, 5]


def test_description_holds_no_data_size():
    program = rope()
    sources = {role: f"/k/{role}.cpp" for role in program.kernels}
    text = spec(program, sources, 110, Numerics())
    assert text == spec(program, sources, 110, Numerics())
    description = json.loads(text)
    assert [k["type"] for k in description["kernels"]] == [
        "reader",
        "compute",
        "writer",
    ]
    reader = description["kernels"][0]
    assert reader["compile_args"] == [{"accessor": 0}, {"accessor": 1}]
    assert reader["common_args"] == [
        {"addr": 0},
        {"addr": 1},
        {"value": 2},
        {"value": 3},
        {"value": 0},
    ]
    assert reader["args"] == {"per_core": 1, "common": 5}
    assert reader["cores"] == [[0, 0, 10, 9]]
    assert reader["runtime_args"][:2] == [[[0, 0], [0]], [[1, 0], [1]]]
    assert len(reader["runtime_args"]) == 110
    compute = description["kernels"][1]
    assert compute["unpack_fp32"] == [1] and compute["fidelity"] == "HiFi2"
    assert description["rings"] == [
        {"cb": 0, "pushes": [4], "written": 4 * 2048},
        {"cb": 1, "pushes": [4], "written": 4 * 4096},
        {"cb": 16, "pushes": [1], "written": 2048},
    ]


def test_description_differs_only_in_cores_between_classes():
    program = swiglu()
    sources = {role: f"/k/{role}.cpp" for role in program.kernels}
    small = json.loads(spec(program, sources, 4, Numerics()))
    large = json.loads(spec(program, sources, 64, Numerics()))
    assert small["cbs"][0]["cores"] == [[0, 0, 3, 0]]
    for description in (small, large):
        for entry in (*description["cbs"], *description["kernels"]):
            entry.pop("cores")
            entry.pop("runtime_args", None)
    assert small == large


def test_auto_takes_the_first_candidate_that_verifies():
    @km.program("blocked")
    def blocked(p, x: km.TensorRef, out: km.Out, block: km.Static[int] = BLOCKS):
        start, count = p.split(p.runtime("tiles"))
        cb_x, cb_out = p.cb(0, like=x, pages=8), p.cb(16, like=out, pages=8)

        @p.dm0
        def reader(k):
            for done, n in k.blocks(count, block):
                slot = cb_x.reserve(block)
                for i in k.range(n):
                    k.read(x.page(start + done + i), slot[i])
                k.reads_done()
                cb_x.push(block)

        @p.compute
        def compute(c):
            c.startup(cb_x, cb_x, cb_out)
            c.init("copy", cb_x)
            for _, n in c.blocks(count, block):
                cb_x.wait(block)
                for i in c.range(n):
                    with c.registers():
                        c.copy(cb_x, i, 0)
                        cb_out.reserve(1)
                        c.pack(0, cb_out)
                        cb_out.push(1)
                cb_x.pop(block)

        @p.dm1
        def writer(k):
            for i in k.range(count):
                cb_out.wait(1)
                k.write(cb_out, out.page(start + i))
                k.flushed()
                cb_out.pop(1)
            k.writes_done()

    program = blocked.trace(MACHINE, {"x": "bfloat16", "out": "bfloat16"})
    assert program.statics == {"block": 4}
