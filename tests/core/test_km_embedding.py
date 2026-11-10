"""The row-gather program at the metal level, on the CPU: one kernel that reads and
writes, a strided split, values loaded from L1, and its sources against the recorded
ones."""

import json
from pathlib import Path

import pytest
from km_programs import MACHINE

import kohakutt.dsl.metal as km
from km_embedding import embedding_program, plan
from kohakutt.core.numerics import Numerics
from kohakutt.dsl.metal.bridge import spec

GOLDEN = Path(__file__).resolve().parent / "km_golden"
CLASSES = {"embedding_fp32": "float32", "embedding_bf16": "bfloat16"}


def trace(name):
    dtype = CLASSES[name]
    return embedding_program.trace(
        MACHINE, {"table": dtype, "index": "int32", "out": dtype}
    )


def test_plan():
    assert plan((1000, 256), 33) == {
        "n": 33,
        "dt": 8,
        "vt": 32,
        "blocks": 2,
        "units": 16,
    }
    assert plan((3, 700, 64), 500)["units"] == 3 * 16 * 2


@pytest.mark.parametrize("name", CLASSES)
def test_source_equals_recorded(name):
    text = km.source(trace(name), "dm0")
    assert text == (GOLDEN / f"{name}_dm0.cpp.txt").read_text()


def test_one_kernel_reads_and_writes():
    program = trace("embedding_fp32")
    assert list(program.kernels) == ["dm0"]
    text = km.source(program, "dm0")
    assert "noc_async_read(" in text and "noc_async_write_page(" in text
    assert "cb_push_back" not in text and "cb_stage" not in text
    assert "cb_stage = 2" in km.source(trace("embedding_bf16"), "dm0")


def test_units_are_dealt_out_in_turn():
    program = trace("embedding_fp32")
    text = km.source(program, "dm0")
    assert "for (uint32_t u = core; u < units; u += cores) {" in text
    runtime = plan((1000, 256), 33)
    assert program.core_count(runtime) == 16
    assert program.core_count(plan((1000, 256), 500)) == 110
    # 10 units run on 12 cores: two start past the last unit.
    assert program.core_count(plan((1000, 160), 33)) == 12
    assert program.value_list(runtime, 16) == [33, 8, 32, 2, 16]


def test_description_holds_no_data_size_and_the_core_count():
    program = trace("embedding_bf16")
    description = json.loads(spec(program, {"dm0": "/k/dm0.cpp"}, 12, Numerics()))
    (kernel,) = description["kernels"]
    assert kernel["type"] == "reader"
    values = [{"value": program.values.index(v)} for v in program.values]
    common = kernel["common_args"]
    assert common[:3] == [{"addr": 1}, {"addr": 0}, {"addr": 2}]
    assert sorted(common[3:-1], key=str) == sorted(values, key=str)
    assert common[-1] == 12
    assert kernel["args"] == {"per_core": 1, "common": 9}
    assert description["rings"] == [
        {"cb": 0, "pushes": [], "written": 4096},
        {"cb": 1, "pushes": [], "written": 2048},
        {"cb": 2, "pushes": [], "written": 4096},
    ]
    assert program.footprint() == 4096 + 2048 + 4096
