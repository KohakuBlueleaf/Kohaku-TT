"""The broadcast binary program at the metal level, on the CPU: its launch values
per shape, its sources against the recorded ones, and a description without sizes."""

import json
from pathlib import Path

import pytest
from km_programs import MACHINE

import kohakutt.dsl.metal as km
from km_binary import BLOCKS, RING, binary_program, plan
from kohakutt.core.numerics import Numerics
from kohakutt.dsl.metal.bridge import spec

GOLDEN = Path(__file__).resolve().parent / "km_golden"
BF16 = {"a": "bfloat16", "b": "bfloat16", "out": "bfloat16"}
FP32 = {"a": "float32", "b": "float32", "out": "float32"}
# (golden name, tensor formats, static values).
CLASSES = {
    "binary_add_bf16": (BF16, {"op": "add"}),
    "binary_div_fp32": (FP32, {"op": "div"}),
    "binary_complex_conj_b": (FP32, {"op": "complex", "conj_b": True}),
}


def trace(name):
    dtypes, static = CLASSES[name]
    return binary_program.trace(MACHINE, dtypes, **static)


def test_plan_of_a_head_broadcast():
    values = plan((8, 256, 320), (256, 320))
    assert values == {
        "nr": 8,
        "sr": 80,
        "fill": 0,
        "block": 2,
        "groups": 80,
        "min_groups": 1,
    }
    assert plan((2, 12, 256, 320), (256, 320))["block"] == 3
    assert plan((7, 64, 64), (64, 64))["block"] == 1


def test_plan_of_a_row_broadcast():
    values = plan((640, 1024), (1, 1024))
    assert values == {
        "nr": 20,
        "sr": 32,
        "fill": 1,
        "block": 2,
        "groups": 32,
        "min_groups": 1,
    }
    assert plan((4, 320, 512), (1, 512))["nr"] == 40


def test_plan_of_equal_shapes_and_of_others():
    assert plan((3, 64, 64), (3, 64, 64))["nr"] == 1
    assert plan((3, 64, 64), (3, 64, 64))["min_groups"] == 4
    assert plan((3, 64, 64), (64, 1)) is None
    assert plan((3, 64, 64), (2, 64, 64)) is None


@pytest.mark.parametrize("name", CLASSES)
@pytest.mark.parametrize("role", ["dm0", "compute", "dm1"])
def test_source_equals_recorded(name, role):
    text = km.source(trace(name), role)
    assert text == (GOLDEN / f"{name}_{role}.cpp.txt").read_text()


def test_block_is_a_runtime_value_of_the_sources():
    program = trace("binary_add_bf16")
    assert program.choices == {"block": BLOCKS}
    for role in program.kernels:
        text = km.source(program, role)
        assert "block = get_common_arg_val" in text
        assert "cb_a = 0" in text or "cb_out = 16" in text
    assert all(RING % n == 0 for n in BLOCKS)


def test_launch_values_follow_the_plan():
    program = trace("binary_add_bf16")
    runtime = plan((8, 256, 320), (256, 320))
    assert program.core_count(runtime) == 96
    values = dict(zip(program.values, program.value_list(runtime, 96)))
    assert (values["groups_base"], values["groups_extra"]) == (0, 80)
    assert (values["nr"], values["sr"], values["block"]) == (8, 80, 2)
    small = plan((3, 64, 64), (3, 64, 64))
    assert program.core_count(small) == 3
    with pytest.raises(ValueError, match="block = 4 is not one of"):
        program.value_list({**runtime, "block": 4}, 64)


def test_description_holds_no_data_size():
    program = trace("binary_add_bf16")
    sources = {role: f"/k/{role}.cpp" for role in program.kernels}
    description = json.loads(spec(program, sources, 64, Numerics()))
    assert description["rings"] == [
        {"cb": 0, "pushes": [1, 2, 3], "written": 3 * 2048},
        {"cb": 1, "pushes": [1], "written": 2048},
        {"cb": 16, "pushes": [1, 2, 3], "written": 3 * 2048},
    ]
    compute = description["kernels"][1]
    assert compute["fidelity"] == "LoFi" and compute["approx"] is False
    assert compute["unpack_fp32"] == []
    fp32 = trace("binary_div_fp32")
    description = json.loads(spec(fp32, sources, 64, Numerics()))
    assert description["kernels"][1]["unpack_fp32"] == [0, 1]
    assert program.footprint() == (RING + 2 + RING) * 2048


def test_buffers_fit_the_budget_of_the_largest_block():
    """The 12-page buffers take what the backend's kernel takes at block 3."""
    in_tree = (4 * max(BLOCKS) + 2 + 4 * max(BLOCKS)) * 4096
    program = binary_program.trace(MACHINE, FP32, l1_budget=in_tree, op="mul")
    assert program.footprint() == in_tree
    with pytest.raises(km.VerifyError) as error:
        binary_program.trace(MACHINE, FP32, l1_budget=in_tree - 1, op="mul")
    assert error.value.rule == "L1-1"
