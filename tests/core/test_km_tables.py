"""Tables (per-core job lists written at every launch) without a card: the program
verifies, its sources and description, the launch's checks of the words, and the
forms refused when built."""

import importlib.util
import json
from pathlib import Path

import pytest

import kohakutt.dsl.metal as km
from km_tables import MACHINE, MOST, ROWS, table_program
from kohakutt.dsl.metal import bridge

SEGMENTS = (
    Path(__file__).resolve().parents[2] / "examples/kernels/dsl_metal/segments_km.py"
)


def _segments():
    spec = importlib.util.spec_from_file_location("segments_km_example", SEGMENTS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_segments_example_plans_inside_its_tensors():
    seg = _segments()
    with pytest.raises(ValueError, match="at least one segment"):
        seg.plan([], 4, 8)
    with pytest.raises(ValueError, match="outside x"):
        seg.plan([(6, 3)], 4, 8)
    tables, tiles = seg.plan([(6, 2), (0, 3)], 4, 8)
    assert tiles == 5 and tables[:2] == [[[6, 0, 2]], [[0, 2, 3]]]
    program = seg.segments_program.trace(MACHINE, seg.DTYPES)
    counts = {"x": 8, "out": tiles}
    assert bridge.table_words(program, MACHINE.cores, {"jobs": tables}, counts)
    with pytest.raises(ValueError, match="of out, which has 4"):
        bridge.table_words(program, MACHINE.cores, {"jobs": tables}, {"x": 8, "out": 4})


def test_job_list_program_verifies_and_reads_its_block():
    program = table_program()
    km.verify(program)
    reader = km.source(program, "dm0")
    assert "const uint32_t jobs_at = 0;" in reader
    assert "j < get_arg_val<uint32_t>(jobs_at)" in reader
    assert "get_arg_val<uint32_t>(jobs_at + (j * 3) + 3)" in reader


def test_table_words_are_launch_values_of_each_core(monkeypatch):
    monkeypatch.setattr(bridge, "_compute_options", lambda *_: {})
    program = table_program()
    sources = {name: f"{name}.cpp" for name in program.kernels}
    spec = json.loads(bridge.spec(program, sources, MACHINE.cores, None))
    size = 1 + ROWS * 3
    reader = spec["kernels"][0]
    assert reader["args"]["per_core"] == size
    base = len(program.values)
    core, items = reader["runtime_args"][2]
    assert items[0] == {"value": base + 2 * size}
    words = bridge.table_words(
        program, MACHINE.cores, {"jobs": [[], [], [[5, 0, 3]]]}, {"x": 8, "out": 3}
    )
    assert len(words) == MACHINE.cores * (size + 1 + ROWS)
    assert words[2 * size : 2 * size + 4] == [1, 5, 0, 3]


def test_launch_refuses_words_past_the_table():
    program = table_program()
    with pytest.raises(ValueError, match="holds 4 rows"):
        bridge.table_words(program, 2, {"jobs": [[[0, 0, 1]] * (ROWS + 1)]})
    with pytest.raises(ValueError, match=f"column 2 takes 0 .. {MOST}"):
        bridge.table_words(program, 2, {"jobs": [[[0, 0, MOST + 1]]]})
    with pytest.raises(ValueError, match="takes 0 .."):
        bridge.table_words(program, 2, {"jobs": [[[-1, 0, 1]]]})
    with pytest.raises(ValueError, match="has rows for 3 cores"):
        bridge.table_words(program, 2, {"jobs": [[], [], []]})
    with pytest.raises(ValueError, match="no table"):
        bridge.table_words(program, 2, {"job": []})


def test_launch_refuses_pages_past_the_tensor():
    program = table_program()
    pages = {"x": 8, "out": 3}
    assert bridge.table_words(program, 1, {"jobs": [[[5, 0, 3]]]}, pages)
    with pytest.raises(ValueError, match="reads pages 6 .. 8 of x, which has 8"):
        bridge.table_words(program, 1, {"jobs": [[[6, 0, 3]]]}, pages)
    with pytest.raises(ValueError, match="of out, which has 3"):
        bridge.table_words(program, 1, {"jobs": [[[0, 1, 3]]]}, pages)
    with pytest.raises(ValueError, match="which has None"):
        bridge.table_words(program, 1, {"jobs": [[[0, 0, 1]]]})
    assert bridge.tile_pages([3, 40, 33]) == 3 * 2 * 2
    assert bridge.tile_pages([64]) == 2


def test_launch_refuses_malformed_tables_with_its_own_errors():
    program = table_program()
    for tables in (
        {"jobs": iter([[[0, 0, 1]]])},
        {"jobs": {0: [[0, 0, 1]]}},
        {"jobs": [[(0, 0, 1)], 5]},
        [("jobs", [])],
    ):
        with pytest.raises(TypeError, match=r"kohakutt\.dsl\.metal"):
            bridge.table_words(program, 2, tables, {"x": 8, "out": 8})


def test_tables_refused_when_built():
    p = km.Program("t", MACHINE)
    with pytest.raises(TypeError, match="static int"):
        p.table("a", width=3, rows=p.sized("r", of=p.runtime("n"), limit=8))
    p.table("jobs", width=3, rows=4)
    with pytest.raises(ValueError, match="two tables named jobs"):
        p.table("jobs", width=1, rows=4)
    t = p.table("other", width=2, rows=4)
    with pytest.raises(ValueError, match="columns 0 .. 1"):
        t[0, 2]


def test_table_most_and_pages_checked_when_built():
    p = km.Program("t", MACHINE)
    x = p.input("x", "float32")
    for most in ({5: 3}, {-1: 3}, {2: -1}, {2: "x"}, {2: 2**32}):
        with pytest.raises(ValueError, match="table a"):
            p.table("a", width=3, rows=4, most=most)
    for pages in ({0: (x, 1)}, {0: (x, 0)}, {3: (x, 2)}):
        with pytest.raises(ValueError, match="table b"):
            p.table("b", width=3, rows=4, most={2: 8}, pages=pages)
    other = km.Program("u", MACHINE).input("y", "float32")
    for pages in ({0: (other, 2)}, {0: x}):
        with pytest.raises(TypeError, match="pages"):
            p.table("c", width=3, rows=4, most={2: 8}, pages=pages)
