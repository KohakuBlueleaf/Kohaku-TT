"""The row softmax example (examples/kernels/dsl_metal/softmax_km.py) verifies and is
written with the matrix unit's reductions and broadcast whole-tile ops."""

import importlib.util
from pathlib import Path

from km_programs import MACHINE

import kohakutt.dsl.metal as km

EXAMPLE = Path(__file__).parents[2] / "examples/kernels/dsl_metal/softmax_km.py"


def _example():
    spec = importlib.util.spec_from_file_location("softmax_km", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_softmax_example_verifies_and_uses_the_matrix_unit():
    program = _example().softmax_program(MACHINE)
    km.verify(program)
    text = km.source(program, "compute")
    assert (
        "reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_x, cb_s, j, 0, 0);"
        in text
    )
    assert "sub_tiles_bcast_cols(cb_x, cb_max, j2, 0, 0);" in text
    assert "mul_tiles_bcast_cols(cb_exps, cb_sum, j4, 0, 0);" in text
    assert text.count("reduce_uninit();") == 2
