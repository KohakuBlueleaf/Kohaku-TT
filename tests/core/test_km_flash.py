"""The flash attention example (examples/kernels/dsl_metal/flash_km.py) verifies in
every lane length on the p150 grid, and its plan takes each query tile row once, in
the backend kernel's block sizes and lane lengths at the benchmark shapes."""

import pytest
from km_programs import MACHINE, example

import kohakutt.dsl.metal as km

DTYPES = {"q": "bfloat16", "k": "bfloat16", "v": "bfloat16", "out": "bfloat16"}


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("length", [1, 2, 5, 10])
def test_every_lane_length_verifies(causal, length):
    flash = example("flash_km")
    program = flash.flash_program.trace(MACHINE, DTYPES, causal=causal, length=length)
    readers = [k for k in program.kernels.values() if k.role == "dm0"]
    assert len(readers) == (1 if length == 1 else 2 * (MACHINE.rows // length))
    assert km.source(program, "compute").count("fk::exp_section(") == 1


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(1, 14, 64), (2, 3, 10), (1, 24, 138), (4, 16, 7)])
def test_the_plan_takes_each_row_once(causal, shape):
    flash = example("flash_km")
    b, h, sqt = shape
    for length in (1, 2, 5, 10):
        lanes, span = MACHINE.cols * (MACHINE.rows // length), length * 13
        chunks = (flash.causal_chunks if causal else flash.lane_chunks)(
            b * h, sqt, span, lanes
        )
        seen = [
            (head, row)
            for lane in chunks
            for chunk in lane
            if chunk is not None
            for head, r0, n in [chunk]
            for row in range(r0, r0 + n)
        ]
        assert sorted(seen) == [(i, r) for i in range(b * h) for r in range(sqt)]


def test_the_benchmark_plans_are_the_backend_kernels():
    flash = example("flash_km")
    budget = km.L1_BUDGET
    for shape, causal in [((1, 14, 2048, 64), False), ((1, 14, 2048, 64), True)]:
        values, length = flash.plan(shape, shape, shape, causal, None, budget, 11, 10)
        assert (values["r"], values["kb"], length) == (13, 12, 5)
    shape = (1, 24, 4396, 128)
    values, length = flash.plan(shape, shape, shape, False, None, budget, 11, 10)
    assert (values["r"], values["kb"], values["tail"], length) == (13, 12, 12, 5)
    assert values["jobs"] == 4
