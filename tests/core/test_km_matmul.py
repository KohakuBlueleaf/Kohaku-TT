"""The matmul example of the metal level on the CPU: recorded sources, the launch
plan, and what a sweep of shapes builds (examples/kernels/dsl_metal/matmul_km.py)."""

from pathlib import Path

import pytest
from km_programs import MACHINE, example, matmul

import kohakutt.dsl.metal as km
from km_rebuild import rebuild

GOLDEN = Path(__file__).resolve().parent / "km_golden"
# (a's shape, N, w stored [N, K]) of the card sweep in tests/hardware.
SWEEP = [
    ((4096, 256), 512, False),
    ((4096, 256), 512, True),
    ((2, 2048, 512), 768, False),
    ((8192, 768), 1024, False),
    ((8192, 1024), 1024, True),
    ((16384, 768), 1024, False),
    ((16384, 384), 96, False),
    ((6000, 64), 1000, False),
    ((3, 1000, 320), 40, True),
    ((16384, 1024), 256, False),
    ((131072, 64), 32, False),
]


def _plan(a_shape, n, w_nk, program=None):
    w_shape = (n, a_shape[-1]) if w_nk else (a_shape[-1], n)
    return example("matmul_km").plan(
        a_shape, w_shape, w_nk, MACHINE.cores, program=program
    )


@pytest.mark.parametrize("role", ["dm0", "compute", "dm1"])
def test_source_equals_recorded(role):
    text = km.source(matmul(), role)
    assert text == (GOLDEN / f"matmul_{role}.cpp.txt").read_text()


def test_the_program_verifies_within_its_budget_and_rebuilds():
    program = matmul()
    budget = example("matmul_km").BUDGET
    km.verify(program, l1_budget=budget)
    assert program.footprint() == (2 * 32 + 128 + 8) * 2048 <= budget
    built = rebuild(program)
    for role in program.kernels:
        assert km.source(built, role) == km.source(program, role)
    assert km.fingerprint(built) == km.fingerprint(program)


def test_the_kernels_hold_no_size():
    program = matmul()
    for role in program.kernels:
        text = km.source(program, role)
        assert "if (core >= groups * per_group) {" in text
        for size in ("768", "1024", "16384", "512"):
            assert size not in text, (role, size)
    compute = km.source(program, "compute")
    assert "matmul_block_init(cb_a, cb_w, w_nk, cols, 1, kc);" in compute
    assert (
        "matmul_block(cb_a, cb_w, k, (k0 + k) * cols, 0, w_nk, cols, 1, kc);" in compute
    )
    assert compute.count("get_common_arg_val") == 10


def test_plan():
    values = _plan((16384, 768), 1024, False)
    assert values == {
        "kt": 24,
        "ntot": 32,
        "mt": 512,
        "nt": 4,
        "kc": 24,
        "groups": 8,
        "per_group": 13,
        "w_nk": 0,
    }
    assert _plan((3, 1000, 320), 40, True)["nt"] == 2
    assert _plan((131072, 64), 32, False)["nt"] == 1
    assert _plan((4096, 250), 512, False) is None
    assert _plan((4096, 2048), 512, False) is None
    assert _plan((4096, 256), 2048, False) is None
    program = matmul()
    assert _plan((16384, 768), 1024, False, program)["nt"] == 4
    plan = example("matmul_km").plan
    tight = plan((16384, 1024), (1024, 1024), False, 110, 300 * 1024, program)
    assert tight["nt"] == 2 and program.sizes(tight) == {"a_pages": 32, "w_pages": 64}


def test_a_sweep_builds_per_class_not_per_shape():
    program = matmul()
    builds, exact = set(), set()
    for a_shape, n, w_nk in SWEEP:
        values = _plan(a_shape, n, w_nk, program)
        cores, sizes = program.core_count(values), program.sizes(values)
        assert values["groups"] * values["per_group"] <= cores
        assert sizes["a_pages"] >= values["kc"]
        assert sizes["w_pages"] >= values["kt"] * values["nt"]
        builds.add((cores, *sizes.values()))
        exact.add((cores, values["kc"], values["kt"] * values["nt"]))
        program.value_list(values, cores)
    assert (len(SWEEP), len(exact), len(builds)) == (11, 8, 8)
