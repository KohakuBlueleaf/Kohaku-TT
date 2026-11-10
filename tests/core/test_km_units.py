"""The compute constructs of the metal level on the CPU: the test program of
tests/km_units.py emits the recorded sources, a generator rebuilds it, and each
construct emits the TT-Metal call it names."""

from pathlib import Path

import pytest
from km_programs import MACHINE

import kohakutt.dsl.metal as km
from km_rebuild import rebuild
from km_units import units_program
from kohakutt.dsl.metal.ops import OPS

GOLDEN = Path(__file__).resolve().parent / "km_golden"


@pytest.mark.parametrize("role", ["dm0", "compute", "dm1"])
def test_source_equals_recorded(role):
    text = km.source(units_program(MACHINE), role)
    assert text == (GOLDEN / f"units_{role}.cpp.txt").read_text()


def test_a_generator_rebuilds_it():
    program = units_program(MACHINE)
    built = rebuild(program)
    for role in program.kernels:
        assert km.source(built, role) == km.source(program, role)
    assert km.fingerprint(built) == km.fingerprint(program)


def test_a_branch_reads_as_if_else():
    text = km.source(units_program(MACHINE), "compute")
    assert "        if (flag != 0) {\n" in text
    assert "        } else {\n            binop_with_scalar_tile_init();\n" in text
    assert "sub_unary_tile(0, 0x3f800000u);" in text
    assert "add_unary_tile(0, bias);" in text


def _compute(body, **options):
    """The compute source of a one-kernel program whose body is `body(c, a, b)` over
    two bf16 buffers."""
    p = km.Program("lines", MACHINE)
    x = p.input("x", "bfloat16")
    a, b = p.cb(0, like=x, pages=8, name="a"), p.cb(1, like=x, pages=8, name="b")
    c = p.kernel("compute", **options)
    body(c, a, b)
    return km.source(p, "compute")


def test_unpack_variants():
    def body(c, a, b):
        c.startup(a, None, b)
        c.format(srcb=b)
        c.format(a, b)
        for name in ("copy", "transpose", "bcast_col"):
            c.init(name, a)
        c.copy_block(a, 2, 0, 3)
        c.transpose(a, 1, 2)
        c.bcast(a, 0, 3, "col")

    text = _compute(body)
    for line in (
        '#include "api/compute/compute_kernel_hw_startup.h"',
        '#include "api/compute/transpose.h"',
        "compute_kernel_hw_startup(cb_a, cb_b);",
        "reconfig_data_format_srcb(cb_b);",
        "reconfig_data_format(cb_a, cb_b);",
        "transpose_init(cb_a);",
        "unary_bcast_init<BroadcastType::COL>(cb_a);",
        "copy_block(cb_a, 2, 0, 3);",
        "transpose_tile(cb_a, 1, 2);",
        "unary_bcast<BroadcastType::COL>(cb_a, 0, 3);",
    ):
        assert f"    {line}\n" in text or f"{line}\n" in text, line


def test_matmul_takes_runtime_shapes():
    def body(c, a, b):
        nt = c.program.runtime("nt", choices=(1, 2, 4))
        kc, flip = c.program.runtime("kc"), c.program.runtime("flip")
        c.startup(a, b, b, reverse=True)
        product = c.matmul_init(a, b, transpose=flip, ct=nt, rt=1, kt=kc)
        with c.loop(4, name="k") as k:
            c.matmul(product, k, k * nt, 0)

    text = _compute(body)
    assert "compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_b, cb_b);" in text
    assert "matmul_block_init(cb_a, cb_b, flip, nt, 1, kc);" in text
    assert "matmul_block(cb_a, cb_b, k, k * nt, 0, flip, nt, 1, kc);" in text
    assert '#include "api/compute/matmul.h"' in text


def test_vector_unit_function_modes_and_arguments():
    fn = km.SfpuFn(
        "whole", "inline void whole(uint32_t, uint32_t, uint32_t, uint32_t) {}"
    )

    def body(c, a, b):
        cols = c.program.runtime("cols")
        c.sfpu(fn, 0, 1, 0, mode="none", args=[cols, 3])
        c.sfpu(fn, 0, 1, 2)

    text = _compute(body)
    call = "MATH((_llk_math_eltwise_binary_sfpu_params_(whole, 0, 1, "
    assert call + "0, VectorMode::None, cols, 3)));" in text
    assert call + "2, VectorMode::RC)));" in text


@pytest.mark.parametrize("name", sorted(OPS))
def test_every_table_op_emits_its_call_and_setup(name):
    entry = OPS[name]

    def body(c, a, b):
        c.init(name)
        c.op(name, *range(entry.slots), values=[0.5] * entry.values)

    text = _compute(body)
    slots = ", ".join(str(i) for i in range(entry.slots))
    scalar = ", 0x3f000000u" * entry.values
    assert f"    {entry.init}();\n    {entry.call}({slots}{scalar});\n" in text
    assert f'#include "{entry.header}"' in text


def test_the_builder_refuses_what_the_table_lacks():
    p = km.Program("refused", MACHINE)
    c = p.kernel("compute")
    with pytest.raises(ValueError, match="no tile op named 'cbrt'"):
        c.op("cbrt", 0)
    with pytest.raises(TypeError, match="max takes 3 register slots and 0 scalar"):
        c.op("max", 0, 1)
    with pytest.raises(ValueError, match="no setup call named 'cbrt'"):
        c.init("cbrt")
    with (
        pytest.raises(TypeError, match="otherwise follows a when block"),
        c.otherwise(),
    ):
        pass
