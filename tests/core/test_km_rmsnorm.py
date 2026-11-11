"""The RMS norm example of the metal level on the CPU
(examples/kernels/dsl_metal/rmsnorm_km.py): every variant verifies, rows travel as
control words, and the launch values follow the backend's norm."""

import struct

import pytest
from km_programs import MACHINE, example

import kohakutt.dsl.metal as km


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("gamma", [None, "bfloat16", "float32"])
def test_every_variant_verifies(dtype, gamma):
    program = example("rmsnorm_km").rmsnorm_program(MACHINE, dtype, gamma)
    km.verify(program)
    assert program.footprint() <= 1280 * 1024


def test_rows_are_control_words_and_the_size_is_no_compile_value():
    program = example("rmsnorm_km").rmsnorm_program(MACHINE, "bfloat16", "bfloat16")
    reader = km.source(program, "dm0")
    compute = km.source(program, "compute")
    writer = km.source(program, "dm1")
    for text in (reader, compute, writer):
        assert "for (uint32_t it = 0;; ++it) {" in text
        assert "768" not in text and "4096" not in text.replace(", 4096)", "")
    assert "r = fetched + groups;" in reader and "r = r + groups;" in reader
    assert "const uint32_t row = read_tile_value(cb_to_compute, 0, 0);" in compute
    assert "two_pass_stats_update_rows<true>(0, 0, j == (seg - 1) ? last : 32);" in (
        compute
    )
    # y: two tiles per call with S and P loaded once per band, pairs rotated by row.
    call = "MATH((_llk_math_eltwise_binary_sfpu_params_("
    assert f"{call}norm_y_t2, 0, 2, 0, VectorMode::None, 2, 3, 3)));" in compute
    assert f"{call}km_join_cols, 1, 2, 1, VectorMode::None)));" in compute
    assert "(row + i) % pairs" in compute and "(row + i) % pairs" in writer
    assert "noc_async_write_page((row * wt) + (2 * k), y, get_read_ptr(cb_y));" in (
        writer
    )


def _float(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def test_launch_values():
    runtime = example("rmsnorm_km").runtime
    values = runtime((2, 1000, 320), 1e-6, 110)
    assert values["rows"] == 64 and values["wt"] == 10 and values["last"] == 32
    assert values["groups"] == 64 and values["dynamic"] == 0
    assert _float(values["inv_w"]) == pytest.approx(1 / 320, rel=1e-7)
    assert _float(values["eps"]) == pytest.approx(1e-6, rel=1e-7)
    ragged = runtime((4096, 1000), 1e-5, 110)
    assert ragged["rows"] == 128 and ragged["groups"] == 110 and ragged["dynamic"]
    assert ragged["wt"] == 32 and ragged["last"] == 8
