"""Tensors by rows at the metal level (no card): page sizes, the launch description's
row positions, row page counts and the width check."""

import json

import pytest
from km_programs import MACHINE

import kohakutt.dsl.metal as km
from km_rowmajor import WIDTH, rows_program
from kohakutt.core.numerics import Numerics
from kohakutt.dsl.metal import bridge


def test_a_tensor_by_rows_has_row_pages():
    p = rows_program(MACHINE)
    x, out, y = p.tensors
    assert (x.rows, x.page_bytes) == (WIDTH, WIDTH * 2)
    assert out.rows is None and out.page_bytes == 2048
    assert "TensorAccessor(x_args, x_addr, 128)" in km.source(p, "dm0")


def test_the_description_names_the_tensors_by_rows():
    p = rows_program(MACHINE)
    sources = {name: f"{name}.cpp" for name in p.kernels}
    description = json.loads(bridge.spec(p, sources, 4, Numerics()))
    assert description["rows"] == [0, 2]
    page = {cb["index"]: cb.get("page_bytes") for cb in description["cbs"]}
    assert page[0] == page[17] == WIDTH * 2 and page[1] is None


def test_row_pages_and_the_width_are_checked():
    p = rows_program(MACHINE)
    x, out, _ = p.tensors
    assert bridge._pages(x, [96, WIDTH]) == 96
    assert bridge._pages(x, [2, 48, WIDTH]) == 96
    assert bridge._pages(out, [96, WIDTH]) == 3 * 2
    with pytest.raises(ValueError, match="rows of 64"):
        bridge._pages(x, [96, 32])


def test_a_tensor_by_rows_takes_a_static_width_and_an_element_format():
    p = km.Program("bad", MACHINE)
    with pytest.raises(ValueError, match="static width"):
        p.input("x", "bfloat8_b", rows=64)
    with pytest.raises(ValueError, match="static width"):
        p.input("y", "bfloat16", rows=p.runtime("w"))
