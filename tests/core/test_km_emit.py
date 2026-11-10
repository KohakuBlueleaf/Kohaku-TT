"""The metal-level C++ emitter on the CPU: the sources of the example programs equal
the recorded ones (tests/core/km_golden), and files are named by content."""

from pathlib import Path

import pytest
from km_programs import MACHINE, rope, swiglu

import kohakutt.dsl.metal as km
from kohakutt.dsl.metal.body import Startup
from kohakutt.dsl.metal.emit import COMPUTE_CALLS

GOLDEN = Path(__file__).resolve().parent / "km_golden"
PROGRAMS = {"rope": rope, "swiglu": swiglu}


@pytest.mark.parametrize("name", PROGRAMS)
@pytest.mark.parametrize("role", ["dm0", "compute", "dm1"])
def test_source_equals_recorded(name, role):
    text = km.source(PROGRAMS[name](), role)
    assert text == (GOLDEN / f"{name}_{role}.cpp.txt").read_text()


def test_sources_hold_no_data_size():
    program = rope()
    for role in program.kernels:
        text = km.source(program, role)
        assert "get_common_arg_val" in text
        assert "tiles_base" in text and "5184" not in text


def test_files_are_named_by_content_and_written_once(tmp_path):
    program = rope()
    paths = km.emit(program, tmp_path)
    assert sorted(paths) == ["compute", "dm0", "dm1"]
    for role, path in paths.items():
        assert path.name.startswith(f"rope_{role}_") and path.suffix == ".cpp"
        assert path.read_text() == km.source(program, role)
    stamps = {role: path.stat().st_mtime_ns for role, path in paths.items()}
    again = km.emit(rope(), tmp_path)
    assert again == paths
    assert {role: path.stat().st_mtime_ns for role, path in again.items()} == stamps
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        p.name for p in paths.values()
    )


def test_cache_dir_is_outside_the_package(monkeypatch, tmp_path):
    monkeypatch.setenv("KOHAKUTT_CACHE", str(tmp_path / "set"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert km.cache_dir() == tmp_path / "set" / "km"
    monkeypatch.delenv("KOHAKUTT_CACHE")
    assert km.cache_dir() == tmp_path / "xdg" / "kohakutt" / "km"
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert km.cache_dir() == tmp_path / "home" / ".cache" / "kohakutt" / "km"


def test_static_value_changes_the_source_name(tmp_path):
    from km_programs import MACHINE, ROPE_DTYPES, example

    template = example("rope_km").rope_program
    four = km.emit(template.trace(MACHINE, ROPE_DTYPES, block=4), tmp_path)
    two = km.emit(template.trace(MACHINE, ROPE_DTYPES, block=2), tmp_path)
    assert four["dm0"] != two["dm0"] and four["dm1"] == two["dm1"]


def test_clashing_names_are_refused():
    @km.program("clash")
    def clash(p, x: km.TensorRef, out: km.Out):
        start, count = p.split(p.runtime("tiles"))
        cb_x, cb_out = p.cb(0, like=x), p.cb(16, like=out)

        @p.dm0
        def reader(k):
            for i in k.range(count, name="x"):
                slot = cb_x.reserve(1)
                k.read(x.page(start + i), slot)
                k.reads_done()
                cb_x.push(1)

        @p.compute
        def compute(c):
            c.startup(cb_x, cb_x, cb_out)
            c.init("copy", cb_x)
            for _ in c.range(count):
                cb_x.wait(1)
                with c.registers():
                    c.copy(cb_x, 0, 0)
                    cb_out.reserve(1)
                    c.pack(0, cb_out)
                    cb_out.push(1)
                cb_x.pop(1)

        @p.dm1
        def writer(k):
            for i in k.range(count):
                cb_out.wait(1)
                k.write(cb_out, out.page(start + i))
                k.flushed()
                cb_out.pop(1)
            k.writes_done()

    program = clash.trace(km.Machine(), {"x": "bfloat16", "out": "bfloat16"})
    with pytest.raises(ValueError, match="declares x twice"):
        km.source(program, "dm0")


def test_a_reversed_startup_of_one_input_names_it_twice():
    p = km.Program("startup", MACHINE)
    cb_x = p.cb(0, dtype="bfloat16", name="x")
    cb_out = p.cb(16, dtype="bfloat16", name="out")
    text = COMPUTE_CALLS[Startup](Startup(cb_x, None, cb_out, reverse=True))
    assert text == "compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_x, cb_out)"
    text = COMPUTE_CALLS[Startup](Startup(cb_x, None, cb_out))
    assert text == "compute_kernel_hw_startup(cb_x, cb_out)"


def test_names_a_star_import_brings():
    assert "min" not in km.__all__ and km.min(3, 5).value() == 3
    p = km.Program("names", MACHINE)
    k = p.kernel("dm0")
    k.macro("WIDE", 1)
    assert p.kernels["dm0"].defines == {"WIDE": 1}
    assert not hasattr(k, "define")
