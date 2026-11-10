"""Sized values and layouts of the metal level on the CPU: a buffer sized from a
runtime value changes the description and never a source, and a layout picks the
core count and lets cores past it return."""

import json

import pytest
from km_programs import MACHINE
from test_km_verify_seeded import sized

import kohakutt.dsl.metal as km
from km_rebuild import rebuild


def _description(program, runtime):
    from kohakutt.dsl.metal.bridge import spec

    sources = dict.fromkeys(program.kernels, "source")
    count, sizes = program.core_count(runtime), program.sizes(runtime)
    return json.loads(spec(program, sources, count, None, sizes))


def _runtime(per=12, chunk=4, used=16):
    return {"per": per, "chunk": chunk, "used": used}


@pytest.fixture
def no_numerics(monkeypatch):
    import kohakutt.dsl.metal.bridge as bridge

    options = {"fp32_dest": True}
    monkeypatch.setattr(bridge, "_compute_options", lambda *_: options)


def test_a_sized_value_rounds_up_to_its_classes():
    program = sized()
    block = program.sized_values["block"]
    assert block.choices == (1, 2, 3, 4, 6, 8) and not block.exact
    assert [program.sizes(_runtime(chunk=c))["block"] for c in (1, 4, 5, 7, 8)] == [
        1,
        4,
        6,
        8,
        8,
    ]
    values = program.value_list(_runtime(chunk=5), 16)
    assert values[program.values.index("block")] == 6
    assert values[program.values.index("chunk")] == 5


def test_an_exact_sized_value_states_its_limit():
    p = km.Program("exact", MACHINE)
    n = p.runtime("n")
    pages = p.sized("pages", of=n * 2, exact=True, limit=12)
    assert pages.choices == tuple(range(1, 13)) and pages.exact
    assert p.sizes({"n": 5}) == {"pages": 10}
    with pytest.raises(ValueError, match="pages = 14 is outside 1 .. 12"):
        p.sizes({"n": 7})
    with pytest.raises(TypeError, match="states its limit"):
        p.sized("other", of=n, exact=True)
    with pytest.raises(ValueError, match="two values named pages"):
        p.sized("pages", of=n)


def test_sources_hold_no_size_and_a_new_class_changes_the_description_only(no_numerics):
    program = sized()
    for role in program.kernels:
        text = km.source(program, role)
        assert "const uint32_t block = get_common_arg_val<uint32_t>(" in text or (
            role == "dm1"
        )
    small, large = _description(program, _runtime(chunk=2)), _description(
        program, _runtime(chunk=7)
    )
    assert [cb["tiles"] for cb in small["cbs"]] == [4, 2]
    assert [cb["tiles"] for cb in large["cbs"]] == [16, 2]
    assert small["rings"][0] == {"cb": 0, "pushes": [2], "written": 2 * 2048}
    assert large["rings"][0] == {"cb": 0, "pushes": [8], "written": 8 * 2048}
    for entry in (small, large):
        entry.pop("cbs")
        entry.pop("rings")
    assert small == large
    same = _description(program, _runtime(chunk=7, per=4000))
    assert same["cbs"] == _description(program, _runtime(chunk=8))["cbs"]


def test_the_footprint_follows_the_sized_values():
    program = sized()
    assert program.footprint({"block": 3}) == (6 + 2) * 2048
    assert program.footprint() == (16 + 2) * 2048
    km.verify(program, l1_budget=18 * 2048)
    with pytest.raises(km.VerifyError, match="L1-1"):
        km.verify(program, l1_budget=18 * 2048 - 1)


def test_the_footprint_is_the_most_over_every_size():
    """A buffer that shrinks as its sized value grows is largest at the smallest."""
    p = km.Program("shrinking", MACHINE)
    block = p.sized("block", of=p.runtime("n"), limit=8)
    p.cb(0, dtype="bfloat16", pages=km.select(block < 4, 8, 2))
    p.cb(1, dtype="bfloat16", pages=block)
    assert p.footprint({"block": 8}) == (2 + 8) * 2048
    assert p.footprint() == (8 + 3) * 2048


def test_past_its_limit_the_footprint_bounds_each_buffer_alone():
    p = km.Program("many", MACHINE)
    a, b, c = (p.sized(n, of=p.runtime(n * 2), exact=True, limit=40) for n in "abc")
    p.cb(0, dtype="bfloat16", pages=km.select(a < 10, b, 1))
    p.cb(1, dtype="bfloat16", pages=km.select(a < 10, 1, c))
    assert p.footprint({"a": 5, "b": 40, "c": 40}) == (40 + 1) * 2048
    assert p.footprint() == (40 + 40) * 2048


def test_a_launcher_given_a_budget_refuses_a_program_that_never_fits_it():
    program = sized()
    program.scratch(km.L1_BUDGET - program.footprint() + 1)
    with pytest.raises(km.VerifyError, match="L1-1"):
        km.Launcher(program, l1_budget=km.L1_BUDGET)
    km.Launcher(program)


def test_a_launch_checks_its_buffers_against_the_free_l1():
    from kohakutt.dsl.metal.bridge import l1_check

    program = sized()
    l1_check(program, {"block": 3}, (6 + 2) * 2048)
    l1_check(program, {"block": 8}, None)
    with pytest.raises(ValueError, match="seed_sized: .* take 16384 bytes .* 16383"):
        l1_check(program, {"block": 3}, (6 + 2) * 2048 - 1)


def test_the_launch_check_counts_the_padding_between_buffers():
    """Each buffer starts at a multiple of 64 bytes, as the device places them."""
    from kohakutt.dsl.metal.bridge import l1_bytes, l1_check

    program = sized()
    program.scratch(10)
    program.scratch(100)
    sizes = {"block": 3}
    assert program.footprint(sizes) == (6 + 2) * 2048 + 110
    assert l1_bytes(program, sizes) == (6 + 2) * 2048 + 64 + 100
    l1_check(program, sizes, (6 + 2) * 2048 + 164)
    with pytest.raises(ValueError, match="take 16548 bytes of L1 per core, 16547"):
        l1_check(program, sizes, (6 + 2) * 2048 + 163)


def test_the_description_names_its_program(no_numerics):
    assert _description(sized(), _runtime())["name"] == "seed_sized"


def test_a_launch_too_large_for_the_free_l1_is_refused_before_queueing(
    no_numerics, monkeypatch
):
    import torch

    import kohakutt.dsl.metal.bridge as bridge

    launcher = bridge.Launcher(sized())
    monkeypatch.setattr(bridge, "l1_free", lambda: (6 + 2) * 2048)
    inputs = {"x": torch.empty(32, 32 * 64, dtype=torch.bfloat16)}
    outputs = {"out": ((32, 32 * 64), torch.bfloat16)}
    with pytest.raises(ValueError, match="take 20480 bytes of L1 per core, 16384"):
        launcher.prepare(inputs, outputs, _runtime(chunk=4), numerics="n")
    assert not launcher._specs
    _, rest = launcher.prepare(inputs, outputs, _runtime(chunk=3), numerics="n")
    assert json.loads(rest[0])["cbs"][0]["tiles"] == 6
    asked = []
    monkeypatch.setattr(bridge, "l1_free", lambda: asked.append(1) or 0)
    launcher.prepare(inputs, outputs, _runtime(chunk=3, per=24), numerics="n")
    assert not asked


def test_a_layout_picks_the_core_count_and_guards_the_kernels(no_numerics):
    program = sized()
    assert [program.core_count(_runtime(used=n)) for n in (1, 5, 16, 97, 110)] == [
        1,
        6,
        16,
        110,
        110,
    ]
    with pytest.raises(ValueError, match="a layout of 111 cores on a grid of 110"):
        program.core_count(_runtime(used=111))
    for role in program.kernels:
        text = km.source(program, role)
        assert "    if (core >= used) {\n        return;\n    }\n" in text
        assert "const uint32_t first = core * per;" in text or role == "compute"
    entry = _description(program, _runtime(used=5))
    assert entry["kernels"][0]["runtime_args"][-1] == [[5, 0], [5]]


def test_one_split_or_layout_per_program():
    p = km.Program("twice", MACHINE)
    p.layout(cores=p.runtime("used"))
    with pytest.raises(ValueError, match="one split or layout per program"):
        p.layout(cores=4)
    with pytest.raises(ValueError, match="one split"):
        p.split(p.runtime("tiles"))


def test_a_generator_rebuilds_sized_values_layouts_and_definitions(no_numerics):
    program = sized()
    built = rebuild(program)
    for role in program.kernels:
        assert km.source(built, role) == km.source(program, role)
    assert km.dump(built) == km.dump(program)
    runtime = _runtime(chunk=5)
    assert _description(built, runtime) == _description(program, runtime)
    text = km.dump(program)
    assert "  sized block = chunk: rounded up to one of [1, 2, 3, 4, 6, 8]\n" in text
    assert "  layout: used cores have work\n" in text


def test_comparisons_and_selection():
    p = km.Program("values", MACHINE)
    n, flag = p.runtime("n"), p.runtime("flag")
    picked = km.select(flag.ne(0), n * 2, n + 1)
    assert picked.cpp() == "flag != 0 ? n * 2 : n + 1"
    assert picked.eval({"n": 5, "flag": 0}) == 6
    assert (n >= 3).eval({"n": 3}) == 1 and (n > 3).eval({"n": 3}) == 0
    assert (64 // n).cpp() == "64 / n" and (n <= 2).cpp() == "n <= 2"
    assert km.min(n, 32).eval({"n": 40}) == 32
    with pytest.raises(TypeError, match="select takes a comparison"):
        km.select(n, 1, 2)
