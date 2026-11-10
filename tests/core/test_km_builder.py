"""The explicit builder of the metal level on the CPU: programs constructed by code
(no decorator) emit the same sources as the decorated ones, and the model has a
text form, a hash and host-side values."""

import json

import pytest
from km_programs import MACHINE, rope, swiglu

import kohakutt.dsl.metal as km
from km_binary import binary_program
from km_chain import chain_program
from km_embedding import embedding_program
from km_rebuild import rebuild

BF16 = {"a": "bfloat16", "b": "bfloat16", "out": "bfloat16"}
FP32 = {"a": "float32", "b": "float32", "out": "float32"}
EMBEDDING = {"table": "bfloat16", "index": "int32", "out": "bfloat16"}
PROGRAMS = {
    "rope": rope,
    "swiglu": swiglu,
    "binary add": lambda: binary_program.trace(MACHINE, BF16),
    "binary div fp32": lambda: binary_program.trace(MACHINE, FP32, op="div"),
    "binary complex": lambda: binary_program.trace(
        MACHINE, FP32, op="complex", conj_b=True
    ),
    "embedding bf16": lambda: embedding_program.trace(MACHINE, EMBEDDING),
    "embedding fp32": lambda: embedding_program.trace(
        MACHINE, {**EMBEDDING, "table": "float32", "out": "float32"}
    ),
}


def _description(program, count):
    from kohakutt.dsl.metal.bridge import spec

    sources = dict.fromkeys(program.kernels, "source")
    return json.loads(spec(program, sources, count, None))


@pytest.mark.parametrize("name", PROGRAMS)
def test_a_program_built_by_code_equals_the_decorated_one(name, monkeypatch):
    import kohakutt.dsl.metal.bridge as bridge

    monkeypatch.setattr(
        bridge, "_compute_options", lambda options, numerics: {"fp32_dest": True}
    )
    decorated = PROGRAMS[name]()
    built = rebuild(decorated)
    assert built is not decorated and built.kernels.keys() == decorated.kernels.keys()
    for role in decorated.kernels:
        assert km.source(built, role) == km.source(decorated, role), role
    for count in (1, 8, MACHINE.cores):
        assert _description(built, count) == _description(decorated, count)
    assert km.dump(built) == km.dump(decorated)
    assert km.fingerprint(built) == km.fingerprint(decorated)


def test_fingerprint_names_the_kernel_class():
    add = binary_program.trace(MACHINE, BF16)
    assert km.fingerprint(add) == km.fingerprint(binary_program.trace(MACHINE, BF16))
    others = [
        binary_program.trace(MACHINE, BF16, op="sub"),
        binary_program.trace(MACHINE, FP32),
        binary_program.trace(km.Machine(8, 8), BF16),
    ]
    prints = {km.fingerprint(p) for p in [add, *others]}
    assert len(prints) == 4


def test_dump_shows_declarations_and_statements():
    text = km.dump(rope())
    assert text.startswith("program rope on 11 x 10 cores\n")
    for line in (
        "  input table: float32, dram",
        "  runtime period",
        "  split over the cores: tiles, at least 1 per core",
        "kernel dm0 on all",
        "        CBOp(kind=reserve, cb=x, pages=4, slot=x_ptr)",
        "kernel compute on all",
        "  helper rotate_face",
    ):
        assert line in text.splitlines(), line


def test_values_evaluate_on_the_host():
    p = km.Program("values", MACHINE)
    tiles = p.runtime("tiles")
    start, count = p.split(tiles)
    env = {"tiles_base": 7, "tiles_extra": 3, "core": 2, "cores": 16}
    assert (start.eval(env), count.eval(env)) == (16, 8)
    assert ((tiles + 31) // 32 * p.core_count_value).eval({"tiles": 33, **env}) == 32
    assert (p.core - 5).eval(env) == 2**32 - 3
    k = p.kernel("dm0")
    with k.loop(count) as i, pytest.raises(ValueError, match="no value on the host"):
        i.eval(env)


def test_per_core_values_from_the_core_index():
    p = km.Program("percore", MACHINE)
    x, out = p.input("x", "bfloat16"), p.output("out", "bfloat16")
    row, col = p.per_core("at", [p.core // 11, p.core % 11], count=2)
    cb = p.cb(0, like=x)
    k = p.kernel("dm0")
    slot = k.reserve(cb, 1)
    k.read(x.page(row * 11 + col), slot)
    k.reads_done()
    k.write(slot, out.page(row * 11 + col))
    k.writes_done()
    km.verify(p)
    args = _description(p, 24)["kernels"][0]["runtime_args"]
    assert [a[1] for a in args][:1] + [a[1] for a in args][-1:] == [[0, 0], [2, 1]]
    bad = km.Program("percore", MACHINE)
    bad.per_core("at", bad.runtime("n") + bad.core)
    with pytest.raises(km.VerifyError, match="BIND-2"):
        km.verify(bad)


def test_the_explicit_builder_checks_what_it_is_given():
    p = km.Program("checks", MACHINE)
    p.input("x", "bfloat16")
    with pytest.raises(ValueError, match="two tensors named x"):
        p.output("x", "bfloat16")
    p.kernel("dm0")
    with pytest.raises(ValueError, match="two dm0 kernels"):
        p.kernel("dm0")
    with pytest.raises(ValueError, match="unknown kernel role"):
        p.kernel("reader")
    with pytest.raises(TypeError, match="takes no options"):
        p.kernel("dm1", fp32_dest=True)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_a_generated_chain_verifies_and_is_one_pass(dtype):
    nodes = [("mul", 1), ("add", 2), ("sub", 0), ("div", 1), ("mul", 2), ("add", 1)]
    program = chain_program(MACHINE, nodes, 3, dtype)
    km.verify(program)
    compute = km.source(program, "compute")
    assert compute.count("_binary_tile(0, 1, 0);") == len(nodes)
    assert compute.count("pack_tile(") == 1
    assert km.source(program, "dm0").count("noc_async_read_page(") == 3
    longer = chain_program(MACHINE, nodes + [("mul", 0)], 3, dtype)
    assert km.fingerprint(longer) != km.fingerprint(program)
