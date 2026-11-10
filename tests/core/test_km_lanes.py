"""Kernel variants, lanes (multicast) and the K-block passes of the metal level without
a card: the programs verify, their sources and descriptions, and the forms the model
refuses when they are built."""

import json

import pytest
from km_programs import example

import kohakutt.dsl.metal as km
from km_lanes import MACHINE, lane_program


def _gemm():
    return example("gemm_km")


def _spec(program, count):
    from kohakutt.dsl.metal.bridge import spec

    sources = {name: f"{name}.cpp" for name in program.kernels}
    return json.loads(spec(program, sources, count, None))


def test_lane_program_verifies_and_emits_the_handshake():
    program = lane_program()
    km.verify(program)
    send, recv = km.source(program, "send"), km.source(program, "recv")
    assert "noc_async_write_multicast(x_ptr, get_noc_multicast_addr(" in send
    assert (
        "noc_async_writes_flushed();\n" in send.split("noc_semaphore_set_multicast")[0]
    )
    assert "noc_semaphore_wait(" in send and "lane_to_n" in send
    assert "noc_semaphore_inc(get_noc_addr(lane_from_x, lane_from_y," in recv
    assert "noc_async_atomic_barrier();" in recv


def test_lane_description_has_variants_and_core_coordinates(monkeypatch):
    import kohakutt.dsl.metal.bridge as bridge

    monkeypatch.setattr(bridge, "_compute_options", lambda *_: {})
    spec = _spec(lane_program(), MACHINE.cores)
    kernels = {k["source"]: k for k in spec["kernels"]}
    assert kernels["send.cpp"]["cores"] == [[0, 0, 3, 0]]
    assert kernels["recv.cpp"]["cores"] == [[0, 1, 3, 2]]
    args = dict((tuple(core), a) for core, a in kernels["send.cpp"]["runtime_args"])
    # Column 2's rectangle: rows 1 .. 2 on NoC 0, two receivers.
    count, sx, sy, ex, ey = args[(2, 0)]
    assert (sx, sy, ex, ey) == ({"noc": [2, 1, 0]}, {"noc": [2, 1, 1]}) + (
        {"noc": [2, 2, 0]},
        {"noc": [2, 2, 1]},
    ) and count == 2
    recv = dict((tuple(core), a) for core, a in kernels["recv.cpp"]["runtime_args"])
    assert recv[(3, 2)] == [{"noc": [3, 0, 0]}, {"noc": [3, 0, 1]}]


def test_gemm_program_verifies_with_its_lane_and_passes():
    module = _gemm()
    program = module.traced(11, 10)
    names = sorted(program.kernels)
    assert names == ["compute", "dm1", "w_recv", "w_send"]
    compute = km.source(program, "compute")
    assert (
        "compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_w, cb_out);" in compute
    )
    assert compute.count("pack_reconfig_l1_acc(1);") == 1
    assert "copy_block(cb_part, 0, 0, sub);" in compute
    values = module.plan(128, 32, 24, 11, 10)
    assert values["nk"] == 2 and values["pcm"] == 2
    assert program.core_count(values) == 110


def test_refused_when_built():
    p = km.Program("refused", MACHINE)
    p.kernel("compute")
    with pytest.raises(ValueError, match="compute is one kernel named compute"):
        p.kernel("compute", cores=p.cores.rows(1), name="other")
    with pytest.raises(ValueError, match="compute is one kernel named compute"):
        p.kernel("dm1", name="compute")
    with pytest.raises(ValueError, match="two compute kernels"):
        p.kernel("compute")
    p.semaphore(0, name="s")
    with pytest.raises(ValueError, match="two semaphores named s"):
        p.semaphore(0, name="s")
    p.core_ref(0, name="peer")
    with pytest.raises(ValueError, match="two cores named peer"):
        p.core_ref(1, name="peer")
    with pytest.raises(ValueError, match="senders are one row"):
        p.lane(p.cores.rows(0, 1), p.cores.rows(2))
    p.kernel("dm0")
    with pytest.raises(ValueError, match="each name their rows"):
        p.kernel("dm0", cores=p.cores.rows(1), name="more")
    part = p.partials(
        24, pages=p.sized("block", of=p.runtime("b"), exact=True, limit=8)
    )
    with pytest.raises(ValueError, match="only km.matmul_passes"):
        p.kernel("dm1").reserve(part, 1)


def test_matmul_passes_take_a_block_sized_partials_buffer():
    p = km.Program("passes", MACHINE)
    a, w, out = (
        p.input("a", "bfloat16"),
        p.input("w", "bfloat16"),
        p.output("o", "bfloat16"),
    )
    sh, sw = p.choose(("sh", "sw"), [(1, 1)])
    bm, bn = p.runtime("bm", multiple_of="sh"), p.runtime("bn", multiple_of="sw")
    wrong = p.partials(24, pages=p.sized("wrong", of=bm * 2, exact=True, limit=8))
    c = p.kernel("compute")
    shape = dict(bm=bm, bn=bn, sh=sh, sw=sw, kb=1, nk=p.runtime("nk"), kb_last=1)
    shape.update(a_block=bm, w_block=bn, transpose=0)
    shape["sub"] = p.sized("sub", of=sh * sw, exact=True, limit=4)
    cbs = p.cb(0, like=a), p.cb(1, like=w), p.cb(16, like=out)
    with pytest.raises(ValueError, match="holds bm \\* bn pages"):
        km.matmul_passes(c, shape, cbs[0], cbs[1], wrong, cbs[2])


def test_counterparts_of_the_new_seeds_verify():
    import test_km_verify_seeded as seeded

    seeded.loop_over_a_loop_variable("v", same=True)
    seeded.copy_of_sub_pages(rounded=False)


def test_values_declared_once_and_bounded_at_launch():
    p = km.Program("values", MACHINE)
    p.choose(("sh", "sw"), [(1, 1), (2, 2)])
    with pytest.raises(ValueError, match="declared twice"):
        p.choose(("sh", "sw"), [(4, 4)])
    with pytest.raises(ValueError, match="declared twice"):
        p.runtime("sh")
    p.runtime("n", multiple_of=2)
    with pytest.raises(ValueError, match="declared twice, differently"):
        p.runtime("n", multiple_of=4)
    kb = p.runtime("kb")
    p.limit(p.runtime("kb_last"), low=1, high=kb)
    with pytest.raises(ValueError, match="kb_last = 5 is outside"):
        p.value_list({"sh": 1, "sw": 1, "n": 2, "kb": 4, "kb_last": 5}, 1)


def test_passes_refuse_undeclared_bounds_and_inline_on_partials():
    p = km.Program("bounds", MACHINE)
    with pytest.raises(ValueError, match="nk declared >= 1"):
        km.k_passes(p.kernel("dm0"), p.runtime("nk"), lambda *_: None)
    part = p.partials(
        24, pages=p.sized("block", of=p.runtime("b"), exact=True, limit=8)
    )
    with pytest.raises(ValueError, match="does not use a partials buffer"):
        p.kernel("dm1").inline("", uses=[km.CBUse(part, reserve=1, push=1)])
