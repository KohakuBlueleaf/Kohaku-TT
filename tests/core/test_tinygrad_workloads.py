"""KTT device on the numpy executor (no card): the workloads match tinygrad's CPU."""

import pytest

pytest.importorskip("tinygrad.engine")

from kohakutt.tinygrad_backend import install, workloads

CASES = workloads.WORKLOADS


@pytest.fixture(scope="module", autouse=True)
def numpy_device():
    install(executor="numpy", overrides=False)


@pytest.mark.parametrize("name,fn,shapes", CASES, ids=[c[0] for c in CASES])
def test_workload_matches_cpu(name, fn, shapes):
    err, kernels, calls = workloads.run_case(fn, shapes, name=name)
    assert kernels >= 1
    assert err < 1e-4, (name, err)
