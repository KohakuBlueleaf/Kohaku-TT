"""KTT device on card 0 (own process, KTT_HW=1): kernels match tinygrad's CPU backend.
Errors are "direct" (within DIRECT_FACTOR of the bridge op's own error), "exact"
(data movement only) or "recorded" (finite and below 1)."""

import numpy as np
import pytest

pytest.importorskip("tinygrad.engine")

from kohakutt.tinygrad_backend import core, executor, install, overrides, workloads
from kohakutt.tinygrad_backend import device as D

pytestmark = [
    pytest.mark.hw,
    pytest.mark.skipif(not core.available(), reason="_ktt_native.so not built"),
]

DIRECT_FACTOR = 2.0
BASE = {c[0]: c for c in workloads.WORKLOADS}
OV = {c[0]: c for c in workloads.OVERRIDE_WORKLOADS}
GRAD = {c[0]: c for c in workloads.GRAD_WORKLOADS}
EXACT = {"transpose", "max pool 2x2"}
ALL_ON_DEVICE = [
    "linear+bias",
    "linear+bias+gelu",
    "layernorm",
    "rmsnorm (manual)",
    "softmax",
    "elementwise gelu(x*y+z)",
    "sdpa",
    "transpose",
    "sum all 4096x4096",
    "max pool 2x2",
]


def rel(x, ref):
    x, ref = np.asarray(x, np.float64), np.asarray(ref, np.float64)
    return float(np.max(np.abs(x - ref)) / (np.max(np.abs(ref)) + 1e-12))


def bf16_up(a):
    return core.upload(
        (np.ascontiguousarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16),
        core.BF16,
    )


def fp32_down(t):
    d = core.download(t)
    return (d.astype(np.uint32) << 16).view(np.float32) if d.dtype == np.uint16 else d


def direct_mm_error(shapes):
    x, w = workloads.inputs(shapes)[:2]
    ref = x.astype(np.float64) @ w.T.astype(np.float64)
    return rel(
        core.download(core.mm(core.upload(x), core.upload(w), transpose_b=True)), ref
    )


def direct_sdpa_error(causal):
    q, k, v = workloads.inputs([(1, 4, 128, 64)] * 3)
    s = np.einsum("bhqd,bhkd->bhqk", q.astype(np.float64), k.astype(np.float64)) / 8.0
    if causal:
        s = s + np.triu(np.full((128, 128), -np.inf), 1)
    p = np.exp(s - s.max(-1, keepdims=True))
    ref = np.einsum(
        "bhqk,bhkd->bhqd", p / p.sum(-1, keepdims=True), v.astype(np.float64)
    )
    return rel(
        fp32_down(core.sdpa(bf16_up(q), bf16_up(k), bf16_up(v), None, causal)), ref
    )


def direct_conv_error(stride):
    x, w = workloads.inputs([(2, 16, 32, 32), (32, 16, 3, 3)])
    pad = 1 if stride == 1 else 0
    xp = np.pad(x.astype(np.float64), ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    oh = (xp.shape[2] - 3) // stride + 1
    cols = np.stack(
        [
            xp[:, :, i : i + stride * oh : stride, j : j + stride * oh : stride]
            for i in range(3)
            for j in range(3)
        ],
        2,
    )
    ref = np.einsum("nckhw,ock->nohw", cols, w.reshape(32, 16, 9).astype(np.float64))
    out = core.conv2d(
        core.upload(x), core.upload(w), (stride, stride), (pad, pad), (1, 1), 1
    )
    return rel(core.download(out), ref)


def direct_norm_error(kind):
    (x,) = workloads.inputs([(64, 256)])[:1]
    d = x.astype(np.float64)
    if kind == "ln":
        y, _, _ = core.layer_norm(core.upload(x), 1, 1e-5, None, None)
        ref = (d - d.mean(1, keepdims=True)) / np.sqrt(d.var(1, keepdims=True) + 1e-5)
    else:
        y, _, _ = core.rms_norm_train(core.upload(x), 1e-6, None)
        ref = d / np.sqrt((d**2).mean(1, keepdims=True) + 1e-6)
    return rel(core.download(y), ref)


DIRECT = {  # case name -> function giving the bridge op's own error on the same data
    "linear+bias": lambda: direct_mm_error([(64, 256), (512, 256)]),
    "linear+bias+gelu": lambda: direct_mm_error([(64, 256), (512, 256)]),
    "ov sdpa": lambda: direct_sdpa_error(False),
    "ov sdpa causal": lambda: direct_sdpa_error(True),
    "ov conv2d k3 p1": lambda: direct_conv_error(1),
    "ov conv2d k3 s2": lambda: direct_conv_error(2),
}
# Norm cases are "recorded": the fp32 CPU reference's own roundoff (~1e-7) equals them.


@pytest.fixture(scope="module", autouse=True)
def card():
    install(executor="bridge", overrides=False)
    D.get_backend().ensure_open()
    yield
    D.get_backend().close()


def judge(name, err, record_property):
    record_property("error", err)
    print(f"[error] {name}: {err:.3e}")
    if name in DIRECT:
        direct = DIRECT[name]()
        record_property("direct_bridge_op_error", direct)
        print(f"[error] {name}: bridge op on its own {direct:.3e}")
        assert err <= DIRECT_FACTOR * max(direct, 1e-7), (name, err, direct)
    elif name in EXACT:
        assert err == 0.0, (name, err)
    else:
        assert np.isfinite(err) and err < 1.0, (name, err)


@pytest.mark.parametrize("name", ALL_ON_DEVICE)
def test_workload_on_device(name, record_property):
    _, fn, shapes = BASE[name]
    executor.FALLBACKS.clear()
    overrides.enable(False)
    err, kernels, calls = workloads.run_case(fn, shapes, name=name)
    judge(name, err, record_property)
    assert not executor.FALLBACKS, dict(executor.FALLBACKS)
    assert D.STATS["device"] == kernels


@pytest.mark.parametrize("name", list(OV))
def test_override_on_device(name, record_property):
    _, fn, shapes = OV[name]
    overrides.enable(True)
    try:
        err, kernels, calls = workloads.run_case(fn, shapes, name=name)
    finally:
        overrides.enable(False)
    judge(name, err, record_property)
    assert any(k.startswith("lib:") for k in calls), calls


@pytest.mark.parametrize("name", list(GRAD))
def test_gradient_on_device(name, record_property):
    _, fn, shapes = GRAD[name]
    overrides.enable(True)
    try:
        errs, kernels, calls = workloads.run_grad_case(fn, shapes)
    finally:
        overrides.enable(False)
    record_property("errors", errs)
    print(f"[error] {name}: {[f'{e:.2e}' for e in errs]}")
    assert all(np.isfinite(e) and e < 1.0 for e in errs), errs


def test_data_stays_on_the_card_between_kernels():
    from kohakutt.tinygrad_backend.compat import Tensor

    D.reset_stats()
    before = core.counters()
    x = Tensor(np.ones((64, 64), np.float32), device="KTT").realize()
    z = (((x + 1) * 2).realize() + 1).realize()
    assert np.allclose(z.numpy(), 5.0)
    after = core.counters()
    assert after["uploads"] - before["uploads"] == 1
    assert after["downloads"] - before["downloads"] == 1


@pytest.mark.parametrize("name", ["pad + slice", "cat", "permute 4d + reshape"])
def test_memory_ops_exact_without_gather(name):
    _, fn, shapes = BASE[name]
    executor.FALLBACKS.clear()
    executor.ROUTES.clear()
    overrides.enable(False)
    err, kernels, calls = workloads.run_case(fn, shapes, name=name)
    assert err == 0.0, err
    assert not executor.FALLBACKS and not executor.ROUTES["reads_gather"], (
        dict(executor.FALLBACKS),
        dict(executor.ROUTES),
    )


@pytest.mark.parametrize("name", ["conv2d k3 p1", "conv2d k3 s2"])
def test_conv_without_override_uses_memory_ops(name, record_property):
    _, fn, shapes = BASE[name]
    executor.FALLBACKS.clear()
    executor.ROUTES.clear()
    overrides.enable(False)
    err, kernels, calls = workloads.run_case(fn, shapes, name=name)
    record_property("error", err)
    direct = direct_conv_error(1 if "p1" in name else 2)
    print(f"[error] {name}: {err:.3e} (direct bridge conv {direct:.3e})")
    assert np.isfinite(err) and err < 1.0
    assert not executor.FALLBACKS and not executor.ROUTES["reads_gather"], (
        dict(executor.FALLBACKS),
        dict(executor.ROUTES),
    )


def test_cumsum_is_the_bridge_cumsum(record_property):
    _, fn, shapes = BASE["cumsum"]
    (x,) = workloads.inputs(shapes)[:1]
    ref = np.cumsum(x.astype(np.float64), -1)
    direct = rel(core.download(core.cumsum(core.upload(x), 1)), ref)
    executor.FALLBACKS.clear()
    executor.BRIDGE_CALLS.clear()
    err, kernels, calls = workloads.run_case(fn, shapes, name="cumsum")
    record_property("error", err)
    record_property("direct_bridge_op_error", direct)
    print(f"[error] cumsum: {err:.3e}, bridge cumsum on its own {direct:.3e}")
    assert not executor.FALLBACKS and executor.BRIDGE_CALLS["cumsum"] >= 1
    # Recorded, not bounded: the fp32 CPU reference has roundoff of this size.
    assert np.isfinite(err) and err < 1.0


def test_embedding_is_embedding_rows(record_property):
    _, fn, shapes = BASE["embedding-like gather"]
    executor.FALLBACKS.clear()
    executor.BRIDGE_CALLS.clear()
    err, kernels, calls = workloads.run_case(fn, shapes, name="embedding-like gather")
    record_property("error", err)
    print(f"[error] embedding: {err:.3e}")
    assert not executor.FALLBACKS and executor.BRIDGE_CALLS["embedding_rows"] == 1
    assert np.isfinite(err) and err < 1.0
