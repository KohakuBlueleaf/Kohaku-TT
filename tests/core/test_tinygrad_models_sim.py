"""Model parity (MLP, convnet, ViT, GPT) on the numpy model of the binding against
tinygrad's CPU backend: forward errors are bf16-sized only where a bf16 kernel runs
(bound: 16 bf16 steps); training starts at the CPU loss and decreases it."""

import numpy as np
import pytest

pytest.importorskip("tinygrad.engine")

from kohakutt.tinygrad_backend import device as D
from kohakutt.tinygrad_backend import executor, install, models, overrides

BF16_EPS = 2.0**-8
BF16_MODELS = {
    "vit",
    "gpt",
    "cnn",
    "cnn_pool",
}  # attention (vit, gpt) or conv (cnn) runs through a bf16 kernel
EXACT_MODELS = {
    "mlp",
    "gpt_mask",
}  # only fp32 ops (the explicit mask disables the attention kernel)


@pytest.fixture(scope="module", autouse=True)
def sim_device():
    install(executor="sim", overrides=True)
    yield
    overrides.enable(False)
    D.get_backend().close()


def rel(a, b):
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-9))


@pytest.mark.parametrize("kind", list(models.MODELS))
def test_forward_matches_cpu(kind):
    m = models.MODELS[kind]()
    err = rel(m.run("KTT"), m.run("CPU"))
    assert err < (16 * BF16_EPS if kind in BF16_MODELS else 1e-5), (kind, err)


@pytest.mark.parametrize("kind", ["mlp", "cnn", "cnn_pool"])
def test_training_tracks_cpu(kind):
    m = models.MODELS[kind]()
    executor.FALLBACKS.clear()
    lk, _ = m.train("KTT", 3)
    assert not executor.FALLBACKS, dict(executor.FALLBACKS)
    lc, _ = m.train("CPU", 3)
    assert abs(lk[0] - lc[0]) / lc[0] < (16 * BF16_EPS if kind in BF16_MODELS else 1e-5)
    assert lk[-1] < lk[0] and lc[-1] < lc[0]


def test_adam_weight_difference_is_the_key_bias():
    # The key bias has a zero true gradient (softmax ignores a row shift): Adam turns
    # its fp32 rounding noise into +-lr steps on either backend, so only it differs.
    D_ = 64
    m = models.gpt(D=D_, depth=1, heads=2, T=16, vocab=64, explicit_mask=True)
    _, pk = m.train("KTT", 2)
    _, pc = m.train("CPU", 2)
    differ = {k: rel(pk[k], pc[k]) for k in pc if rel(pk[k], pc[k]) > 1e-3}
    assert set(differ) <= {"b0.att.qkv.b"}, differ
    d = np.abs(pk["b0.att.qkv.b"] - pc["b0.att.qkv.b"])
    assert d[:D_].max() < 1e-4 and d[2 * D_ :].max() < 1e-4, d
