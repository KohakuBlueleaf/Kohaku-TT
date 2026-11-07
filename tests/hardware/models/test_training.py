"""Regression: eager training on tt and captured training replay."""

import pytest
import torch

from kohakutt.bench.capture import graph_check
from kohakutt.bench.models.train_check import train


@pytest.mark.hw
@pytest.mark.parametrize("kind", ["mlp", "block", "cnn"])
@pytest.mark.parametrize("foreach", [False, True])
def test_training_on_tt(native, kind, foreach):
    native.reset_fallback_counts()
    losses, _ = train("tt", 5, foreach, clip=foreach, kind=kind)
    assert losses[-1] < losses[0]
    assert native.fallback_counts() == {}


@pytest.mark.hw
def test_captured_function_matches_eager(native):
    assert graph_check.small()["replay_eq_eager"]


@pytest.mark.hw
@pytest.mark.parametrize("kind", ["mlp", "cnn"])
def test_captured_training_matches_eager(native, kind):
    record = graph_check.training(6, kind)
    assert record["max_rel_dev"] == 0.0
    assert record["loss_decreases"]
    assert record["replay_ms_per_step"] < record["eager_ms_per_step"]
    torch.tt.synchronize()
