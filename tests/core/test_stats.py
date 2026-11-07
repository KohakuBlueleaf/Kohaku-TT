"""diff_stats and format_stats."""

import math

import torch

from kohakutt import diff_stats, format_stats

KEYS = {"abs_max", "abs_median", "abs_mean", "rel_max", "rel_median", "rel_mean"}


def test_identical_tensors_have_zero_error_and_cosine_one():
    ref = torch.randn(64, 64, dtype=torch.float64)
    stats = diff_stats(ref.clone(), ref)
    assert all(stats[k] == 0.0 for k in KEYS)
    assert math.isclose(stats["cosine"], 1.0, rel_tol=1e-12)
    assert stats["nonfinite"] == 0


def test_known_perturbation():
    ref = torch.tensor([1.0, 2.0, 4.0, -8.0], dtype=torch.float64)
    result = ref + torch.tensor([0.1, -0.2, 0.0, 0.4], dtype=torch.float64)
    stats = diff_stats(result, ref)
    assert math.isclose(stats["abs_max"], 0.4)
    assert math.isclose(stats["abs_mean"], 0.7 / 4)
    # torch.median of an even count is the lower middle value.
    assert math.isclose(stats["abs_median"], 0.1)
    assert math.isclose(stats["rel_max"], 0.1)
    assert math.isclose(stats["rel_mean"], (0.1 + 0.1 + 0.0 + 0.05) / 4)
    assert math.isclose(stats["rel_median"], 0.05)
    expected = float(result @ ref / (result.norm() * ref.norm()))
    assert math.isclose(stats["cosine"], expected, rel_tol=1e-12)


def test_zero_reference_positions_are_excluded_from_rel():
    ref = torch.tensor([0.0, 2.0], dtype=torch.float64)
    result = torch.tensor([1.0, 2.2], dtype=torch.float64)
    stats = diff_stats(result, ref)
    assert math.isclose(stats["abs_max"], 1.0)
    assert math.isclose(stats["rel_max"], 0.1)
    assert math.isclose(stats["rel_mean"], 0.1)


def test_nonfinite_results_are_counted_and_skipped():
    ref = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    result = torch.tensor([1.0, float("nan"), float("inf"), 4.5], dtype=torch.float64)
    stats = diff_stats(result, ref)
    assert stats["nonfinite"] == 2
    assert math.isclose(stats["abs_max"], 0.5)
    assert math.isclose(stats["rel_max"], 0.125)


def test_mixed_dtypes_compare_in_float64():
    ref = torch.randn(32, 32, dtype=torch.float64)
    stats = diff_stats(ref.bfloat16(), ref)
    assert 0.0 < stats["rel_max"] <= 2.0**-8
    assert stats["cosine"] > 0.0


def test_format_stats_lists_every_statistic():
    line = format_stats(diff_stats(torch.ones(4), torch.ones(4)))
    for word in ("abs max", "rel max", "med", "mean", "cos", "nonfinite"):
        assert word in line
