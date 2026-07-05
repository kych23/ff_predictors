"""Tests for the round -> risk-appetite quantile schedule."""
import numpy as np

from src.recommender.quantile_schedule import (
    EARLY_ALPHA,
    LATE_ALPHA,
    round_to_alpha,
    value_vector_at_alpha,
)


def test_round_boundaries():
    assert round_to_alpha(1, 15) == EARLY_ALPHA
    assert round_to_alpha(15, 15) == LATE_ALPHA


def test_alpha_monotonic_across_rounds():
    alphas = [round_to_alpha(r, 15) for r in range(1, 16)]
    assert all(a <= b for a, b in zip(alphas, alphas[1:]))


def test_single_round_draft():
    assert round_to_alpha(1, 1) == EARLY_ALPHA


def test_out_of_range_round_clamped():
    assert round_to_alpha(0, 15) == EARLY_ALPHA
    assert round_to_alpha(99, 15) == LATE_ALPHA


def test_value_at_known_quantiles():
    p10, p50, p90 = np.array([5.0]), np.array([10.0]), np.array([15.0])
    assert value_vector_at_alpha(p10, p50, p90, 0.10)[0] == 5.0
    assert value_vector_at_alpha(p10, p50, p90, 0.50)[0] == 10.0
    assert value_vector_at_alpha(p10, p50, p90, 0.90)[0] == 15.0


def test_value_interpolates_between_quantiles():
    p10, p50, p90 = np.array([5.0]), np.array([10.0]), np.array([15.0])
    v = value_vector_at_alpha(p10, p50, p90, 0.30)[0]
    assert 5.0 < v < 10.0
    v = value_vector_at_alpha(p10, p50, p90, 0.70)[0]
    assert 10.0 < v < 15.0


def test_alpha_outside_band_clamped():
    p10, p50, p90 = np.array([5.0]), np.array([10.0]), np.array([15.0])
    assert value_vector_at_alpha(p10, p50, p90, 0.01)[0] == 5.0
    assert value_vector_at_alpha(p10, p50, p90, 0.99)[0] == 15.0
