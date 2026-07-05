"""Black-box extensions for the round -> risk-appetite quantile schedule."""
import numpy as np

from src.recommender.quantile_schedule import (
    EARLY_ALPHA,
    LATE_ALPHA,
    round_to_alpha,
    value_vector_at_alpha,
)


def test_early_alpha_below_late_alpha():
    # early rounds are risk-averse (low quantile), late rounds chase upside
    assert EARLY_ALPHA < LATE_ALPHA


def test_alpha_always_within_band():
    for total in (1, 8, 15, 20):
        for r in range(-2, total + 5):
            a = round_to_alpha(r, total)
            assert EARLY_ALPHA <= a <= LATE_ALPHA


def test_value_monotone_in_alpha():
    p10, p50, p90 = np.array([5.0]), np.array([10.0]), np.array([15.0])
    alphas = [0.10, 0.25, 0.50, 0.66, 0.80, 0.90]
    vals = [value_vector_at_alpha(p10, p50, p90, a)[0] for a in alphas]
    for lo, hi in zip(vals, vals[1:]):
        assert lo <= hi


def test_value_vectorized_shape():
    n = 5
    p10 = np.arange(n, dtype=float)
    p50 = p10 + 5.0
    p90 = p10 + 10.0
    v = value_vector_at_alpha(p10, p50, p90, 0.5)
    assert len(v) == n
    assert (v == p50).all()


def test_degenerate_quantiles_constant_value():
    p = np.array([7.0, 7.0])
    for a in (0.10, 0.37, 0.50, 0.83, 0.90):
        v = value_vector_at_alpha(p, p, p, a)
        assert (v == 7.0).all()


def test_value_stays_within_p10_p90_band():
    rng = np.random.default_rng(0)
    p50 = rng.uniform(5, 20, 20)
    p10 = p50 - rng.uniform(1, 5, 20)
    p90 = p50 + rng.uniform(1, 5, 20)
    for a in (0.05, 0.10, 0.33, 0.50, 0.71, 0.90, 0.95):
        v = value_vector_at_alpha(p10, p50, p90, a)
        assert (v >= p10 - 1e-12).all()
        assert (v <= p90 + 1e-12).all()


def test_full_draft_alpha_schedule_spans_band():
    alphas = [round_to_alpha(r, 15) for r in range(1, 16)]
    assert alphas[0] == EARLY_ALPHA
    assert alphas[-1] == LATE_ALPHA
    # strictly reaches both ends and never leaves the band
    assert min(alphas) == EARLY_ALPHA
    assert max(alphas) == LATE_ALPHA
