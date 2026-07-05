"""Black-box extensions for projection evaluation metrics.

Signature note (from test_eval.py asymmetry test): pinball_loss(y_true, y_pred, q).
"""
import numpy as np

from src.projection.eval import (
    interval_coverage,
    ndcg_at_k,
    pinball_loss,
    spearman,
    top_n_hit_rate,
)


# --- pinball ---

def test_pinball_symmetric_at_median():
    assert pinball_loss([12], [10], 0.5) == pinball_loss([8], [10], 0.5)


def test_pinball_low_quantile_penalizes_underprediction_of_pred():
    # q=0.1: prediction should sit LOW; y >> pred is cheap, y << pred is costly
    cheap = pinball_loss([15], [10], 0.1)   # actual above the P10 line: fine
    costly = pinball_loss([5], [10], 0.1)   # actual below the P10 line: bad
    assert costly > cheap


def test_pinball_nonnegative():
    rng = np.random.default_rng(3)
    y = rng.normal(10, 5, 50)
    pred = rng.normal(10, 5, 50)
    for q in (0.1, 0.5, 0.9):
        assert pinball_loss(y, pred, q) >= 0.0


def test_pinball_zero_only_when_equal():
    assert pinball_loss([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 0.9) == 0.0
    assert pinball_loss([1.0, 2.0, 3.0], [1.0, 2.5, 3.0], 0.9) > 0.0


# --- interval coverage ---

def test_interval_coverage_partial():
    assert interval_coverage([7, 20], [5, 5], [10, 10]) == 0.5


def test_interval_coverage_mixed_bounds():
    # one inside, one below, one above, one on lower boundary
    cov = interval_coverage([7, 1, 99, 5], [5, 5, 5, 5], [10, 10, 10, 10])
    assert cov == 0.5


# --- ndcg ---

def test_ndcg_bounds_on_random_data():
    rng = np.random.default_rng(7)
    for _ in range(10):
        scores = rng.normal(size=20)
        rel = rng.uniform(0, 10, 20)
        v = ndcg_at_k(scores, rel, 10)
        assert 0.0 <= v <= 1.0 + 1e-12


def test_ndcg_single_swap_below_perfect():
    rel = [10, 8, 6, 4, 2]
    perfect = ndcg_at_k(rel, rel, 5)
    swapped = ndcg_at_k([10, 6, 8, 4, 2], rel, 5)
    assert swapped < perfect == 1.0


def test_ndcg_k_one_best_item_first():
    assert ndcg_at_k([10, 1, 1], [10, 1, 1], 1) == 1.0


# --- spearman ---

def test_spearman_monotone_transform_invariant():
    import pytest
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert spearman(x, list(np.exp(x))) == pytest.approx(1.0)


def test_spearman_bounds():
    rng = np.random.default_rng(11)
    a = rng.normal(size=30)
    b = rng.normal(size=30)
    assert -1.0 <= spearman(a, b) <= 1.0


# --- top-n hit rate ---

def test_top_n_hit_rate_half_overlap():
    # top-2 by score picks the true #1 plus a dud -> 1/2 hit
    relevance = [10, 9, 1, 2]
    scores = [10, 1, 9, 2]
    assert top_n_hit_rate(scores, relevance, 2) == 0.5


def test_top_n_hit_rate_zero_overlap():
    relevance = [10, 9, 1, 1, 1, 1]
    scores = [0, 0, 0, 0, 5, 6]
    assert top_n_hit_rate(scores, relevance, 2) == 0.0


def test_top_n_hit_rate_bounds():
    rng = np.random.default_rng(13)
    scores = rng.normal(size=25)
    rel = rng.uniform(0, 30, 25)
    v = top_n_hit_rate(scores, rel, 10)
    assert 0.0 <= v <= 1.0
