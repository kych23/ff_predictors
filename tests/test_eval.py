"""Tests for projection evaluation metrics."""
import numpy as np

from src.projection.eval import (
    dcg_at_k,
    interval_coverage,
    ndcg_at_k,
    pinball_loss,
    spearman,
    top_n_hit_rate,
)


def test_ndcg_perfect_ranking():
    scores = [10, 8, 6, 4, 2]
    relevance = [10, 8, 6, 4, 2]
    assert ndcg_at_k(scores, relevance, 5) == 1.0


def test_ndcg_all_zero_relevance():
    assert ndcg_at_k([1, 2, 3], [0, 0, 0], 3) == 0.0


def test_ndcg_reversed_ranking():
    scores = [2, 4, 6, 8, 10]
    relevance = [10, 8, 6, 4, 2]
    val = ndcg_at_k(scores, relevance, 5)
    assert 0.0 < val < 1.0


def test_ndcg_k_larger_than_data():
    val = ndcg_at_k([10, 5], [10, 5], k=10)
    assert val == 1.0


def test_pinball_loss_perfect():
    assert pinball_loss([10, 20], [10, 20], 0.5) == 0.0


def test_pinball_loss_asymmetric():
    loss_over = pinball_loss([15], [10], 0.9)
    loss_under = pinball_loss([5], [10], 0.9)
    assert loss_over > loss_under


def test_interval_coverage_all_inside():
    assert interval_coverage([7, 8, 9], [5, 5, 5], [10, 10, 10]) == 1.0


def test_interval_coverage_at_boundary():
    assert interval_coverage([5.0], [5.0], [10.0]) == 1.0
    assert interval_coverage([10.0], [5.0], [10.0]) == 1.0


def test_interval_coverage_all_outside():
    assert interval_coverage([1, 2], [5, 5], [10, 10]) == 0.0


def test_spearman_perfect():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0


def test_spearman_inverse():
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0


def test_top_n_hit_rate_perfect():
    assert top_n_hit_rate([10, 8, 6, 4, 2], [10, 8, 6, 4, 2], 3) == 1.0


def test_top_n_hit_rate_miss():
    val = top_n_hit_rate([2, 4, 6, 8, 10], [10, 8, 6, 4, 2], 2)
    assert val < 1.0
