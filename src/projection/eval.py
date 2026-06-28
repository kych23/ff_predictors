"""Projection metrics, shared with the benchmark (DrafterSpec.md §4.6/§4.7.1).

Ports the old regression metrics and adds the ranking metrics the gate needs:
NDCG@k (primary gate metric, §4.7.1), Spearman, top-N hit rate, and pinball
(quantile) loss for risk calibration.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def spearman(y_true, y_pred) -> float:
    rho, _ = spearmanr(y_true, y_pred, nan_policy="omit")
    return float(rho)


def pinball_loss(y_true, y_pred, quantile: float) -> float:
    """Quantile (pinball) loss — lower is better; validates interval edges."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    return float(np.nanmean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    rel = np.asarray(relevance, dtype=float)[:k]
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    return float(np.sum(rel * discounts))


def ndcg_at_k(scores, relevance, k: int) -> float:
    """NDCG@k: rank items by ``scores``, reward by true ``relevance`` (actual FPPG).

    Primary Tier-1 gate metric (§4.7.1) — draft value is dominated by the top of
    the board, so this weights getting the top right.
    """
    scores = np.asarray(scores, dtype=float)
    relevance = np.asarray(relevance, dtype=float)
    order = np.argsort(-scores)
    ranked_rel = relevance[order]
    ideal_rel = np.sort(relevance)[::-1]
    idcg = dcg_at_k(ideal_rel, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked_rel, k) / idcg


def top_n_hit_rate(scores, relevance, n: int) -> float:
    """Fraction of the true top-n that the predicted top-n captures."""
    scores = np.asarray(scores, dtype=float)
    relevance = np.asarray(relevance, dtype=float)
    pred_top = set(np.argsort(-scores)[:n])
    true_top = set(np.argsort(-relevance)[:n])
    if not true_top:
        return 0.0
    return len(pred_top & true_top) / len(true_top)


def interval_coverage(y_true, p_low, p_high) -> float:
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(p_low, dtype=float)
    hi = np.asarray(p_high, dtype=float)
    inside = (y >= lo) & (y <= hi)
    return float(np.nanmean(inside.astype(float)))
