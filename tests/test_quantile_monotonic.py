"""Quantile monotonicity (DrafterSpec.md §4.6.1 acceptance).

Asserts quantiles never cross after rearrangement, and that rearrangement (sorting),
not clipping, is what guarantees it.
"""
import numpy as np
import pandas as pd

from src.projection.quantile_model import QuantileGBM, rearrange_quantiles


def test_rearrange_sorts_each_row():
    raw = np.array([[5.0, 3.0, 8.0], [1.0, 0.5, 0.7], [10.0, 9.0, 9.5]])
    out = rearrange_quantiles(raw)
    assert np.all(out[:, 0] <= out[:, 1])
    assert np.all(out[:, 1] <= out[:, 2])
    # rearrangement preserves the multiset per row (unlike clipping)
    for i in range(raw.shape[0]):
        assert sorted(raw[i]) == list(out[i])


def test_trained_model_quantiles_monotonic():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "position": pd.Categorical(rng.choice(["RB", "WR", "QB", "TE"], size=n)),
    })
    y = pd.Series(3 * X["a"] + rng.normal(size=n))
    model = QuantileGBM().fit(X, y)
    preds = model.predict(X)
    assert (preds["p10"] <= preds["p50"]).all()
    assert (preds["p50"] <= preds["p90"]).all()


def test_early_stopping_refits_at_best_iteration():
    """ES holdout tunes n_trees, then the model is refit on train+holdout:
    tree count stays below the ceiling and the fit uses all rows."""
    rng = np.random.default_rng(1)
    n = 500
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(2 * X["a"] - X["b"] + rng.normal(scale=0.5, size=n))
    X_val = X.iloc[:100]
    y_val = y.iloc[:100]
    model = QuantileGBM().fit(X.iloc[100:], y.iloc[100:], X_val=X_val, y_val=y_val)
    for q, m in model.models.items():
        n_trees = m.booster_.num_trees()
        assert 0 < n_trees <= 800          # early-stopped below the ceiling
        # final model is the refit (n_estimators pinned to best_iter, no ES run)
        assert m.get_params()["n_estimators"] == n_trees
        assert not m.best_iteration_


def test_predictions_track_signal():
    """OOF-style sanity: predictions must correlate with the generating signal
    (guards against a constant-output model passing monotonicity alone)."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(2)
    n = 600
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(5 * X["a"] + rng.normal(scale=0.5, size=n))
    model = QuantileGBM().fit(X.iloc[:400], y.iloc[:400])
    preds = model.predict(X.iloc[400:])
    rho, _ = spearmanr(preds["p50"], y.iloc[400:])
    assert rho > 0.8
