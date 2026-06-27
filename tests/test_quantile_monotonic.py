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
