"""Pooled quantile GBM — the forecasting core (DrafterSpec.md §4.6.1).

One pooled LightGBM per quantile (P10/P50/P90) trained across ALL positions with
``position`` as a categorical feature — NOT four per-position models (small-N TE/QB
would overfit; §4.6.1). Trees split on ``position`` where data supports and borrow
strength elsewhere; per-position rank is taken after inference.

Quantile crossing is fixed by **rearrangement** (sorting the three predictions per
row), NOT clipping — clipping is asymmetric and biases the interval, which feeds the
recommender's risk math (Chernozhukov et al. 2010, §4.6.1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

QUANTILES = (0.10, 0.50, 0.90)
QUANTILE_NAMES = {0.10: "p10", 0.50: "p50", 0.90: "p90"}

DEFAULT_PARAMS = {
    "objective": "quantile",
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
}


def rearrange_quantiles(preds: np.ndarray) -> np.ndarray:
    """Sort each row's quantile predictions ascending (monotonicity guarantee)."""
    return np.sort(preds, axis=1)


@dataclass
class QuantileGBM:
    """Three pooled quantile models (one per level) sharing a feature schema."""

    quantiles: tuple = QUANTILES
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    models: Dict[float, object] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantileGBM":
        if lgb is None:
            raise ImportError("lightgbm is required to train the projection engine")
        self.feature_names = list(X.columns)
        cat = [c for c in X.columns if str(X[c].dtype) == "category"]
        for q in self.quantiles:
            params = dict(self.params)
            params["alpha"] = q
            model = lgb.LGBMRegressor(**params)
            model.fit(X, y, categorical_feature=cat or "auto")
            self.models[q] = model
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with monotonic p10/p50/p90 columns (rearranged)."""
        X = X.reindex(columns=self.feature_names)
        raw = np.column_stack([self.models[q].predict(X) for q in self.quantiles])
        ordered = rearrange_quantiles(raw)
        cols = [QUANTILE_NAMES[q] for q in sorted(self.quantiles)]
        return pd.DataFrame(ordered, columns=cols, index=X.index)
