"""Pooled quantile GBM — the forecasting core.

One pooled LightGBM per quantile (P10/P50/P90) trained across ALL positions with
``position`` as a categorical feature — NOT four per-position models (small-N TE/QB
would overfit; ). Trees split on ``position`` where data supports and borrow
strength elsewhere; per-position rank is taken after inference.

Quantile crossing is fixed by **rearrangement** (sorting the three predictions per
row), NOT clipping — clipping is asymmetric and biases the interval, which feeds the
recommender's risk math (Chernozhukov et al. 2010).


PORTED from src/projection/quantile_model.py (DraftEngineDesign.md §9.2).
Edits: v2 config API (load_league), scoring moves to domain, and the
leakage guard is reached through platform.asof — the only platform
subpackage this layer may import (§9.0, §11.3).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

QUANTILES = (0.10, 0.50, 0.90)
QUANTILE_NAMES = {0.10: "p10", 0.50: "p50", 0.90: "p90"}

DEFAULT_PARAMS = {
    "objective": "quantile",
    "n_estimators": 800,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,   # pin bagging/feature-fraction seeds for reproducible retrains
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
    # training category levels per categorical column; predict() realigns serving
    # frames to these so LightGBM's category codes can never silently permute.
    categories: Dict[str, list] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: pd.Series, *,
            X_val: pd.DataFrame | None = None,
            y_val: pd.Series | None = None,
            early_stopping_rounds: int = 50) -> "QuantileGBM":
        """Fit the three quantile models with early stopping + refit.

        Early stopping picks the tree count on a holdout (the caller's temporal
        ``X_val`` when given, else a random 15% carve-out), then each model is
        REFIT on train+holdout at that tree count — the holdout only tunes
        n_trees, its rows are never lost from training.
        """
        if lgb is None:
            raise ImportError("lightgbm is required to train the projection engine")
        self.feature_names = list(X.columns)
        cat = [c for c in X.columns if str(X[c].dtype) == "category"]
        self.categories = {c: list(X[c].cat.categories) for c in cat}
        if X_val is not None:
            # align holdout category levels to training's so the ES eval and the
            # train+holdout refit share one code space (concat of mismatched
            # categoricals silently degrades to object dtype).
            X_val = X_val.copy()
            for c in cat:
                if c in X_val.columns:
                    X_val[c] = pd.Categorical(X_val[c], categories=self.categories[c])

        use_early_stop = True
        if X_val is None or y_val is None:
            n = len(X)
            if n < 60:
                use_early_stop = False
            else:
                rng = np.random.RandomState(42)
                val_idx = rng.choice(n, size=max(int(n * 0.15), 10), replace=False)
                train_idx = np.setdiff1d(np.arange(n), val_idx)
                X_val = X.iloc[val_idx]
                y_val = y.iloc[val_idx]
                X = X.iloc[train_idx]
                y = y.iloc[train_idx]

        for q in self.quantiles:
            params = dict(self.params)
            params["alpha"] = q
            fit_kwargs: dict = {"categorical_feature": cat or "auto"}
            if not use_early_stop:
                model = lgb.LGBMRegressor(**params)
                model.fit(X, y, **fit_kwargs)
                self.models[q] = model
                continue

            probe = lgb.LGBMRegressor(**params)
            probe.fit(X, y, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False),
                                 lgb.log_evaluation(period=0)],
                      **fit_kwargs)
            best_iter = probe.best_iteration_ or params["n_estimators"]
            params["n_estimators"] = int(best_iter)
            model = lgb.LGBMRegressor(**params)
            model.fit(pd.concat([X, X_val]), pd.concat([y, y_val]), **fit_kwargs)
            self.models[q] = model
        return self

    def feature_importance(self, quantile: float = 0.50) -> pd.Series:
        """Gain importance of the fitted model at ``quantile``, descending."""
        model = self.models.get(quantile)
        if model is None:
            return pd.Series(dtype=float)
        gains = model.booster_.feature_importance(importance_type="gain")
        return pd.Series(gains, index=self.feature_names).sort_values(ascending=False)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with monotonic p10/p50/p90 columns (rearranged)."""
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            logger.warning("predict: %d training feature(s) absent from serving "
                           "frame, filled NaN: %s", len(missing), missing)
        X = X.reindex(columns=self.feature_names)
        for c, cats in self.categories.items():
            X[c] = pd.Categorical(X[c], categories=cats)
        raw = np.column_stack([self.models[q].predict(X) for q in self.quantiles])
        ordered = rearrange_quantiles(raw)
        cols = [QUANTILE_NAMES[q] for q in sorted(self.quantiles)]
        return pd.DataFrame(ordered, columns=cols, index=X.index)
