"""Per-player-type interval calibration (DrafterSpec.md §4.6.1).

Naive quantile intervals are overconfident exactly for rookies/role-changers
(out-of-distribution, few similar training rows). This rescales interval widths per
player-type bucket from OOF residuals so a rookie's interval is honestly wide.

Width scales come from **split-conformal** calibration: the scale per side is a
quantile of the normalized nonconformity score (residual ÷ predicted half-width),
which targets the empirical coverage fraction directly. The earlier mean-matching
scale (avg half-width matched to a residual quantile) under-covered whenever
half-widths were heterogeneous across rows — matching means is not matching the
coverage quantile. The conformal scale fixes that and includes the finite-sample
``(n+1)/n`` correction.

Buckets are made mutually exclusive by the priority
``rookie > 2nd-year > team-changed vet > established vet`` (thin-history classes win,
since history depth drives width more than situational change). A ``min_bucket_n``
fallback drops to global calibration so it never runs on a degenerate sample.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

GLOBAL = "_global"


def assign_bucket(row: pd.Series) -> str:
    """Mutually-exclusive player-type bucket by the §4.6.1 priority."""
    if bool(row.get("is_rookie", False)):
        return "rookie"
    soh = row.get("seasons_of_history")
    if soh is not None and not pd.isna(soh) and float(soh) <= 1:
        return "second_year"
    if float(row.get("team_changed", 0) or 0) == 1:
        return "team_changed_vet"
    return "established_vet"


@dataclass
class IntervalCalibrator:
    min_bucket_n: int = 30
    lower_alpha: float = 0.10   # target mass below P10  (=> 0.80 central coverage)
    upper_alpha: float = 0.10   # target mass above P90
    # per bucket: (lower_scale, upper_scale) applied to half-widths around p50
    scales: Dict[str, tuple] = field(default_factory=dict)

    def fit(self, oof: pd.DataFrame) -> "IntervalCalibrator":
        """Fit split-conformal width scales from OOF rows (cols p10,p50,p90,y,bucket).

        Targets the nominal 80% central interval (P10..P90) per bucket, falling back
        to the global scale for buckets thinner than ``min_bucket_n``.
        """
        self.scales[GLOBAL] = self._fit_one(oof)
        for bucket, grp in oof.groupby("bucket"):
            if len(grp) >= self.min_bucket_n:
                self.scales[bucket] = self._fit_one(grp)
        return self

    def _fit_one(self, df: pd.DataFrame) -> tuple:
        """Split-conformal scale per side: quantile of (residual / half-width).

        Lower bound L = p50 - ls*(p50-p10) covers at 1-lower_alpha iff
        ls = Q_{1-lower_alpha} of the score (p50 - y)/(p50 - p10). Symmetric for the
        upper side. The (n+1)/n bump is the standard finite-sample conformal
        correction (clamped to the max observed score).
        """
        p50 = df["p50"].to_numpy(dtype=float)
        y = df["y"].to_numpy(dtype=float)
        lower_hw = np.clip(p50 - df["p10"].to_numpy(dtype=float), 1e-6, None)
        upper_hw = np.clip(df["p90"].to_numpy(dtype=float) - p50, 1e-6, None)
        score_lower = (p50 - y) / lower_hw   # > ls  <=> y falls below L
        score_upper = (y - p50) / upper_hw   # > us  <=> y falls above U
        ls = self._conformal_scale(score_lower, self.lower_alpha)
        us = self._conformal_scale(score_upper, self.upper_alpha)
        # never shrink below the raw model interval; only widen overconfident ones
        return (max(ls, 1.0), max(us, 1.0))

    @staticmethod
    def _conformal_scale(scores: np.ndarray, alpha: float) -> float:
        scores = scores[~np.isnan(scores)]
        n = scores.size
        if n == 0:
            return 1.0
        q = min(1.0, (1.0 - alpha) * (n + 1) / n)   # finite-sample conformal level
        return float(np.quantile(scores, q))

    def transform(self, preds: pd.DataFrame, buckets: pd.Series) -> pd.DataFrame:
        out = preds.copy()
        for i, bucket in buckets.items():
            ls, us = self.scales.get(bucket, self.scales.get(GLOBAL, (1.0, 1.0)))
            p50 = out.at[i, "p50"]
            out.at[i, "p10"] = p50 - ls * (p50 - out.at[i, "p10"])
            out.at[i, "p90"] = p50 + us * (out.at[i, "p90"] - p50)
        # re-assert monotonicity after scaling
        vals = np.sort(out[["p10", "p50", "p90"]].to_numpy(), axis=1)
        out[["p10", "p50", "p90"]] = vals
        return out
