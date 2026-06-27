"""Per-player-type interval calibration (DrafterSpec.md §4.6.1).

Naive quantile intervals are overconfident exactly for rookies/role-changers
(out-of-distribution, few similar training rows). This rescales interval widths per
player-type bucket from OOF residuals so a rookie's interval is honestly wide.

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
    # per bucket: (lower_scale, upper_scale) applied to half-widths around p50
    scales: Dict[str, tuple] = field(default_factory=dict)

    def fit(self, oof: pd.DataFrame) -> "IntervalCalibrator":
        """Fit width scales from OOF rows with columns p10,p50,p90,y,bucket.

        Targets the nominal 80% central interval (P10..P90): scales the predicted
        half-widths to match empirical residual quantiles.
        """
        self.scales[GLOBAL] = self._fit_one(oof)
        for bucket, grp in oof.groupby("bucket"):
            if len(grp) >= self.min_bucket_n:
                self.scales[bucket] = self._fit_one(grp)
        return self

    @staticmethod
    def _fit_one(df: pd.DataFrame) -> tuple:
        resid = df["y"] - df["p50"]
        lower_resid = np.nanquantile(resid, 0.10)   # negative
        upper_resid = np.nanquantile(resid, 0.90)   # positive
        lower_hw = np.nanmean((df["p50"] - df["p10"]).clip(lower=1e-6))
        upper_hw = np.nanmean((df["p90"] - df["p50"]).clip(lower=1e-6))
        ls = float(abs(lower_resid) / lower_hw) if lower_hw > 0 else 1.0
        us = float(upper_resid / upper_hw) if upper_hw > 0 else 1.0
        # never shrink below the raw model interval; only widen overconfident ones
        return (max(ls, 1.0), max(us, 1.0))

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
