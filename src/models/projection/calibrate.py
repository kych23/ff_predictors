"""Per-player-type interval calibration.

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


PORTED from src/projection/calibrate.py (DraftEngineDesign.md §9.2).
Edits: v2 config API (load_league), scoring moves to domain, and the
leakage guard is reached through platform.asof — the only platform
subpackage this layer may import (§9.0, §11.3).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

GLOBAL = "_global"


def assign_bucket(row: pd.Series) -> str:
    """Mutually-exclusive player-type bucket by the priority."""
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
    scales: dict[str, tuple] = field(default_factory=dict)

    def fit(self, oof: pd.DataFrame) -> IntervalCalibrator:
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
        """Widen intervals per bucket. ``buckets`` aligns to ``preds`` by row
        order (vectorized — safe under duplicate index labels)."""
        if len(buckets) != len(preds):
            raise ValueError(
                f"buckets length {len(buckets)} != preds length {len(preds)}")
        out = preds.copy()
        default = self.scales.get(GLOBAL, (1.0, 1.0))
        scale_of = {b: self.scales.get(b, default) for b in buckets.unique()}
        ls = np.array([scale_of[b][0] for b in buckets], dtype=float)
        us = np.array([scale_of[b][1] for b in buckets], dtype=float)
        p10 = out["p10"].to_numpy(dtype=float)
        p50 = out["p50"].to_numpy(dtype=float)
        p90 = out["p90"].to_numpy(dtype=float)
        widened = np.column_stack([p50 - ls * (p50 - p10), p50, p50 + us * (p90 - p50)])
        # re-assert monotonicity after scaling
        out[["p10", "p50", "p90"]] = np.sort(widened, axis=1)
        return out


def coverage_by_bucket(oof: pd.DataFrame) -> pd.DataFrame:
    """Empirical P10–P90 coverage per bucket (diagnostic; nominal is 0.80).

    ``oof`` needs columns p10/p90/y/bucket. Returns bucket, n, coverage rows
    with an ``_all`` aggregate first.
    """
    df = oof.copy()
    df["inside"] = (df["y"] >= df["p10"]) & (df["y"] <= df["p90"])
    rows = [{"bucket": "_all", "n": len(df), "coverage": float(df["inside"].mean())}]
    for b, g in df.groupby("bucket"):
        rows.append({"bucket": b, "n": len(g), "coverage": float(g["inside"].mean())})
    return pd.DataFrame(rows)


def calibrate_oof_lofo(oof: pd.DataFrame, min_bucket_n: int) -> pd.DataFrame:
    """Leave-one-season-out calibration of OOF predictions (honest coverage).

    Fitting a calibrator on all OOF residuals and transforming those same rows
    lets each row's own residual shrink its own nonconformity score, inflating
    reported coverage. Here each test season is transformed by a calibrator fit
    on the OTHER test seasons only. Falls back to fit-on-self when there is a
    single test season (LOFO is undefined there).

    ``oof`` needs columns p10/p50/p90/y/bucket/season. Returns a copy with
    calibrated quantile columns; the input is not mutated.
    """
    seasons = oof["season"].unique()
    out = oof.copy()
    if len(seasons) < 2:
        cal = IntervalCalibrator(min_bucket_n).fit(oof)
        out[["p10", "p50", "p90"]] = cal.transform(
            oof[["p10", "p50", "p90"]], oof["bucket"])
        return out
    for s in seasons:
        mask = out["season"] == s
        cal = IntervalCalibrator(min_bucket_n).fit(oof[~mask])
        out.loc[mask, ["p10", "p50", "p90"]] = cal.transform(
            oof.loc[mask, ["p10", "p50", "p90"]], oof.loc[mask, "bucket"]).values
    return out
