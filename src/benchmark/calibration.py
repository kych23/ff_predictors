"""Tier-3 risk calibration benchmark (DrafterSpec.md §4.7).

Validates the risk-modeling edge: interval coverage BY player-type (not just overall
— naive intervals are overconfident exactly for rookies/role-changers, §4.6.1) plus
pinball loss at each quantile.
"""
from __future__ import annotations

import pandas as pd

from src.projection.calibrate import assign_bucket
from src.projection.eval import interval_coverage, pinball_loss

NOMINAL_COVERAGE = 0.80  # P10..P90 central interval


def run_calibration_benchmark(oof: pd.DataFrame) -> pd.DataFrame:
    """Per-bucket + overall coverage and pinball loss.

    oof columns: p10, p50, p90, y, and enough fields for ``assign_bucket``
    (is_rookie, seasons_of_history, team_changed).
    """
    df = oof.copy()
    for col, default in [("is_rookie", False), ("seasons_of_history", float("nan")),
                         ("team_changed", 0)]:
        if col not in df.columns:
            df[col] = default
    df["bucket"] = df.apply(assign_bucket, axis=1)

    def _row(g: pd.DataFrame, label: str) -> dict:
        return {
            "bucket": label,
            "n": len(g),
            "coverage_p10_p90": interval_coverage(g["y"], g["p10"], g["p90"]),
            "nominal": NOMINAL_COVERAGE,
            "pinball_p10": pinball_loss(g["y"], g["p10"], 0.10),
            "pinball_p50": pinball_loss(g["y"], g["p50"], 0.50),
            "pinball_p90": pinball_loss(g["y"], g["p90"], 0.90),
        }

    rows = [_row(df, "overall")]
    for bucket, g in df.groupby("bucket"):
        rows.append(_row(g, bucket))
    return pd.DataFrame(rows)
