"""The scoring function — the one place league rules become numbers (§4.4.1).

Applies ``league.yaml`` scoring to raw ``load_player_stats`` columns. NEVER uses
the precomputed ``fantasy_points_ppr`` — computing from raw box scores through our
own config is what makes the custom-scoring edge real (§3, §4.4.1).

Authoritative column map (§4.4.1):
  * fumbles  = rushing_fumbles_lost + receiving_fumbles_lost + sack_fumbles_lost (×-2)
  * 2-pt     = passing_2pt + rushing_2pt + receiving_2pt (×2)
  * return TD = special_teams_tds (×6)  (exact for skill players; non-return ST = DST)
Fumbles and 2-pt are each split across THREE columns and MUST be summed.
"""
from __future__ import annotations

import pandas as pd

from src.config import ScoringConfig, load_config

# Component -> contributing raw column(s). Multi-column components are summed.
FUMBLE_COLS = ["rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost"]
TWO_PT_COLS = ["passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions"]

# All raw columns scoring reads, for validation/staging.
REQUIRED_COLS = [
    "passing_yards", "passing_tds", "passing_interceptions",
    "rushing_yards", "rushing_tds",
    "receptions", "receiving_yards", "receiving_tds",
    "special_teams_tds",
    *TWO_PT_COLS, *FUMBLE_COLS,
]


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Numeric column, 0-filled if absent (a missing component contributes 0 points)."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def _sum_cols(df: pd.DataFrame, cols) -> pd.Series:
    total = pd.Series(0.0, index=df.index)
    for c in cols:
        total = total + _col(df, c)
    return total


def score_dataframe(df: pd.DataFrame, scoring: ScoringConfig | None = None) -> pd.Series:
    """Fantasy points per row from raw component columns (vectorized)."""
    if scoring is None:
        scoring = load_config().scoring
    if df.empty:
        return pd.Series(dtype=float)

    pts = (
        _col(df, "passing_yards") * scoring.pass_yd
        + _col(df, "passing_tds") * scoring.pass_td
        + _col(df, "passing_interceptions") * scoring.int
        + _col(df, "rushing_yards") * scoring.rush_yd
        + _col(df, "rushing_tds") * scoring.rush_td
        + _col(df, "receptions") * scoring.rec
        + _col(df, "receiving_yards") * scoring.rec_yd
        + _col(df, "receiving_tds") * scoring.rec_td
        + _sum_cols(df, TWO_PT_COLS) * scoring.two_pt
        + _sum_cols(df, FUMBLE_COLS) * scoring.fumble_lost
        + _col(df, "special_teams_tds") * scoring.return_td
    )
    return pts.astype(float)


def score_stat_line(stat_line: dict, scoring: ScoringConfig | None = None) -> float:
    """Score a single raw stat-line dict (used by tests and label staging)."""
    return float(score_dataframe(pd.DataFrame([stat_line]), scoring).iloc[0])
