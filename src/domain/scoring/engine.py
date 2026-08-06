"""Offensive scoring — the one place league rules become numbers (§10.2).

Applies ``league.yaml`` scoring to raw nflverse ``load_player_stats``
columns. **Never** uses the precomputed ``fantasy_points_ppr`` column:
computing from raw box scores through our own config is what makes custom
scoring real and what makes the §21.1 reconciliation gate meaningful.

Column names below were verified against ``nflreadpy`` 0.1.3 on 2026-08-04, not
assumed. Two components split across three columns each and MUST be summed —
missing one is a silent scoring error that no test of the total would catch.
"""
from __future__ import annotations

from typing import Final, Mapping, Sequence

import numpy as np
import pandas as pd

#: Fumbles lost, split across three columns.
FUMBLE_COLS: Final = (
    "rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost",
)
#: Two-point conversions, split across three columns.
TWO_PT_COLS: Final = (
    "passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions",
)

#: scoring key -> column(s) contributing to it. Multi-column entries are summed.
OFFENSE_COLUMN_MAP: Final[Mapping[str, tuple[str, ...]]] = {
    "pass_yd": ("passing_yards",),
    "pass_td": ("passing_tds",),
    "int": ("passing_interceptions",),
    "rush_yd": ("rushing_yards",),
    "rush_td": ("rushing_tds",),
    "rec": ("receptions",),
    "rec_yd": ("receiving_yards",),
    "rec_td": ("receiving_tds",),
    # exact for skill players; non-return special-teams scores belong to DST
    "return_td": ("special_teams_tds",),
    "two_pt": TWO_PT_COLS,
    "fumble_lost": FUMBLE_COLS,
    # MEASURED 2026-08-04: 39 events across 2012-2025. Earlier revisions
    # declared this underivable and carried a scoring exception for it.
    "off_fumble_return_td": ("fumble_recovery_tds",),
}

#: Every raw column the offensive scorer reads.
REQUIRED_COLUMNS: Final = tuple(
    sorted({c for cols in OFFENSE_COLUMN_MAP.values() for c in cols})
)


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    """Numeric column, 0-filled if absent — a missing component scores 0."""
    if col not in df.columns:
        return pd.Series(0.0, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def score_offense(df: pd.DataFrame, offense: Mapping[str, float]) -> pd.Series:
    """Fantasy points per row from raw component columns (vectorized).

    Unknown scoring keys raise rather than being ignored: a typo in
    ``league.yaml`` that silently scored nothing would be invisible in the
    output and would still change ``model_version``.
    """
    unknown = set(offense) - set(OFFENSE_COLUMN_MAP)
    if unknown:
        raise KeyError(
            f"scoring.offense has keys with no column mapping: {sorted(unknown)}. "
            f"Known keys: {sorted(OFFENSE_COLUMN_MAP)}"
        )

    # A frame with rows but no stat columns must score 0 per row, not vanish —
    # hence len(df), not df.empty.
    if len(df) == 0:
        return pd.Series(dtype="float64")

    total = pd.Series(0.0, index=df.index, dtype="float64")
    for key, coefficient in offense.items():
        component = sum(
            (_numeric(df, c) for c in OFFENSE_COLUMN_MAP[key]),
            start=pd.Series(0.0, index=df.index, dtype="float64"),
        )
        total = total + component * float(coefficient)
    return total


def score_stat_line(stat_line: Mapping[str, float], offense: Mapping[str, float]) -> float:
    """Score a single raw stat-line mapping."""
    return float(score_offense(pd.DataFrame([dict(stat_line)]), offense).iloc[0])


def score_tensor(
    stats: np.ndarray, columns: Sequence[str], offense: Mapping[str, float]
) -> np.ndarray:
    """Vectorized scorer for the simulation path (§9.2).

    ``stats`` has shape ``(..., n_columns)`` with ``columns`` naming the last
    axis. Returns ``stats.shape[:-1]``. Kept alongside the DataFrame path so the
    kernel never carries pandas into its inner loop.
    """
    index = {c: i for i, c in enumerate(columns)}
    out = np.zeros(stats.shape[:-1], dtype=np.float64)
    for key, coefficient in offense.items():
        for col in OFFENSE_COLUMN_MAP[key]:
            if col in index:
                out += stats[..., index[col]] * float(coefficient)
    return out
