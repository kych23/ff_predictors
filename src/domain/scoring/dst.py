"""Defense/special-teams scoring (§10.2, §12.5).

MEASURED 2026-08-04: every configured DST component is derivable from
``nflreadpy.load_team_stats``, with points allowed coming from
``load_schedules``. No play-by-play work needed.

The points-allowed band spans 14 points from shutout to blowout, which makes
DST scoring lumpy and heavily driven by game script — a defense facing a weak
offense in a low-total game collects the shutout tier, the sacks and the
turnovers together. That lumpiness is precisely why §12.5 samples DST from an
empirical weekly distribution rather than a mean: every team starts one, so
expected value nearly cancels across teams while variance does not, and the
weekly-high prize pays for the maximum over twelve.
"""
from __future__ import annotations

from typing import Final, Mapping, Sequence

import numpy as np
import pandas as pd

#: scoring key -> team-stat column(s), summed. Verified against nflreadpy 0.1.3.
DST_COLUMN_MAP: Final[Mapping[str, tuple[str, ...]]] = {
    "sack": ("def_sacks",),
    "interception": ("def_interceptions",),
    # opponent fumbles recovered by this defense
    "fumble_recovery": ("fumble_recovery_opp",),
    "td": ("def_tds",),
    "safety": ("def_safeties",),
    "blocked_kick": ("fg_blocked", "pat_blocked", "pt_blocked"),
    "return_td": ("pt_return_tds",),
    # no dedicated column; rare enough that 0 is the honest default
    "extra_point_returned": (),
}

REQUIRED_COLUMNS: Final = tuple(
    sorted({c for cols in DST_COLUMN_MAP.values() for c in cols})
)


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def points_allowed_score(
    points_allowed: pd.Series | np.ndarray, tiers: Sequence[Sequence[float]]
) -> np.ndarray:
    """Map points allowed onto the configured tier table.

    Tiers are validated to tile [0, 999] with no gap or overlap (§10.5 rule 4),
    so exactly one matches every non-negative input.
    """
    pa = np.asarray(points_allowed, dtype="float64")
    out = np.full(pa.shape, np.nan, dtype="float64")
    for lo, hi, pts in tiers:
        out = np.where((pa >= lo) & (pa <= hi), float(pts), out)
    if np.isnan(out).any():
        bad = np.unique(pa[np.isnan(out)])
        raise ValueError(f"points allowed outside the configured tiers: {bad[:5]}")
    return out


def score_dst(
    df: pd.DataFrame,
    dst: Mapping[str, object],
    *,
    points_allowed_col: str = "points_allowed",
) -> pd.Series:
    """Fantasy points per (team, week) row.

    ``df`` carries the team-stat columns plus ``points_allowed_col``, which the
    caller joins from the schedule (the opponent's realized score).
    """
    if len(df) == 0:
        return pd.Series(dtype="float64")

    total = pd.Series(0.0, index=df.index, dtype="float64")
    for key, cols in DST_COLUMN_MAP.items():
        if key not in dst or not cols:
            continue
        component = sum(
            (_numeric(df, c) for c in cols),
            start=pd.Series(0.0, index=df.index, dtype="float64"),
        )
        total = total + component * float(dst[key])  # type: ignore[arg-type]

    if points_allowed_col not in df.columns:
        raise KeyError(
            f"score_dst needs {points_allowed_col!r}; join the opponent's "
            f"realized score from load_schedules first"
        )
    tiers = dst["points_allowed_tiers"]  # type: ignore[index]
    total = total + points_allowed_score(df[points_allowed_col], tiers)  # type: ignore[arg-type]
    return total


def implied_points(
    total_line: pd.Series, spread_line: pd.Series, *, is_home: pd.Series
) -> pd.Series:
    """Vegas-implied points for a team (§12.5, §6 team context).

    ``total_line/2 - spread_line/2`` for the home team and
    ``total_line/2 + spread_line/2`` for the away team, following nflverse's
    convention that ``spread_line`` is signed from the home team's perspective
    (negative = home favored).

    Only Week-1 lines are preseason-available (§11.3), so this is a season-level
    team-strength proxy, not a per-week conditioner.
    """
    half_spread = spread_line / 2.0
    return total_line / 2.0 - np.where(is_home, half_spread, -half_spread)
