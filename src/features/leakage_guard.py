"""Leakage enforcement at construction time (DrafterSpec.md §4.0/§4.5).

The most important correctness property. Features for a row keyed (player, season Y)
may read ONLY information available before season Y's Week-1 kickoff. Because
features are stored as opaque JSONB, this can't be a DB/SQL check — it is enforced
HERE, at the data-access boundary: every source frame handed to a feature function
is filtered to rows strictly before the cutoff, and a guard asserts nothing from
season >= Y (or after ``as_of``) slips through.

``tests/test_leakage.py`` poisons a future value with a sentinel and asserts it
never reaches any output feature.
"""
from __future__ import annotations

import pandas as pd


class LeakageError(AssertionError):
    """Raised when a source frame contains data at/after the as-of cutoff."""


def prior_seasons(df: pd.DataFrame, target_season: int, season_col: str = "season") -> pd.DataFrame:
    """Return only rows from seasons strictly before ``target_season``.

    This is the season-grained boundary filter for backward-looking sources
    (player_stats, ff_opportunity, snap_counts).
    """
    if df is None or df.empty or season_col not in df.columns:
        return df
    return df[pd.to_numeric(df[season_col], errors="coerce") < target_season].copy()


def assert_no_future(df: pd.DataFrame, target_season: int, season_col: str = "season",
                     *, where: str = "") -> None:
    """Hard assert that ``df`` carries no rows at/after ``target_season``."""
    if df is None or df.empty or season_col not in df.columns:
        return
    bad = pd.to_numeric(df[season_col], errors="coerce") >= target_season
    if bool(bad.any()):
        n = int(bad.sum())
        raise LeakageError(
            f"LEAKAGE: {n} row(s) at/after season {target_season} reached "
            f"feature construction{(' in ' + where) if where else ''}."
        )


def preseason_rows(df: pd.DataFrame, target_season: int, season_col: str = "season",
                   week_col: str = "week") -> pd.DataFrame:
    """Preseason-Y artifacts (depth chart / roster): season == Y, Week-1-effective.

    Allowed because they are known before kickoff. In-season (>= Week 2) rows are
    excluded — only the Week-1-effective snapshot is rule-compliant (§4.0).
    """
    if df is None or df.empty or season_col not in df.columns:
        return df
    out = df[pd.to_numeric(df[season_col], errors="coerce") == target_season].copy()
    if week_col in out.columns:
        wk = pd.to_numeric(out[week_col], errors="coerce")
        out = out[(wk.isna()) | (wk <= 1)]
    return out
