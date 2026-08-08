"""Cutoff filters and their matching hard assertions (§11.2, §21.2).

Every filter is paired with an assertion, and the accessor applies both. A
filter alone is a convention; the assertion is what makes it a boundary.
"""
from __future__ import annotations

import pandas as pd

from src.core.errors import EmptySourceError, LeakageError


def prior_seasons(df: pd.DataFrame, target_season: int, *,
                  season_col: str = "season") -> pd.DataFrame:
    """Rows strictly before ``target_season``."""
    if df is None or df.empty or season_col not in df.columns:
        return df
    return df[pd.to_numeric(df[season_col], errors="coerce") < target_season].copy()


def prior_weeks(df: pd.DataFrame, target_season: int, target_week: int, *,
                season_col: str = "season", week_col: str = "week") -> pd.DataFrame:
    """Rows before ``(target_season, target_week)`` in (season, week) order."""
    if df is None or df.empty or season_col not in df.columns:
        return df
    s = pd.to_numeric(df[season_col], errors="coerce")
    w = (pd.to_numeric(df[week_col], errors="coerce")
         if week_col in df.columns else pd.Series(0, index=df.index))
    return df[(s < target_season) | ((s == target_season) & (w < target_week))].copy()


def preseason_rows(df: pd.DataFrame, target_season: int, *,
                   season_col: str = "season", week_col: str = "week"
                   ) -> pd.DataFrame:
    """Week-1-effective rows of ``target_season``.

    Allowed because they precede kickoff. Week 2 onward is in-season and is
    excluded — this is the only place a row from the target season may pass.
    """
    if df is None or df.empty or season_col not in df.columns:
        return df
    out = df[pd.to_numeric(df[season_col], errors="coerce") == target_season].copy()
    if week_col in out.columns:
        wk = pd.to_numeric(out[week_col], errors="coerce")
        out = out[wk.isna() | (wk <= 1)]
    return out


def assert_no_future(df: pd.DataFrame, target_season: int, *,
                     season_col: str = "season", where: str = "") -> None:
    if df is None or df.empty or season_col not in df.columns:
        return
    bad = pd.to_numeric(df[season_col], errors="coerce") >= target_season
    if bool(bad.any()):
        raise LeakageError(
            f"LEAKAGE: {int(bad.sum())} row(s) at/after season {target_season} "
            f"reached feature construction{(' in ' + where) if where else ''}"
        )


def assert_no_future_week(df: pd.DataFrame, target_season: int, target_week: int,
                          *, season_col: str = "season", week_col: str = "week",
                          where: str = "") -> None:
    if df is None or df.empty or season_col not in df.columns:
        return
    s = pd.to_numeric(df[season_col], errors="coerce")
    w = (pd.to_numeric(df[week_col], errors="coerce")
         if week_col in df.columns else pd.Series(0, index=df.index))
    bad = (s > target_season) | ((s == target_season) & (w >= target_week))
    if bool(bad.any()):
        raise LeakageError(
            f"LEAKAGE: {int(bad.sum())} row(s) at/after ({target_season}, "
            f"week {target_week}) reached feature construction"
            f"{(' in ' + where) if where else ''}"
        )


def assert_non_empty(df: pd.DataFrame, *, where: str) -> pd.DataFrame:
    """A source that filters to nothing is an error, not a warning.

    Silently-empty sources are how a pipeline produces confident garbage: every
    downstream join succeeds, every feature is NaN, every projection is a prior,
    and nothing raises.
    """
    if df is None or len(df) == 0:
        raise EmptySourceError(
            f"{where} is empty after cutoff filtering. Either the snapshot is "
            f"missing this asset or the cutoff excludes everything."
        )
    return df
