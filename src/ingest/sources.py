""" Wrapper for nflreadpy

Returns pandas DataFrames
"""
from __future__ import annotations

from typing import Iterable, List

import nflreadpy as nfl
import pandas as pd


def _to_pandas(obj) -> pd.DataFrame:
    return obj.to_pandas() if hasattr(obj, "to_pandas") else obj


def nflreadpy_version() -> str:
    try:
        return getattr(nfl, "__version__", "unknown")
    except Exception:
        return "unknown"


def load_players() -> pd.DataFrame:
    return _to_pandas(nfl.load_players())


def load_player_stats(seasons: Iterable[int]) -> pd.DataFrame:
    return _to_pandas(nfl.load_player_stats(seasons=list(seasons)))


def load_schedules(seasons: Iterable[int]) -> pd.DataFrame:
    return _to_pandas(nfl.load_schedules(seasons=list(seasons)))


def load_depth_charts(seasons: Iterable[int]) -> pd.DataFrame:
    return _to_pandas(nfl.load_depth_charts(seasons=list(seasons)))


def load_snap_counts(seasons: Iterable[int]) -> pd.DataFrame:
    return _to_pandas(nfl.load_snap_counts(seasons=list(seasons)))


def load_ff_opportunity(seasons: Iterable[int]) -> pd.DataFrame:
    return _to_pandas(nfl.load_ff_opportunity(seasons=list(seasons)))


def load_draft_picks() -> pd.DataFrame:
    return _to_pandas(nfl.load_draft_picks())


def load_combine() -> pd.DataFrame:
    return _to_pandas(nfl.load_combine())


def load_ff_playerids() -> pd.DataFrame:
    return _to_pandas(nfl.load_ff_playerids())


def load_rosters(seasons: Iterable[int]) -> pd.DataFrame:
    return _to_pandas(nfl.load_rosters(seasons=list(seasons)))


def week1_kickoff_dates(seasons: List[int]) -> dict:
    """Week-1 kickoff date per season

    Derived from the schedule's earliest Week-1 regular-season game date
    """
    sched = load_schedules(seasons)
    out: dict = {}
    if sched.empty or "gameday" not in sched.columns:
        return out
    df = sched.copy()
    df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    wk1 = df[(df.get("week") == 1) & (df.get("game_type", "REG") == "REG")]
    for season, grp in wk1.groupby("season"):
        d = grp["gameday"].min()
        if pd.notnull(d):
            out[int(season)] = d
    return out
