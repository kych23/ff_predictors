""" Wrapper for nflreadpy

Returns pandas DataFrames
"""
from __future__ import annotations

from typing import Iterable, List

import nflreadpy as nfl
import pandas as pd


import logging as _logging

_log = _logging.getLogger(__name__)


def _to_pandas(obj) -> pd.DataFrame:
    return obj.to_pandas() if hasattr(obj, "to_pandas") else obj


def _safe_load_seasons(loader_fn, seasons: list, label: str) -> pd.DataFrame:
    """Load one season at a time; skip seasons whose data isn't published yet (404)."""
    frames = []
    for season in seasons:
        try:
            frames.append(_to_pandas(loader_fn([season])))
        except Exception as e:
            msg = str(e)
            if ("404" in msg or "Not Found" in msg
                    or "ConnectionError" in type(e).__name__
                    or "Season must be between" in msg
                    or ("season" in msg.lower() and "current_season" in msg.lower())):
                _log.warning("%s: skipping season %d (not yet available)", label, season)
            else:
                raise
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def nflreadpy_version() -> str:
    try:
        return getattr(nfl, "__version__", "unknown")
    except Exception:
        return "unknown"


def load_players() -> pd.DataFrame:
    return _to_pandas(nfl.load_players())


def load_player_stats(seasons: Iterable[int]) -> pd.DataFrame:
    return _safe_load_seasons(lambda s: nfl.load_player_stats(seasons=s), list(seasons), "load_player_stats")


def load_schedules(seasons: Iterable[int]) -> pd.DataFrame:
    return _safe_load_seasons(lambda s: nfl.load_schedules(seasons=s), list(seasons), "load_schedules")


def load_depth_charts(seasons: Iterable[int]) -> pd.DataFrame:
    return _safe_load_seasons(lambda s: nfl.load_depth_charts(seasons=s), list(seasons), "load_depth_charts")


def load_snap_counts(seasons: Iterable[int]) -> pd.DataFrame:
    return _safe_load_seasons(lambda s: nfl.load_snap_counts(seasons=s), list(seasons), "load_snap_counts")


def load_ff_opportunity(seasons: Iterable[int]) -> pd.DataFrame:
    return _safe_load_seasons(lambda s: nfl.load_ff_opportunity(seasons=s), list(seasons), "load_ff_opportunity")


def load_draft_picks() -> pd.DataFrame:
    return _to_pandas(nfl.load_draft_picks())


def load_combine() -> pd.DataFrame:
    return _to_pandas(nfl.load_combine())


def load_ff_playerids() -> pd.DataFrame:
    return _to_pandas(nfl.load_ff_playerids())


def load_rosters(seasons: Iterable[int]) -> pd.DataFrame:
    return _safe_load_seasons(lambda s: nfl.load_rosters(seasons=s), list(seasons), "load_rosters")


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
