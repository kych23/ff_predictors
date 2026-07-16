"""Draft board assembly: projections + ADP + names + bye weeks.

Shared by the draft CLI (scripts/draft.py) and the draft API (api/) so the two
can never disagree about what a board row looks like. Uses ADP only on the
recommender side of the wall (survival/timing), per DrafterSpec §4.0.
"""
from __future__ import annotations

import sys

import pandas as pd

from src.config import LeagueConfig
from src.db.loaders import load_adp_df, load_players_df, load_projections_df
from src.db.session import session_scope

#: Columns recommend() consumes, in preference order; filter to what the board has.
PROJ_COLS = ["player_id", "position", "p10", "p50", "p90",
             "adp", "adp_stdev", "team", "bye_week"]


def compute_bye_weeks(season: int) -> dict:
    """Return {team_abbr: bye_week_number} from nflreadpy schedules. Empty dict on failure."""
    try:
        from src.ingest.sources import load_schedules
        sched = load_schedules([season])
        if sched.empty:
            return {}
        if "game_type" in sched.columns:
            sched = sched[sched["game_type"] == "REG"]
        all_weeks = set(pd.to_numeric(sched["week"], errors="coerce").dropna().astype(int))
        teams = set(sched["home_team"].dropna()) | set(sched["away_team"].dropna())
        bye: dict = {}
        for team in teams:
            played = (
                set(pd.to_numeric(sched.loc[sched["home_team"] == team, "week"],
                                  errors="coerce").dropna().astype(int))
                | set(pd.to_numeric(sched.loc[sched["away_team"] == team, "week"],
                                    errors="coerce").dropna().astype(int))
            )
            missing = sorted(all_weeks - played)
            if missing:
                bye[team] = missing[0]
        return bye
    except Exception as exc:
        print(f"warning: bye weeks unavailable ({exc})", file=sys.stderr)
        return {}


def load_board(season: int, cfg: LeagueConfig) -> pd.DataFrame:
    with session_scope() as session:
        proj = load_projections_df(session, season)[
            ["player_id", "position", "p10", "p50", "p90"]]
        adp = load_adp_df(session, cfg, season)[["player_id", "adp", "adp_stdev"]]
        names = load_players_df(session)
    board = proj.merge(adp, on="player_id", how="left").merge(names, on="player_id", how="left")
    bye_map = compute_bye_weeks(season)
    if bye_map:
        board["bye_week"] = board["team"].map(bye_map).where(board["team"].notna())
    return board
