"""Market team scoring-context feature (DrafterSpec.md §4.5).

A FORWARD-LOOKING, market-based view of the offense each player sits in for season Y,
derived from the **Week-1 betting line** (nflverse schedules ``total_line`` /
``spread_line``). Week-1 lines are posted before kickoff, so they are a preseason-safe
artifact (like the Week-1 depth chart) — and unlike lagged production they price the
offseason roster/coaching changes the market has already absorbed.

For a team's Week-1 game the implied team total is ``total_line/2 ± spread_line/2``
(``spread_line`` is home-favored-positive in nflverse). Per team, season Y:
  * ``team_ctx_implied_pts`` — market-implied points for THIS team (offense strength)
  * ``team_ctx_game_total``  — the game's over/under (scoring environment / pace)
  * ``team_ctx_spread``      — points favored (+) / underdog (-) for THIS team

This is a deliberate MARKET signal inside the projection features. It does NOT touch
the ADP wall (that quarantines draft-market data specifically); it is game-line data.
Uses ``total_line``/``spread_line`` ONLY — never ``total`` (the actual final score, a
result that would leak).


PORTED from src/features/team_context.py (DraftEngineDesign.md §9.2).
Edits: v2 config API (load_league), scoring moves to domain, and the
leakage guard is reached through platform.asof — the only platform
subpackage this layer may import (§9.0, §11.3).
"""
from __future__ import annotations

import pandas as pd

from ._utils import schedule_to_team_lines

CTX_COLS = ["team_ctx_implied_pts", "team_ctx_game_total", "team_ctx_spread"]

_RENAME = {"implied_pts": "team_ctx_implied_pts",
           "game_total": "team_ctx_game_total",
           "spread": "team_ctx_spread"}


def build_team_context(schedules_y: pd.DataFrame) -> pd.DataFrame:
    """Per-team Week-1 market scoring context for one season. One row per team.

    ``schedules_y`` must be a SINGLE season's schedule rows (already filtered to season
    Y by the caller). Only Week-1 REG games are used, keeping it unambiguously
    preseason (later-week lines move on in-season results -> would leak).
    """
    empty = pd.DataFrame(columns=["team"] + CTX_COLS)
    if schedules_y is None or schedules_y.empty:
        return empty
    s = schedules_y.copy()
    if "game_type" in s.columns:
        s = s[s["game_type"] == "REG"]
    s["_wk"] = pd.to_numeric(s.get("week"), errors="coerce")
    s = s[s["_wk"] == 1]
    for c in ("total_line", "spread_line"):
        if c not in s.columns:
            return empty
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s.dropna(subset=["total_line", "spread_line", "home_team", "away_team"])
    if s.empty:
        return empty

    out = schedule_to_team_lines(s).rename(columns=_RENAME)
    return out[["team"] + CTX_COLS].reset_index(drop=True)
