"""Shared utilities for feature assembly."""
from __future__ import annotations

import numpy as np
import pandas as pd


def json_safe(v):
    """Convert numpy scalars to Python-native types for JSON serialization."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def schedule_to_team_lines(sched: pd.DataFrame) -> pd.DataFrame:
    """Convert pre-filtered schedule rows to per-team Vegas lines.

    Expects columns: home_team, away_team, total_line, spread_line (all non-null).
    Returns: team, opponent, implied_pts, game_total, spread — one row per team.
    """
    if sched.empty:
        return pd.DataFrame(columns=["team", "opponent", "implied_pts",
                                     "game_total", "spread"])
    home = pd.DataFrame({
        "team": sched["home_team"].astype(str).values,
        "opponent": sched["away_team"].astype(str).values,
        "implied_pts": sched["total_line"].values / 2 + sched["spread_line"].values / 2,
        "game_total": sched["total_line"].values,
        "spread": sched["spread_line"].values,
    })
    away = pd.DataFrame({
        "team": sched["away_team"].astype(str).values,
        "opponent": sched["home_team"].astype(str).values,
        "implied_pts": sched["total_line"].values / 2 - sched["spread_line"].values / 2,
        "game_total": sched["total_line"].values,
        "spread": -sched["spread_line"].values,
    })
    return pd.concat([home, away], ignore_index=True).drop_duplicates("team", keep="first")
