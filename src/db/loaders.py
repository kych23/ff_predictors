"""Shared ORM -> DataFrame loaders for CLI scripts and benchmarks.

Each script used to hand-roll the same 10-line select/list-comprehension blocks;
they live here once. All loaders take an open session (caller owns lifecycle —
use ``session_scope()`` from ``src.db.session``).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from sqlalchemy import select

from src.db.models import (
    Adp,
    Player,
    Projection,
    SeasonLabel,
    WeeklyLabel,
    WeeklyProjection,
)


def load_projections_df(session, season: Optional[int] = None) -> pd.DataFrame:
    stmt = select(Projection)
    if season is not None:
        stmt = stmt.where(Projection.season == season)
    cols = ["player_id", "season", "position", "p10", "p50", "p90"]
    return pd.DataFrame([
        {"player_id": r.player_id, "season": r.season, "position": r.position,
         "p10": r.p10, "p50": r.p50, "p90": r.p90}
        for r in session.execute(stmt).scalars()
    ], columns=cols)


def load_adp_df(session, cfg, season: Optional[int] = None) -> pd.DataFrame:
    stmt = select(Adp).where(Adp.format == cfg.adp.format,
                             Adp.teams == cfg.adp.teams)
    if season is not None:
        stmt = stmt.where(Adp.season == season)
    cols = ["player_id", "season", "position", "adp", "adp_stdev"]
    return pd.DataFrame([
        {"player_id": r.player_id, "season": r.season, "position": r.position,
         "adp": r.adp, "adp_stdev": r.adp_stdev}
        for r in session.execute(stmt).scalars()
    ], columns=cols)


def load_player_names(session) -> dict:
    """{player_id: name} for display."""
    return {r.player_id: r.name for r in session.execute(select(Player)).scalars()}


def load_players_df(session) -> pd.DataFrame:
    return pd.DataFrame([
        {"player_id": r.player_id, "name": r.name, "team": r.latest_team}
        for r in session.execute(select(Player)).scalars()
    ], columns=["player_id", "name", "team"])


def load_season_labels_df(session, season: Optional[int] = None) -> pd.DataFrame:
    stmt = select(SeasonLabel)
    if season is not None:
        stmt = stmt.where(SeasonLabel.season == season)
    return pd.DataFrame([
        {"player_id": r.player_id, "season": r.season, "fppg": r.fppg}
        for r in session.execute(stmt).scalars()
    ], columns=["player_id", "season", "fppg"])


def load_weekly_projections_df(session) -> pd.DataFrame:
    cols = ["player_id", "season", "week", "position", "p10", "p50", "p90"]
    return pd.DataFrame([
        {"player_id": r.player_id, "season": r.season, "week": r.week,
         "position": r.position, "p10": r.p10, "p50": r.p50, "p90": r.p90}
        for r in session.execute(select(WeeklyProjection)).scalars()
    ], columns=cols)


def load_weekly_labels_df(session) -> pd.DataFrame:
    cols = ["player_id", "season", "week", "position", "team", "fantasy_points"]
    return pd.DataFrame([
        {"player_id": r.player_id, "season": r.season, "week": r.week,
         "position": r.position, "team": r.team, "fantasy_points": r.fantasy_points}
        for r in session.execute(select(WeeklyLabel)).scalars()
    ], columns=cols)
