from __future__ import annotations

from typing import Iterable

import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from .models import Game, Label, Player, Team


def _chunk_iter(df: pd.DataFrame, size: int = 5_000) -> Iterable[pd.DataFrame]:
    for start in range(0, len(df), size):
        yield df.iloc[start : start + size]


def upsert_players(df: pd.DataFrame, session: Session) -> int:
    if df.empty:
        return 0
    records = df.rename(
        columns={
            "gsis_id": "player_id",
            "display_name": "name",
            "position": "position",
            "latest_team": "team_current",
        }
    )[ ["player_id", "name", "position", "team_current"] ]\
        .copy()

    # Normalize text fields and treat empty strings as missing
    records["name"] = records["name"].astype(str).str.strip()
    records["position"] = records["position"].astype(str).str.strip()
    records.loc[records["name"] == "", "name"] = pd.NA
    records.loc[records["position"] == "", "position"] = pd.NA
    records = records.dropna(subset=["player_id", "name", "position"])  # enforce NOT NULLs

    total = 0
    for chunk in _chunk_iter(records):
        stmt = insert(Player).values(chunk.to_dict(orient="records"))
        stmt = stmt.on_conflict_do_update(
            index_elements=[Player.player_id],
            set_={
                "name": stmt.excluded.name,
                "position": stmt.excluded.position,
                "team_current": stmt.excluded.team_current,
            },
        )
        session.execute(stmt)
        total += len(chunk)
    return total


def upsert_teams(df: pd.DataFrame, session: Session) -> int:
    if df.empty:
        return 0

    abbrs = pd.Series(pd.concat([df["home_team"], df["away_team"]]).dropna().unique(), name="team_name")
    records = pd.DataFrame({
        "team_name": abbrs,
    })

    total = 0
    for chunk in _chunk_iter(records):
        stmt = insert(Team).values(chunk.to_dict(orient="records"))
        stmt = stmt.on_conflict_do_nothing(index_elements=[Team.team_name])
        session.execute(stmt)
        total += len(chunk)
    return total


def upsert_games(df: pd.DataFrame, session: Session) -> int:
    if df.empty:
        return 0
    records = df.copy()
    required = ["game_id", "season", "week", "home_team", "away_team"]
    missing = [c for c in required if c not in records.columns]
    if missing:
        raise ValueError(f"Missing columns for games: {missing}")

    cols = required
    records = records[cols]

    total = 0
    for chunk in _chunk_iter(records):
        stmt = insert(Game).values(chunk.to_dict(orient="records"))
        stmt = stmt.on_conflict_do_update(
            index_elements=[Game.game_id],
            set_={
                "season": stmt.excluded.season,
                "week": stmt.excluded.week,
                "home_team": stmt.excluded.home_team,
                "away_team": stmt.excluded.away_team,
            },
        )
        session.execute(stmt)
        total += len(chunk)
    return total


def upsert_labels(df: pd.DataFrame, session: Session) -> int:
    if df.empty:
        return 0
    
    if "fantasy_points_ppr" in df.columns:
        df = df.assign(fantasy_points=df["fantasy_points_ppr"])
    else:
        raise ValueError("labels dataframe requires 'fantasy_points_ppr'")

    required = ["player_id", "season", "week", "fantasy_points"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for labels: {missing}")

    records = df[required].copy()
    records["player_id"] = records["player_id"].astype(str).str.strip()
    records.loc[records["player_id"] == "", "player_id"] = pd.NA
    records["season"] = pd.to_numeric(records["season"], errors="coerce").astype("Int64")
    records["week"] = pd.to_numeric(records["week"], errors="coerce").astype("Int64")
    records["fantasy_points"] = pd.to_numeric(records["fantasy_points"], errors="coerce")
    records = records.dropna(subset=["player_id", "season", "week", "fantasy_points"])  

    total = 0
    for chunk in _chunk_iter(records):
        stmt = insert(Label).values(chunk.to_dict(orient="records"))
        stmt = stmt.on_conflict_do_update(
            constraint="labels_pkey",
            set_={"fantasy_points": stmt.excluded.fantasy_points},
        )
        session.execute(stmt)
        total += len(chunk)
    return total


