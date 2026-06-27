"""Idempotent upserts for the season-grain schema (DrafterSpec.md §4.3).

Keeps the proven ``_chunk_iter`` + ``on_conflict_do_update`` pattern from the old
weekly system; one upsert per table. The old precomputed-``fantasy_points_ppr``
label path is **removed** here (M0 owns that transition, §4.4.1): ``weekly_stats_raw``
stores only raw component stats and labels are computed in M1 from the scoring config.
"""
from __future__ import annotations

import math
from typing import Iterable

import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from .models import (
    Adp,
    AdpCoverage,
    IngestSnapshot,
    Player,
    PlayerIdMap,
    Projection,
    SeasonFeature,
    SeasonLabel,
    WeeklyStatsRaw,
)


def _chunk_iter(df: pd.DataFrame, size: int = 5_000) -> Iterable[pd.DataFrame]:
    for start in range(0, len(df), size):
        yield df.iloc[start : start + size]


def _clean(v: object) -> object:
    """Replace NaN/inf/pd.NA with None so psycopg2 never sends them to integer columns."""
    if v is None:
        return None
    if v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _upsert(df: pd.DataFrame, model, index_elements, update_cols, session: Session) -> int:
    """Generic chunked upsert: insert rows, update ``update_cols`` on conflict."""
    if df is None or df.empty:
        return 0
    total = 0
    for chunk in _chunk_iter(df):
        records = [{k: _clean(v) for k, v in row.items()}
                   for row in chunk.to_dict(orient="records")]
        stmt = insert(model).values(records)
        if update_cols:
            stmt = stmt.on_conflict_do_update(
                index_elements=index_elements,
                set_={c: getattr(stmt.excluded, c) for c in update_cols},
            )
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=index_elements)
        session.execute(stmt)
        total += len(chunk)
    return total


def upsert_players(df: pd.DataFrame, session: Session) -> int:
    cols = [
        "player_id", "name", "position", "college", "rookie_year",
        "draft_year", "draft_round", "draft_pick", "draft_team",
        "birth_date", "latest_team",
    ]
    df = df.reindex(columns=cols)
    df = df.dropna(subset=["player_id", "name"])
    return _upsert(df, Player, [Player.player_id], cols[1:], session)


def upsert_player_id_map(df: pd.DataFrame, session: Session) -> int:
    cols = [
        "gsis_id", "fantasypros_id", "sleeper_id", "espn_id", "pfr_id",
        "cfb_id", "merge_name", "years_exp",
    ]
    df = df.reindex(columns=cols).dropna(subset=["gsis_id"])
    df = df.drop_duplicates(subset=["gsis_id"])
    return _upsert(df, PlayerIdMap, [PlayerIdMap.gsis_id], cols[1:], session)


def upsert_weekly_stats_raw(df: pd.DataFrame, session: Session) -> int:
    cols = ["player_id", "season", "week", "season_type", "team", "position", "stats",
            "snapshot_id"]
    df = df.reindex(columns=cols).dropna(subset=["player_id", "season", "week"])
    idx = [WeeklyStatsRaw.player_id, WeeklyStatsRaw.season, WeeklyStatsRaw.week,
           WeeklyStatsRaw.season_type]
    return _upsert(df, WeeklyStatsRaw, idx, ["team", "position", "stats", "snapshot_id"], session)


def upsert_season_labels(df: pd.DataFrame, session: Session) -> int:
    cols = ["player_id", "season", "games_played", "fantasy_points_total", "fppg",
            "snapshot_id"]
    df = df.reindex(columns=cols).dropna(subset=["player_id", "season"])
    idx = [SeasonLabel.player_id, SeasonLabel.season]
    return _upsert(df, SeasonLabel, idx, cols[2:], session)


def upsert_season_features(df: pd.DataFrame, session: Session) -> int:
    cols = ["player_id", "season", "position", "is_rookie", "as_of_date", "features",
            "snapshot_id"]
    df = df.reindex(columns=cols).dropna(subset=["player_id", "season"])
    idx = [SeasonFeature.player_id, SeasonFeature.season]
    return _upsert(df, SeasonFeature, idx, cols[2:], session)


def upsert_projections(df: pd.DataFrame, session: Session) -> int:
    cols = ["player_id", "season", "model_version", "position", "p10", "p50", "p90",
            "pos_rank", "snapshot_id"]
    df = df.reindex(columns=cols).dropna(subset=["player_id", "season", "model_version"])
    idx = [Projection.player_id, Projection.season, Projection.model_version]
    return _upsert(df, Projection, idx, cols[3:], session)


def upsert_adp(df: pd.DataFrame, session: Session) -> int:
    cols = ["source", "season", "format", "teams", "player_id", "name", "position",
            "adp", "adp_stdev", "n_drafts", "snapshot_id"]
    df = df.reindex(columns=cols).dropna(subset=["source", "season", "format", "teams", "player_id"])
    idx = [Adp.source, Adp.season, Adp.format, Adp.teams, Adp.player_id]
    return _upsert(df, Adp, idx, cols[5:], session)


def upsert_adp_coverage(df: pd.DataFrame, session: Session) -> int:
    cols = ["season", "format", "teams", "n_players", "ranking_eligible",
            "sim_eligible", "snapshot_id"]
    df = df.reindex(columns=cols)
    idx = [AdpCoverage.season, AdpCoverage.format, AdpCoverage.teams]
    return _upsert(df, AdpCoverage, idx, cols[3:], session)


def upsert_ingest_snapshot(row: dict, session: Session) -> int:
    stmt = insert(IngestSnapshot).values([row])
    stmt = stmt.on_conflict_do_update(
        index_elements=[IngestSnapshot.snapshot_id],
        set_={c: getattr(stmt.excluded, c) for c in
              ("extracted_at", "nflreadpy_version", "notes", "coverage")},
    )
    session.execute(stmt)
    return 1
