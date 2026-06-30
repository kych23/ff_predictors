"""Per-game season labels from raw weekly box scores

Reads staged weekly_stats_raw (raw components only), scores each week through
``scoring.py``, aggregates to per-(player, season): games_played, fantasy_points_total, fppg. Applies the min_games_train floor (default 8) as a LABEL-QUALITY filter only — it controls per-game-rate sample noise, it does NOT define replacement level.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from sqlalchemy import select

from src.config import load_config
from src.db.models import WeeklyStatsRaw
from src.db.session import SessionLocal
from src.db.upsert_data import upsert_season_labels, upsert_weekly_labels

from .scoring import score_dataframe

logger = logging.getLogger(__name__)


def _load_weekly_raw(session, snapshot_id: Optional[str]) -> pd.DataFrame:
    stmt = select(WeeklyStatsRaw)
    if snapshot_id:
        stmt = stmt.where(WeeklyStatsRaw.snapshot_id == snapshot_id)
    rows = session.execute(stmt).scalars().all()
    if not rows:
        return pd.DataFrame()
    recs = []
    for r in rows:
        rec = {"player_id": r.player_id, "season": r.season, "week": r.week}
        rec.update(r.stats or {})
        rec["position"] = r.position
        rec["team"] = r.team
        rec["season_type"] = r.season_type
        recs.append(rec)
    return pd.DataFrame(recs)


def compute_labels(weekly: pd.DataFrame, min_games: int, snapshot_id: str) -> pd.DataFrame:
    """Aggregate scored weekly rows to season labels with the min-games floor."""
    if weekly.empty:
        return weekly
    weekly = weekly.copy()
    weekly["week_points"] = score_dataframe(weekly)
    agg = (
        weekly.groupby(["player_id", "season"])
        .agg(games_played=("week", "nunique"),
             fantasy_points_total=("week_points", "sum"))
        .reset_index()
    )
    agg["fppg"] = agg["fantasy_points_total"] / agg["games_played"].clip(lower=1)
    # Label-quality floor: drop low-sample seasons from the TRAINING label set.
    agg = agg[agg["games_played"] >= min_games].copy()
    agg["snapshot_id"] = snapshot_id
    return agg


def build_labels(snapshot_id: Optional[str] = None, config_path: Optional[str] = None) -> int:
    cfg = load_config(config_path)
    session = SessionLocal()
    try:
        weekly = _load_weekly_raw(session, snapshot_id)
        if weekly.empty:
            logger.warning("no weekly_stats_raw rows found (snapshot_id=%s)", snapshot_id)
            return 0
        snap = snapshot_id or "latest"
        labels = compute_labels(weekly, cfg.training.min_games_train, snap)
        n = upsert_season_labels(labels, session)
        session.commit()
        logger.info("season_labels upserted: %d (min_games=%d)", n, cfg.training.min_games_train)
        return n
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def compute_weekly_labels(weekly: pd.DataFrame, snapshot_id: str) -> pd.DataFrame:
    """Score each week individually. REG games only. No min_games filter."""
    if weekly.empty:
        return pd.DataFrame()
    weekly = weekly.copy()
    if "season_type" in weekly.columns:
        weekly = weekly[weekly["season_type"] == "REG"].copy()
    weekly["fantasy_points"] = score_dataframe(weekly)
    return weekly[["player_id", "season", "week", "position", "team",
                   "fantasy_points"]].assign(snapshot_id=snapshot_id)


def build_weekly_labels(snapshot_id: Optional[str] = None) -> int:
    """Load WeeklyStatsRaw, compute weekly labels, upsert."""
    session = SessionLocal()
    try:
        weekly = _load_weekly_raw(session, snapshot_id)
        if weekly.empty:
            logger.warning("no weekly_stats_raw rows for weekly labels (snapshot_id=%s)", snapshot_id)
            return 0
        snap = snapshot_id or "latest"
        labels = compute_weekly_labels(weekly, snap)
        n = upsert_weekly_labels(labels, session)
        session.commit()
        logger.info("weekly_labels upserted: %d rows", n)
        return n
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
