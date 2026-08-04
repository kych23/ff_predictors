"""Publish the serving subset from the research DB to the serving DB.

The research DB (local Postgres) holds the full raw/feature/label/backtest
history. The serving DB (Supabase) only needs what the API and live draft/weekly
serving read: players, adp, projections, weekly_projections (all small), the
current-season weekly_features (the 165 MB table, filtered so serving stays
tiny), and the latest ingest_snapshots row for provenance.

Copy is scope-wise idempotent: for each table the target scope is deleted then
re-inserted, so re-publishing is a no-op-equivalent refresh. Runs after every
pipeline run (scripts/run_pipeline.sh).
"""
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import delete, insert, inspect, select

from src.db.models import (
    Adp,
    IngestSnapshot,
    Player,
    Projection,
    WeeklyFeature,
    WeeklyProjection,
)
from src.db.session import serving_session_scope, session_scope

logger = logging.getLogger(__name__)


def _has_table(session, name: str) -> bool:
    # inspect the session's OWN connection, not a fresh engine.connect(): on
    # SQLite :memory: a new connection is a different empty database.
    return inspect(session.connection()).has_table(name)


def _copy(research, serving, model, *, where=None) -> int:
    """Copy rows of ``model`` (optionally filtered by ``where``) research->serving.

    Deletes the same scope on serving first so the copy is a clean refresh.
    Missing source/target tables (e.g. JSONB tables absent on a SQLite test DB)
    are skipped, not errored.
    """
    tbl = model.__table__
    if not _has_table(research, tbl.name) or not _has_table(serving, tbl.name):
        return 0
    sel = select(tbl)
    dele = delete(tbl)
    if where is not None:
        sel = sel.where(where)
        dele = dele.where(where)
    rows = [dict(r._mapping) for r in research.execute(sel)]
    serving.execute(dele)
    if rows:
        serving.execute(insert(tbl), rows)
    return len(rows)


def publish_serving_core(research, serving, *,
                         weekly_feature_season: Optional[int] = None) -> dict:
    """Copy the serving subset using two open sessions. Returns per-table counts.

    Small serving tables (players, adp, projections, weekly_projections) are
    copied in full. weekly_features is filtered to ``weekly_feature_season``
    (default: the max season present) so the big table stays small on serving.
    Only the latest ingest_snapshots row is copied (provenance for the API).
    """
    counts: dict = {}
    counts["players"] = _copy(research, serving, Player)
    counts["adp"] = _copy(research, serving, Adp)
    counts["projections"] = _copy(research, serving, Projection)
    counts["weekly_projections"] = _copy(research, serving, WeeklyProjection)

    if _has_table(research, "weekly_features"):
        season = weekly_feature_season
        if season is None:
            season = research.execute(
                select(WeeklyFeature.season)
                .order_by(WeeklyFeature.season.desc()).limit(1)
            ).scalar()
        if season is not None:
            counts["weekly_features"] = _copy(
                research, serving, WeeklyFeature,
                where=WeeklyFeature.season == season)
        else:
            counts["weekly_features"] = 0

    # latest ingest snapshot only (full history stays in research)
    if _has_table(research, "ingest_snapshots") and _has_table(serving, "ingest_snapshots"):
        latest = research.execute(
            select(IngestSnapshot).order_by(IngestSnapshot.extracted_at.desc()).limit(1)
        ).scalars().first()
        if latest is not None:
            row = {c.name: getattr(latest, c.name) for c in IngestSnapshot.__table__.columns}
            serving.execute(delete(IngestSnapshot).where(
                IngestSnapshot.snapshot_id == latest.snapshot_id))
            serving.execute(insert(IngestSnapshot.__table__), [row])
            counts["ingest_snapshots"] = 1

    return counts


def publish_serving(*, weekly_feature_season: Optional[int] = None) -> dict:
    """Open research + serving sessions and copy the serving subset across."""
    with session_scope("research") as research, serving_session_scope() as serving:
        counts = publish_serving_core(
            research, serving, weekly_feature_season=weekly_feature_season)
        serving.commit()
    logger.info("published serving subset: %s", counts)
    return counts
