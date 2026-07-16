"""Product-side tables (draft sessions).

Kept out of src/db/models.py — that module owns the research schema — but they
share its Base so one metadata covers both. Product tables use only portable
column types (JSON, not JSONB) so tests can create them on SQLite.

snapshot_id records which ingest snapshot's projections served the draft
(provenance, invariant #3); nullable because a session can be created before
any pipeline run in dev environments.
"""
from __future__ import annotations

import uuid

from sqlalchemy import JSON, Column, DateTime, Integer, String
from sqlalchemy.sql import func

from src.db.models import Base


def _new_session_id() -> str:
    return uuid.uuid4().hex


class DraftSession(Base):
    __tablename__ = "draft_sessions"

    session_id = Column(String, primary_key=True, default=_new_session_id)
    season = Column(Integer, nullable=False)
    draft_position = Column(Integer, nullable=False)
    platform = Column(String, nullable=False, default="manual")
    status = Column(String, nullable=False, default="active")
    snapshot_id = Column(String, nullable=True)
    history = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(),
                        onupdate=func.now())
