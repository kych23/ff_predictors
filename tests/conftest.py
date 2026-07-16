"""Shared API-test fixtures: in-memory SQLite bound to the shared Base metadata.

Only product tables (draft_sessions) are created — research tables use JSONB
and are Postgres-only by design.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.db_models import DraftSession
from src.db.models import Base


@pytest.fixture()
def sqlite_engine():
    eng = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(eng, tables=[DraftSession.__table__])
    yield eng
    eng.dispose()


@pytest.fixture()
def db_session(sqlite_engine):
    factory = sessionmaker(bind=sqlite_engine, autoflush=False, future=True)
    session = factory()
    yield session
    session.close()
