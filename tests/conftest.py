"""Shared API-test fixtures: in-memory SQLite bound to the shared Base metadata.

Only product tables (draft_sessions) are created — research tables use JSONB
and are Postgres-only by design.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# §9.4: importing api.db_models at COLLECTION time couples the whole suite to
# the frozen api package, so marker deselection alone cannot decouple it. The
# import moves into the fixtures that actually need it.


def _frozen_api_models():
    """Import the frozen api models lazily, skipping if unavailable.

    Returns (Base, DraftSession) so the fixture body needs no module-level
    names from the frozen package.
    """
    pytest.importorskip("api.db_models")
    from api.db_models import DraftSession
    from src.db.models import Base

    return Base, DraftSession


@pytest.fixture()
def sqlite_engine():
    # StaticPool + check_same_thread=False: FastAPI's TestClient (Task 5) dispatches
    # requests through worker threads distinct from the fixture's thread. Plain
    # sqlite:///:memory: defaults to SingletonThreadPool, which hands each thread
    # its own private (table-less) in-memory database and breaks cross-thread use.
    eng = create_engine("sqlite:///:memory:", future=True,
                        connect_args={"check_same_thread": False},
                        poolclass=StaticPool)
    base, draft_session = _frozen_api_models()
    base.metadata.create_all(eng, tables=[draft_session.__table__])
    yield eng
    eng.dispose()


@pytest.fixture()
def db_session(sqlite_engine):
    factory = sessionmaker(bind=sqlite_engine, autoflush=False, future=True)
    session = factory()
    yield session
    session.close()
