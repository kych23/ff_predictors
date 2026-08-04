import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional
from dotenv import load_dotenv

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

# Two roles, two engines (DrafterSpec storage split):
#   "serving"  -> DATABASE_URL           (Supabase; the API + live serving read this)
#   "research" -> RESEARCH_DATABASE_URL  (local Postgres; pipeline/training/benchmark)
# RESEARCH_DATABASE_URL falls back to DATABASE_URL when unset, so a single-DB
# dev/test setup (and the whole pytest suite) works unchanged.
SERVING = "serving"
RESEARCH = "research"


def _resolve_url(role: str = RESEARCH) -> str:
    serving = os.getenv("DATABASE_URL")
    if role == SERVING:
        if not serving:
            raise ValueError("DATABASE_URL environment variable is not set")
        return serving
    research = os.getenv("RESEARCH_DATABASE_URL") or serving
    if not research:
        raise ValueError(
            "neither RESEARCH_DATABASE_URL nor DATABASE_URL is set")
    return research


def _get_database_url() -> str:
    """Back-compat shim: the serving URL (old single-DB callers)."""
    return _resolve_url(SERVING)


_engines: Dict[str, object] = {}
_factories: Dict[str, sessionmaker] = {}


def get_engine(role: str = RESEARCH):
    """Cached engine per role; created on first use so import never needs a DB."""
    if role not in _engines:
        eng = create_engine(_resolve_url(role), pool_pre_ping=True, future=True)
        _engines[role] = eng
        _factory_for(role).configure(bind=eng)
    return _engines[role]


def _factory_for(role: str) -> sessionmaker:
    if role not in _factories:
        _factories[role] = sessionmaker(autoflush=False, autocommit=False,
                                        class_=Session, future=True)
    return _factories[role]


class _LazyServingFactory(sessionmaker):
    """The API/serving session factory; binds the serving engine on first use."""

    def __call__(self, **kwargs):
        get_engine(SERVING)
        return _factory_for(SERVING)(**kwargs)


# SessionLocal is the serving factory (api/deps get_db, board, start read serving).
SessionLocal = _LazyServingFactory(autoflush=False, autocommit=False,
                                   class_=Session, future=True)


@contextmanager
def session_scope(role: str = RESEARCH) -> Iterator[Session]:
    """Session with rollback-on-error and guaranteed close.

    Defaults to the RESEARCH engine — the pipeline/training/benchmark bulk of
    call sites — so those need no change. Serving readers pass role="serving"
    or use ``serving_session_scope``. Read-only callers just use the session;
    writers call ``session.commit()`` themselves.
    """
    get_engine(role)
    session = _factory_for(role)()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def serving_session_scope() -> Iterator[Session]:
    """Serving-engine scope (Supabase): API + live draft/weekly reads."""
    with session_scope(SERVING) as s:
        yield s


def latest_snapshot_id() -> Optional[str]:
    """Most recent ingest snapshot_id (research DB), or None if empty."""
    from src.db.models import IngestSnapshot
    with session_scope(RESEARCH) as session:
        row = session.execute(
            select(IngestSnapshot).order_by(IngestSnapshot.extracted_at.desc())
        ).scalars().first()
        return row.snapshot_id if row else None


def resolve_snapshot(snapshot_id: Optional[str] = None) -> str:
    """Resolve an explicit snapshot id, else the latest (research); print it.

    Shared by the pipeline scripts so the resolve-or-fail boilerplate lives in
    one place. Raises SystemExit if the research DB has no snapshot yet.
    """
    sid = snapshot_id or latest_snapshot_id()
    if not sid:
        raise SystemExit("No snapshot found — run seed_db.py first.")
    print(f"Using snapshot: {sid}")
    return sid
