import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional
from dotenv import load_dotenv

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

def _get_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL environment variable is not set")
    return url


_engine = None


def get_engine():
    """Create the engine on first use so importing this module never needs a DB."""
    global _engine
    if _engine is None:
        _engine = create_engine(_get_database_url(), pool_pre_ping=True, future=True)
        SessionLocal.configure(bind=_engine)
    return _engine


class _LazySessionFactory(sessionmaker):
    """sessionmaker that binds the engine on first session creation."""

    def __call__(self, **kwargs):
        get_engine()
        return super().__call__(**kwargs)


SessionLocal = _LazySessionFactory(autoflush=False, autocommit=False,
                                   class_=Session, future=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Session with rollback-on-error and guaranteed close.

    Read-only callers just use the session; writers call ``session.commit()``
    themselves (commit stays explicit so partial-write semantics are visible
    at the call site).
    """
    session = SessionLocal()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def latest_snapshot_id() -> Optional[str]:
    """Return the most recently created ingest snapshot_id, or None if DB is empty."""
    from src.db.models import IngestSnapshot
    with session_scope() as session:
        row = session.execute(
            select(IngestSnapshot).order_by(IngestSnapshot.extracted_at.desc())
        ).scalars().first()
        return row.snapshot_id if row else None


def resolve_snapshot(snapshot_id: Optional[str] = None) -> str:
    """Resolve an explicit snapshot id, else fall back to the latest; print it.

    Shared by the pipeline scripts (build_labels/build_features/train_projection/
    run_benchmark) so the resolve-or-fail boilerplate lives in one place. Raises
    SystemExit if the DB has no snapshot yet.
    """
    sid = snapshot_id or latest_snapshot_id()
    if not sid:
        raise SystemExit("No snapshot found — run seed_db.py first.")
    print(f"Using snapshot: {sid}")
    return sid


