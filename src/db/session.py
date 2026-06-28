import os
from typing import Generator, Optional
from dotenv import load_dotenv

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

load_dotenv(dotenv_path=".env")

def _get_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL environment variable is not set")
    return url


engine = create_engine(_get_database_url(), pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session, future=True)

def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def latest_snapshot_id() -> Optional[str]:
    """Return the most recently created ingest snapshot_id, or None if DB is empty."""
    from src.db.models import IngestSnapshot
    session = SessionLocal()
    try:
        row = session.execute(
            select(IngestSnapshot).order_by(IngestSnapshot.extracted_at.desc())
        ).scalars().first()
        return row.snapshot_id if row else None
    finally:
        session.close()


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


