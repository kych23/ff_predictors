"""FastAPI dependencies — every external resource enters through one of these
so tests can override them individually."""
from __future__ import annotations

from typing import Callable, Iterator, Optional

import pandas as pd
from fastapi import Depends

from api.draft_service import DraftService, get_cached_board
from src.config import LeagueConfig, load_config
from src.db.session import SessionLocal


def get_db() -> Iterator:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_cfg() -> LeagueConfig:
    return load_config()


def get_board_for() -> Callable[[int], pd.DataFrame]:
    return get_cached_board


def get_snapshot_id() -> Optional[str]:
    """Best-effort provenance: which ingest snapshot's projections serve this draft."""
    try:
        from src.db.session import latest_snapshot_id
        return latest_snapshot_id()
    except Exception:
        return None


def get_service(db=Depends(get_db), cfg: LeagueConfig = Depends(get_cfg),
                board_for=Depends(get_board_for)) -> DraftService:
    return DraftService(db=db, cfg=cfg, board_for=board_for)
