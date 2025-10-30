from __future__ import annotations

from typing import List

from fastapi import APIRouter
from sqlalchemy import distinct
from sqlalchemy.orm import Session

from db.models import Game
from db.session import SessionLocal


router = APIRouter()


@router.get("/weeks")
def list_weeks(season: int) -> List[int]:
    session: Session = SessionLocal()
    try:
        rows = session.query(distinct(Game.week)).filter(Game.season == season).order_by(Game.week).all()
        return [w[0] for w in rows]
    finally:
        session.close()


@router.get("/opponents")
def list_opponents(season: int, week: int) -> List[str]:
    session: Session = SessionLocal()
    try:
        rows = session.query(distinct(Game.home_team)).filter(Game.season == season, Game.week == week).all()
        rows_away = session.query(distinct(Game.away_team)).filter(Game.season == season, Game.week == week).all()
        teams = {r[0] for r in rows} | {r[0] for r in rows_away}
        return sorted(teams)
    finally:
        session.close()


