from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Query
from sqlalchemy.orm import Session

from db.models import Player
from db.session import SessionLocal


router = APIRouter()


@router.get("/players")
def list_players(position: Optional[str] = Query(default=None), team: Optional[str] = Query(default=None)) -> List[dict]:
    session: Session = SessionLocal()
    try:
        q = session.query(Player)
        if position:
            q = q.filter(Player.position == position)
        if team:
            q = q.filter(Player.team_current == team)
        return [
            {
                "player_id": p.player_id,
                "name": p.name,
                "position": p.position,
                "team_current": p.team_current,
            }
            for p in q.limit(500).all()
        ]
    finally:
        session.close()


