from __future__ import annotations

from fastapi import APIRouter, Depends

from api.deps import get_board_for
from api.schemas import PlayerOut

router = APIRouter()


@router.get("/players", response_model=list[PlayerOut])
def list_players(season: int, board_for=Depends(get_board_for)):
    import pandas as pd
    board = board_for(season)
    return board.where(pd.notna(board), None).to_dict(orient="records")
