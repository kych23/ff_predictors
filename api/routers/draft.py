from __future__ import annotations

from fastapi import APIRouter, Depends

from api.deps import get_service, get_snapshot_id
from api.draft_service import DraftService
from api.schemas import PickIn, RecommendationOut, SessionCreate, StateOut

router = APIRouter(prefix="/draft/sessions")


@router.post("", response_model=StateOut, status_code=201)
def create_session(body: SessionCreate, svc: DraftService = Depends(get_service),
                   snapshot_id=Depends(get_snapshot_id)):
    sess = svc.create_session(season=body.season, draft_position=body.draft_position,
                              snapshot_id=snapshot_id)
    return svc.state(sess.session_id)


@router.get("/{session_id}", response_model=StateOut)
def get_state(session_id: str, svc: DraftService = Depends(get_service)):
    return svc.state(session_id)


@router.post("/{session_id}/picks", response_model=StateOut)
def record_pick(session_id: str, body: PickIn,
                svc: DraftService = Depends(get_service)):
    return svc.record_pick(session_id, player_id=body.player_id,
                           skip=body.skip, mine=body.mine)


@router.post("/{session_id}/undo", response_model=StateOut)
def undo(session_id: str, svc: DraftService = Depends(get_service)):
    return svc.undo(session_id)


@router.post("/{session_id}/bot-pick", response_model=StateOut)
def bot_pick(session_id: str, svc: DraftService = Depends(get_service)):
    return svc.bot_pick(session_id)


@router.get("/{session_id}/recommendations", response_model=list[RecommendationOut])
def recommendations(session_id: str, top_n: int = 10,
                    svc: DraftService = Depends(get_service)):
    return svc.recommendations(session_id, top_n=top_n)
