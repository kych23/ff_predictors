"""Pydantic wire schemas. Field sets mirror the dicts DraftService returns —
schemas validate the contract, the service stays framework-free."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class PlayerOut(BaseModel):
    player_id: str
    name: Optional[str] = None
    team: Optional[str] = None
    position: str
    p10: float
    p50: float
    p90: float
    adp: Optional[float] = None
    bye_week: Optional[int] = None


class SessionCreate(BaseModel):
    season: int
    draft_position: int


class PickIn(BaseModel):
    player_id: Optional[str] = None
    skip: bool = False
    mine: Optional[bool] = None


class PickOut(BaseModel):
    pick_number: int
    player_id: Optional[str]
    name: Optional[str]
    mine: bool
    skipped: bool


class RosterEntryOut(BaseModel):
    player_id: str
    name: Optional[str] = None
    position: Optional[str] = None
    team: Optional[str] = None
    bye_week: Optional[int] = None


class StateOut(BaseModel):
    session_id: str
    season: int
    draft_position: int
    platform: str
    status: str
    teams: int
    rounds: int
    my_picks: list[int]
    current_overall_pick: int
    is_my_turn: bool
    next_my_pick: Optional[int]
    remaining_picks: int
    picks: list[PickOut]
    my_roster: list[RosterEntryOut]
    open_starters: dict[str, int]


class RecommendationOut(BaseModel):
    player_id: str
    name: Optional[str]
    position: str
    team: Optional[str] = None
    vona_score: float
    value: float
    p10: float
    p50: float
    p90: float
    adp: Optional[float] = None
    draft_round: int
    target_quantile: float
    forced_completion: bool
