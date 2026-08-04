"""What the draft service needs from any pick source (manual entry, Yahoo, ...)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class LeagueSettings:
    teams: int
    rounds: int
    roster_slots: dict


@dataclass
class DraftState:
    picks: list[tuple[Optional[str], bool]]  # (player_id | None-for-skip, mine)


class PlatformAdapter(Protocol):
    def get_league_settings(self) -> LeagueSettings: ...
    def get_draft_state(self) -> DraftState: ...
