from __future__ import annotations

from api.adapters.base import DraftState, LeagueSettings
from src.config.league_config import LeagueConfig


class ManualAdapter:
    """Reads a session's own event-sourced history as a PlatformAdapter."""

    def __init__(self, history: list, cfg: LeagueConfig):
        self._history = history
        self._cfg = cfg

    def get_league_settings(self) -> LeagueSettings:
        return LeagueSettings(
            teams=self._cfg.teams,
            rounds=self._cfg.roster.rounds,
            roster_slots=dict(self._cfg.roster.slots),
        )

    def get_draft_state(self) -> DraftState:
        picks: list[tuple[str | None, bool]] = []
        for command in self._history:
            for ev in command:
                if ev[0] == "skip":
                    picks.append((None, False))
                else:
                    picks.append((ev[1], bool(ev[2])))
        return DraftState(picks=picks)
