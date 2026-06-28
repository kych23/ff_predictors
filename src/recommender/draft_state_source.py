"""Pluggable live draft-state reader (DrafterSpec.md §4.8.1).

The league drafts on YAHOO, which has no free no-auth live feed. v1 primary mode =
**fast manual CLI entry**: the board is pre-loaded, each pick entered by fuzzy name
match, and an ADP auto-advance shortcut consumes the expected top-ADP available
players up to the next relevant pick, so the user types only DEVIATIONS from ADP.

Also ships a **Sleeper live-poll** mode for reuse/testing. Draft state is
availability, never a projection feature — not a wall violation. (v1.5: a Yahoo
Fantasy API OAuth2 live adapter.)
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from src.ingest.http import ResilientClient
from src.ingest.player_ids import normalize_name


def match_player_candidates(name: str, board: pd.DataFrame, *, cutoff: float = 0.6) -> List[str]:
    """All board player_ids matching a typed name — never silently collapsed.

    Returns every exact normalized-name match (so two distinct same-name players both
    surface and the caller MUST disambiguate, e.g. by position/team), or the players
    behind the single best fuzzy name if there's no exact hit. ``board``: player_id,
    name (+ optional position/team for disambiguation).
    """
    if board.empty:
        return []
    target = normalize_name(name)
    names = board["name"].map(normalize_name)
    exact = board[names == target]
    if len(exact) >= 1:
        return list(exact["player_id"])
    close = difflib.get_close_matches(target, list(names.unique()), n=1, cutoff=cutoff)
    if not close:
        return []
    return list(board[names == close[0]]["player_id"])


def adp_auto_advance(board: pd.DataFrame, drafted: set, n_picks: int) -> List[str]:
    """Return the next ``n_picks`` expected players by ADP (fallback: P50) that are still available.

    Used to fast-forward opponent picks that follow ADP, so the user enters only
    deviations. When ADP is exhausted (late rounds), falls back to P50 projection
    ordering as the best available proxy for what opponents will take.
    """
    avail = board[~board["player_id"].isin(drafted)].copy()
    has_adp = avail.dropna(subset=["adp"])
    if not has_adp.empty:
        avail = has_adp.sort_values("adp")
    elif "p50" in avail.columns:
        avail = avail.dropna(subset=["p50"]).sort_values("p50", ascending=False)
    else:
        return []
    return list(avail.head(n_picks)["player_id"])


class DraftStateSource:
    """Interface: maintains the set of drafted player_ids."""

    def __init__(self, board: pd.DataFrame):
        self.board = board
        self.drafted: set = set()

    def record(self, player_id: str) -> None:
        self.drafted.add(player_id)

    def picks(self) -> List[str]:
        raise NotImplementedError


@dataclass
class ManualDraftSource(DraftStateSource):
    """Manual entry source — driven by scripts/draft.py prompts."""

    board: pd.DataFrame
    drafted: set = field(default_factory=set)

    def candidates(self, name: str) -> List[str]:
        """All player_ids a typed name could mean (>1 = ambiguous, must disambiguate)."""
        return match_player_candidates(name, self.board)

    def picks(self) -> List[str]:
        return list(self.drafted)


class SleeperDraftSource(DraftStateSource):
    """Sleeper live-poll source (for reuse/testing; not this user's path)."""

    def __init__(self, board: pd.DataFrame, draft_id: str,
                 sleeper_to_gsis: Optional[Dict[str, str]] = None):
        super().__init__(board)
        self.draft_id = draft_id
        self.sleeper_to_gsis = sleeper_to_gsis or {}
        self._client = ResilientClient("sleeper", rate_limit_per_min=120)

    def picks(self) -> List[str]:
        url = f"https://api.sleeper.app/v1/draft/{self.draft_id}/picks"
        data = self._client.get_json(url)
        out = []
        for p in data or []:
            sid = str(p.get("player_id"))
            gsis = self.sleeper_to_gsis.get(sid, sid)
            out.append(gsis)
        self.drafted = set(out)
        return out
