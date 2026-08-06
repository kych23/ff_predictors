"""Roster + snake-draft layout tracking.

Tracks my roster, open starter slots, flex eligibility, and the snake pick layout
(my next overall pick from n_teams + draft_position). Draft state is availability,
never a projection feature — not a wall violation.

Snake: my overall pick in round r (1-indexed) is
  (r-1)*n_teams + p              for odd r,
  (r-1)*n_teams + (n_teams-p+1)  for even r (the snake turn).


PORTED from src/recommender/roster_state.py (DraftEngineDesign.md §9.2).
Tier-2 fallback path (§7.3). Behavior unchanged from v1; the only edits
are the v2 config API (load_league, flex_eligibility) and the DEF -> DST
position rename (§10.7).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.core.config.league import LeagueConfig, load_league


def overall_pick(draft_round: int, draft_position: int, n_teams: int) -> int:
    """Overall pick number for my seat at a given round (1-indexed)."""
    if draft_round % 2 == 1:
        return (draft_round - 1) * n_teams + draft_position
    return (draft_round - 1) * n_teams + (n_teams - draft_position + 1)


def my_pick_sequence(draft_position: int, n_teams: int, rounds: int) -> List[int]:
    return [overall_pick(r, draft_position, n_teams) for r in range(1, rounds + 1)]


@dataclass
class RosterState:
    cfg: LeagueConfig
    draft_position: int
    my_roster: List[dict] = field(default_factory=list)   # {player_id, position}
    drafted: set = field(default_factory=set)             # all drafted player_ids (any team)
    slot_fill: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.slot_fill:
            self.slot_fill = {s: 0 for s in self.cfg.roster.slots}

    # --- snake layout ---
    @property
    def my_picks(self) -> List[int]:
        return my_pick_sequence(self.draft_position, self.cfg.teams, self.cfg.roster.rounds)

    def current_overall_pick(self) -> int:
        """1-indexed overall pick about to be made (total drafted + 1)."""
        return len(self.drafted) + 1

    def my_current_round(self) -> int:
        """Current draft round based on overall pick position (picks 1-N = round 1, etc.)."""
        return (self.current_overall_pick() - 1) // self.cfg.teams + 1

    def next_my_pick(self, after_overall: Optional[int] = None) -> Optional[int]:
        """My next overall pick strictly after ``after_overall`` (default: now)."""
        ref = after_overall if after_overall is not None else self.current_overall_pick()
        for p in self.my_picks:
            if p > ref:
                return p
        return None  # no next pick -> VONA survival term = 0

    # --- roster mutations ---
    def reset(self):
        """Wipe all draft state to empty — used before replaying a pick history."""
        self.drafted.clear()
        self.my_roster.clear()
        self.slot_fill = {s: 0 for s in self.cfg.roster.slots}

    def record_pick(self, player_id: str, position: Optional[str] = None, *,
                    mine: bool = False, team: Optional[str] = None,
                    bye_week: Optional[int] = None):
        self.drafted.add(player_id)
        if mine:
            self.my_roster.append({"player_id": player_id, "position": position,
                                   "team": team, "bye_week": bye_week})
            self._assign_slot(position)

    def my_rb_teams(self) -> set:
        """Teams of RBs I've drafted — for handcuff detection."""
        return {p["team"] for p in self.my_roster
                if p.get("position") == "RB" and p.get("team")}

    def my_bye_week_counts(self) -> dict:
        """Bye week -> count of my drafted players on that bye."""
        counts: dict = {}
        for p in self.my_roster:
            bw = p.get("bye_week")
            if bw is not None:
                counts[bw] = counts.get(bw, 0) + 1
        return counts

    def _assign_slot(self, position: Optional[str]):
        """Fill the first matching slot: pure -> flex -> bench."""
        if position is None:
            self.slot_fill["BENCH"] = self.slot_fill.get("BENCH", 0) + 1
            return
        if self.slot_fill.get(position, 0) < self.cfg.roster.slots.get(position, 0):
            self.slot_fill[position] += 1
        elif (position in self.cfg.roster.flex_eligible and
              self.slot_fill.get("FLEX", 0) < self.cfg.roster.slots.get("FLEX", 0)):
            self.slot_fill["FLEX"] += 1
        else:
            self.slot_fill["BENCH"] = self.slot_fill.get("BENCH", 0) + 1

    # --- slot accounting ---
    def open_pure_slots(self, position: str) -> int:
        return max(0, self.cfg.roster.slots.get(position, 0) - self.slot_fill.get(position, 0))

    def open_flex_slots(self) -> int:
        return max(0, self.cfg.roster.slots.get("FLEX", 0) - self.slot_fill.get("FLEX", 0))

    def needs_starter(self, position: str) -> bool:
        """True if this position still has an open pure or eligible flex starter slot."""
        if self.open_pure_slots(position) > 0:
            return True
        return position in self.cfg.roster.flex_eligible and self.open_flex_slots() > 0

    def unfilled_mandatory_slots(self) -> Dict[str, int]:
        """Mandatory (non-bench) starter slots still open, by slot name."""
        out = {}
        for slot, count in self.cfg.roster.slots.items():
            if slot == "BENCH":
                continue
            open_n = count - self.slot_fill.get(slot, 0)
            if open_n > 0:
                out[slot] = open_n
        return out

    def position_count(self, position: str) -> int:
        """How many players at this position I've already drafted (any slot)."""
        return sum(1 for p in self.my_roster if p.get("position") == position)

    def remaining_picks(self) -> int:
        return self.cfg.roster.rounds - len(self.my_roster)

    def total_unfilled_mandatory(self) -> int:
        return sum(self.unfilled_mandatory_slots().values())
