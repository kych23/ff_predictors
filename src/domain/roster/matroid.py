"""Independence oracles for lineup selection (§15.3, AD-5).

The family of simultaneously-assignable player sets is a **transversal
matroid** for any eligibility structure, so greedy selection by value is
optimal always. Laminarity is not a correctness condition — it only decides
whether the oracle is O(1) counters or an incremental bipartite matching.

Both oracles answer the same question and are tested against each other and
against a from-scratch matching in ``tests/test_lineup.py``.
"""
from __future__ import annotations

from typing import Protocol

from src.core.config.slots import SlotPlan


class SlotOracle(Protocol):
    """Incremental independence test over roster slots."""

    def reset(self) -> None: ...
    def can_add(self, position: str) -> bool: ...
    def add(self, position: str) -> None: ...


class CounterOracle:
    """O(1) per query. Valid when the eligibility family is laminar.

    Maintains a count per position and checks it against the precomputed Hall
    caps (``SlotPlan.caps``), which quantify over every position subset — not
    merely the eligibility sets in the family. Quantifying over the family
    alone admits 3 RB + 3 WR on the live roster, where only five copies can
    absorb six RB/WR players.
    """

    def __init__(self, plan: SlotPlan) -> None:
        self._plan = plan
        # Only the subsets containing a given position can be violated by
        # adding one, so index them once rather than scanning all caps.
        self._caps_by_position: dict[str, tuple[tuple[frozenset[str], int], ...]] = {
            pos: tuple((q, cap) for q, cap in plan.caps.items() if pos in q)
            for pos in plan.positions
        }
        self._counts: dict[str, int] = {}
        self.reset()

    def reset(self) -> None:
        self._counts = dict.fromkeys(self._plan.positions, 0)

    def can_add(self, position: str) -> bool:
        relevant = self._caps_by_position.get(position)
        if relevant is None:
            return False  # no slot accepts this position at all
        for subset, cap in relevant:
            used = sum(self._counts[p] for p in subset)
            if used + 1 > cap:
                return False
        return True

    def add(self, position: str) -> None:
        if not self.can_add(position):
            raise ValueError(f"adding {position!r} would violate slot capacity")
        self._counts[position] += 1


class MatchingOracle:
    """Incremental bipartite matching. Valid for any eligibility structure.

    One augmenting-path search (Kuhn's algorithm) per query, O(V*E).
    Hopcroft-Karp buys nothing here because insertions arrive one at a time.
    """

    def __init__(self, plan: SlotPlan) -> None:
        self._copies = plan.copies
        self._positions: list[str] = []
        self._match_copy_to_player: dict[int, int] = {}
        self.reset()

    def reset(self) -> None:
        self._positions = []
        self._match_copy_to_player = {}

    def _augment(self, player_idx: int, seen: set[int],
                 matching: dict[int, int]) -> bool:
        position = self._positions[player_idx]
        for ci, eligible in enumerate(self._copies):
            if position not in eligible or ci in seen:
                continue
            seen.add(ci)
            if ci not in matching or self._augment(matching[ci], seen, matching):
                matching[ci] = player_idx
                return True
        return False

    def can_add(self, position: str) -> bool:
        if len(self._positions) + 1 > len(self._copies):
            return False
        trial = dict(self._match_copy_to_player)
        self._positions.append(position)
        try:
            return self._augment(len(self._positions) - 1, set(), trial)
        finally:
            self._positions.pop()

    def add(self, position: str) -> None:
        self._positions.append(position)
        if not self._augment(len(self._positions) - 1, set(),
                             self._match_copy_to_player):
            self._positions.pop()
            raise ValueError(f"adding {position!r} would violate slot capacity")


def build_oracle(plan: SlotPlan) -> SlotOracle:
    """CounterOracle when the family is laminar, MatchingOracle otherwise."""
    return CounterOracle(plan) if plan.laminar else MatchingOracle(plan)
