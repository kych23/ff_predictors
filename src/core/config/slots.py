"""Slot algebra and the lineup independence oracle (§15.3).

Roster slots expand into one *copy* per unit of capacity, each carrying the set
of positions it accepts. The family of simultaneously-assignable player sets is
then a **transversal matroid**, so greedy selection by value is optimal for any
eligibility structure — laminarity is not a correctness condition, it only
decides whether the independence oracle is O(1) counters or incremental
bipartite matching.

The caps below are Hall's condition. Getting them wrong is subtle and costly,
so the two ways to get it wrong are recorded here:

  * ``cap(Q)`` counts copies whose eligibility **intersects** Q, not copies
    whose eligibility is a subset of Q. The subset reading gives cap({RB}) = 2
    for the live roster (the two pure RB copies), which contradicts the RB<=3
    it is supposed to produce and would stop greedy putting a third RB in FLEX.

  * Q ranges over **every subset of positions**, not merely the distinct
    eligibility sets in the family. For the live roster the family is
    {QB},{RB},{WR},{TE},{RB,WR,TE},{K},{DST}, which omits {RB,WR} — and
    3 RB + 3 WR passes every family cap while only RB,RB,WR,WR,FLEX = 5 copies
    can absorb six RB/WR players. That state is reachable by incremental
    insertion, so the oracle would admit a six-starter lineup into five slots.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Sequence

from src.core.constants import SLOT_BENCH


@dataclass(frozen=True)
class SlotPlan:
    """Expanded roster capacity plus the derived independence constraints."""

    #: One entry per unit of starting capacity, holding the positions it accepts.
    copies: tuple[frozenset[str], ...]
    #: Positions appearing anywhere in the plan or the roster.
    positions: tuple[str, ...]
    #: Hall constraints: position-subset -> maximum simultaneous starters.
    #: Non-binding subsets are pruned (see ``_derive_caps``).
    caps: Mapping[frozenset[str], int]
    #: True iff every pair of distinct eligibility sets is nested or disjoint.
    #: Selects the O(1) counter oracle over incremental matching.
    laminar: bool

    @property
    def n_starters(self) -> int:
        return len(self.copies)


def is_laminar(sets: Iterable[frozenset[str]]) -> bool:
    """True iff every pair of sets is nested or disjoint."""
    distinct = {frozenset(s) for s in sets}
    for a, b in combinations(distinct, 2):
        if a & b and not (a <= b or b <= a):
            return False
    return True


def _derive_caps(
    copies: Sequence[frozenset[str]], positions: Sequence[str]
) -> dict[frozenset[str], int]:
    """Hall's condition for every position subset.

    ``cap(Q)`` = number of copies whose eligibility intersects Q. A selection is
    independent iff, for every Q, the number of selected players with position
    in Q is at most ``cap(Q)``.

    Pruning: a subset whose cap equals the total copy count is non-binding and
    dropped — EXCEPT the full position set, which must always be retained. Its
    cap is always ``n``, and it is the only constraint carrying
    ``total starters <= n``; dropping it unconditionally lets a plan whose caps
    all equal ``n`` (e.g. a single copy eligible for {A, B}) admit unlimited
    starters.
    """
    n = len(copies)
    full = frozenset(positions)
    caps: dict[frozenset[str], int] = {}

    for size in range(1, len(positions) + 1):
        for combo in combinations(sorted(positions), size):
            q = frozenset(combo)
            cap = sum(1 for c in copies if c & q)
            if cap == n and q != full:
                continue  # non-binding
            caps[q] = cap

    caps[full] = n
    return caps


def build_slot_plan(
    slots: Mapping[str, int],
    flex_eligibility: Mapping[str, Sequence[str]],
    *,
    extra_positions: Iterable[str] = (),
) -> SlotPlan:
    """Expand ``slots`` into copies and derive the independence constraints.

    A slot named in ``flex_eligibility`` accepts the positions listed there; any
    other non-BENCH slot accepts only the position sharing its name. BENCH is
    not a starting slot and is excluded.

    ``extra_positions`` lets the caller include positions that exist on the
    roster but have no dedicated slot, so they are quantified over rather than
    left unconstrained.
    """
    copies: list[frozenset[str]] = []
    for slot, count in slots.items():
        if slot == SLOT_BENCH:
            continue
        if count < 0:
            raise ValueError(f"slot {slot!r} has negative count {count}")
        eligible = frozenset(flex_eligibility.get(slot, (slot,)))
        copies.extend([eligible] * int(count))

    positions = sorted({p for c in copies for p in c} | set(extra_positions))
    plan_copies = tuple(copies)
    return SlotPlan(
        copies=plan_copies,
        positions=tuple(positions),
        caps=_derive_caps(plan_copies, positions),
        laminar=is_laminar(plan_copies),
    )


def is_feasible(plan: SlotPlan, counts: Mapping[str, int]) -> bool:
    """Can ``counts`` players be assigned to distinct slots simultaneously?

    Reference implementation of the oracle's verdict, used by tests and by the
    counter oracle in ``domain.roster.matroid``.
    """
    for q, cap in plan.caps.items():
        if sum(n for pos, n in counts.items() if pos in q) > cap:
            return False
    return True
