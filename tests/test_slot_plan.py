"""Slot algebra and the Hall-condition oracle (§15.3, §21.7).

The oracle is checked against an actual bipartite matching rather than against
its own arithmetic, because two independent reviewers caught two different
errors in this derivation and both looked correct on inspection.
"""
from __future__ import annotations

import itertools
import random

import pytest

from src.core.config.slots import build_slot_plan, is_feasible, is_laminar

LIVE_SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1, "BENCH": 6}
LIVE_FLEX = {"FLEX": ["RB", "WR", "TE"]}


def _matches(copies, counts) -> bool:
    """Ground truth: can these players be assigned to distinct slots?

    Kuhn's algorithm on the player->copy bipartite graph. Players of one
    position are interchangeable, so a player is just its position.
    """
    players = [p for pos, n in counts.items() for p in [pos] * n]
    if len(players) > len(copies):
        return False
    assigned: dict[int, int] = {}  # copy index -> player index

    def try_assign(pi: int, seen: set[int]) -> bool:
        for ci, elig in enumerate(copies):
            if players[pi] not in elig or ci in seen:
                continue
            seen.add(ci)
            if ci not in assigned or try_assign(assigned[ci], seen):
                assigned[ci] = pi
                return True
        return False

    return all(try_assign(pi, set()) for pi in range(len(players)))


@pytest.fixture(scope="module")
def live_plan():
    return build_slot_plan(LIVE_SLOTS, LIVE_FLEX)


def test_live_roster_caps(live_plan):
    """The caps the design states, including the pairs r4 omitted."""
    caps = live_plan.caps
    assert caps[frozenset({"RB"})] == 3          # 2 pure + FLEX
    assert caps[frozenset({"WR"})] == 3
    assert caps[frozenset({"TE"})] == 2
    assert caps[frozenset({"QB"})] == 1
    assert caps[frozenset({"K"})] == 1
    assert caps[frozenset({"DST"})] == 1
    # the binding pairs the "family only" quantifier missed
    assert caps[frozenset({"RB", "WR"})] == 5
    assert caps[frozenset({"RB", "TE"})] == 4
    assert caps[frozenset({"WR", "TE"})] == 4
    assert caps[frozenset({"RB", "WR", "TE"})] == 6


def test_flex_slot_name_is_not_a_position(live_plan):
    """FLEX names a slot, not a position; it must not enter the universe."""
    assert "FLEX" not in live_plan.positions
    assert set(live_plan.positions) == {"QB", "RB", "WR", "TE", "K", "DST"}


def test_the_counterexample_that_broke_r4(live_plan):
    """3 RB + 3 WR passes every single-position cap but is infeasible.

    Only RB, RB, WR, WR, FLEX = 5 copies can absorb six RB/WR players. This is
    reachable by incremental insertion, so an oracle quantified over the
    eligibility family alone would admit a 6-starter lineup into 5 slots.
    """
    counts = {"RB": 3, "WR": 3}
    assert live_plan.caps[frozenset({"RB"})] >= 3
    assert live_plan.caps[frozenset({"WR"})] >= 3
    assert live_plan.caps[frozenset({"RB", "WR", "TE"})] >= 6
    assert not is_feasible(live_plan, counts)
    assert not _matches(live_plan.copies, counts)


@pytest.mark.parametrize("counts", [
    {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},   # exact starters
    {"QB": 1, "RB": 3, "WR": 2, "TE": 1, "K": 1, "DST": 1},   # RB in flex
    {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DST": 1},   # WR in flex
    {"QB": 1, "RB": 2, "WR": 2, "TE": 2, "K": 1, "DST": 1},   # TE in flex
    {"RB": 3, "WR": 3},                                        # infeasible
    {"RB": 3, "TE": 2},                                        # infeasible
    {"WR": 3, "TE": 2},                                        # infeasible
    {"QB": 2},                                                 # infeasible
])
def test_oracle_matches_bipartite_matching(live_plan, counts):
    assert is_feasible(live_plan, counts) == _matches(live_plan.copies, counts)


def test_exhaustive_against_matching(live_plan):
    """Every count vector up to 4 per position, checked against matching."""
    positions = live_plan.positions
    mismatches = []
    for combo in itertools.product(range(5), repeat=len(positions)):
        counts = dict(zip(positions, combo))
        if is_feasible(live_plan, counts) != _matches(live_plan.copies, counts):
            mismatches.append(counts)
    assert not mismatches, f"oracle disagrees with matching on: {mismatches[:5]}"


def test_superflex_allows_two_quarterbacks():
    plan = build_slot_plan(
        {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "SUPERFLEX": 1, "BENCH": 6},
        {"SUPERFLEX": ["QB", "RB", "WR", "TE"]},
    )
    assert plan.caps[frozenset({"QB"})] == 2
    assert is_feasible(plan, {"QB": 2, "RB": 2, "WR": 2, "TE": 1})
    assert not is_feasible(plan, {"QB": 3})


def test_overlapping_flexes_are_not_laminar():
    """Two overlapping flex types break laminarity -> matching oracle."""
    plan = build_slot_plan(
        {"QB": 1, "RB": 1, "WR": 1, "TE": 1, "RBWR": 1, "WRTE": 1, "BENCH": 4},
        {"RBWR": ["RB", "WR"], "WRTE": ["WR", "TE"]},
    )
    assert not plan.laminar
    assert is_laminar(build_slot_plan(LIVE_SLOTS, LIVE_FLEX).copies)


def test_non_laminar_oracle_still_exact():
    """Correctness does not depend on laminarity — only oracle cost does."""
    plan = build_slot_plan(
        {"QB": 1, "RB": 1, "WR": 1, "TE": 1, "RBWR": 1, "WRTE": 1, "BENCH": 4},
        {"RBWR": ["RB", "WR"], "WRTE": ["WR", "TE"]},
    )
    for combo in itertools.product(range(4), repeat=len(plan.positions)):
        counts = dict(zip(plan.positions, combo))
        assert is_feasible(plan, counts) == _matches(plan.copies, counts), counts


def test_random_plans_against_matching():
    """Randomized plans, including non-laminar ones."""
    rng = random.Random(20260804)
    pool = ["QB", "RB", "WR", "TE", "K", "DST"]
    for _ in range(120):
        slots: dict[str, int] = {p: rng.randint(0, 2) for p in pool}
        slots = {k: v for k, v in slots.items() if v}
        flex: dict[str, list[str]] = {}
        for i in range(rng.randint(0, 2)):
            name = f"FLEX{i}"
            elig = rng.sample(pool, rng.randint(2, 4))
            slots[name] = 1
            flex[name] = elig
        if not slots:
            continue
        plan = build_slot_plan(slots, flex)
        if not plan.positions:
            continue
        for _ in range(40):
            counts = {p: rng.randint(0, 3) for p in plan.positions}
            assert is_feasible(plan, counts) == _matches(plan.copies, counts), (
                f"plan={slots} flex={flex} counts={counts}"
            )


def test_full_set_constraint_is_never_pruned():
    """A plan whose every cap equals n must still bound total starters.

    Degenerate witness for the pruning rule: one copy eligible for {A, B}.
    Every cap equals n = 1, so an unconditional "drop when cap == n" would
    delete every constraint and admit unlimited starters.
    """
    plan = build_slot_plan({"FLEX": 1, "BENCH": 1}, {"FLEX": ["A", "B"]})
    assert plan.caps[frozenset({"A", "B"})] == 1
    assert is_feasible(plan, {"A": 1})
    assert not is_feasible(plan, {"A": 1, "B": 1})
    assert not is_feasible(plan, {"A": 2})
