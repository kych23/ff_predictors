"""Tier-3 board (§7.4), tier-1 reuse (§7.3) and stage timings (§17.4).

The ladder's value is entirely in what happens when things go wrong, so these
tests spend their time on the failure paths: a tier that raises, a tier 0 that
times out partway through, and a tier 3 whose pool is empty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.app.cockpit.ladder import recommend
from src.core.errors import DataError
from src.engine.decision.ladder_tiers import (
    DrawCache,
    StageTimings,
    make_tier0,
    make_tier1,
    make_tier3,
)
from src.engine.decision.tiers import (
    assign_tiers,
    build_tier_table,
    tier3_recommendation,
)


# ------------------------------------------------------------- §7.4 tiers
def test_tiers_are_numbered_best_first():
    """Raw KMeans labels are arbitrary integers. Without the relabel by
    descending centre, 'tier 1' is a random cluster id and the artifact is
    worse than useless on draft night."""
    values = np.array([30.0, 29.0, 28.0, 10.0, 9.0, 8.0, 2.0, 1.0])
    tiers = assign_tiers(values, k=3)
    assert tiers[0] == 1
    assert tiers[-1] == tiers.max()
    assert list(tiers) == sorted(tiers), "descending value -> ascending tier"


def test_degenerate_values_collapse_to_one_tier():
    assert list(assign_tiers(np.zeros(20), k=8)) == [1] * 20


def test_fewer_players_than_k_does_not_raise():
    tiers = assign_tiers(np.array([5.0, 3.0]), k=8)
    assert len(tiers) == 2 and set(tiers) == {1, 2}


def test_empty_input_is_empty_output():
    assert len(assign_tiers(np.array([]))) == 0


def _board():
    return pd.DataFrame({
        "player_id": [f"p{i}" for i in range(12)],
        "player_name": [f"Player {i}" for i in range(12)],
        "position": ["RB"] * 6 + ["DST"] * 6,
        "value": list(np.linspace(20, 5, 6)) + [0.0] * 6,
        "adp": [10, 20, 30, 40, 50, 60, 95, 100, 107, 120, 130, 140],
    })


def test_unvalued_positions_fall_back_to_adp_order():
    """K and DST are not modeled, so every one of them ties at 0.0. Without an
    ADP tiebreak the within-position order is whatever groupby produced — and
    tier 3 is exactly the moment an arbitrary order becomes a real pick."""
    table = build_tier_table(_board())
    dst = table[table["position"] == "DST"]
    assert list(dst["adp"]) == sorted(dst["adp"])


def test_ranks_restart_within_each_position():
    table = build_tier_table(_board())
    for _, group in table.groupby("position"):
        assert list(group["rank"]) == list(range(1, len(group) + 1))


def test_missing_columns_are_named_not_guessed():
    with pytest.raises(DataError, match="position"):
        build_tier_table(pd.DataFrame({"player_id": ["a"], "value": [1.0]}))


def test_tier3_respects_roster_need():
    table = build_tier_table(_board())
    pick = tier3_recommendation(table, ["DST"], drafted=set())
    assert pick["position"] == "DST"


def test_tier3_skips_drafted_players():
    table = build_tier_table(_board())
    first = tier3_recommendation(table, ["RB"], drafted=set())
    second = tier3_recommendation(table, ["RB"], drafted={first["player_id"]})
    assert second["player_id"] != first["player_id"]


def test_tier3_returns_none_rather_than_an_empty_pick():
    """A silent empty is how a fallback turns a bad night into a missed pick."""
    table = build_tier_table(_board())
    everyone = set(table["player_id"])
    assert tier3_recommendation(table, ["RB"], drafted=everyone) is None
    assert tier3_recommendation(table, [], drafted=set()) is None


# --------------------------------------------------------- §7.3 tier 1
def _evaluate_factory(edge: dict[str, float], reps: int = 64, calls=None):
    def evaluate(candidate: str, draw: int) -> np.ndarray:
        if calls is not None:
            calls.append((candidate, draw))
        rng = np.random.default_rng(abs(hash((candidate, draw))) % 2**32)
        return edge[candidate] + rng.normal(0, 1.0, reps)
    return evaluate


def test_tier1_reuses_tier0_draws_and_simulates_nothing_new():
    """THE point of tier 1. If it recomputed, the remaining budget would go on
    rediscovering what is already in memory."""
    calls: list = []
    cache = DrawCache(_evaluate_factory({"a": 10.0, "b": 4.0}, calls=calls))
    timings = StageTimings()
    names = {"a": "A", "b": "B"}

    make_tier0(cache, ["a", "b"], names, timings, indifference_zone=2.0,
               initial_draws=2, max_draws=4)(allowance_s=5.0)
    spent = len(calls)
    assert spent > 0

    rec = make_tier1(cache, names, timings, indifference_zone=2.0)(5.0)
    assert len(calls) == spent, "tier 1 must not evaluate anything new"
    # it read tier 0's work rather than re-requesting it
    assert cache.n_draws == spent
    assert set(cache.completed()) == {"a", "b"}
    assert rec.tier == 1
    assert rec.leader == "a"
    assert rec.stopped_because == "reused_tier0_draws"


def test_tier1_flags_unequal_draw_counts_rather_than_hiding_them():
    """Halving leaves survivors with more draws than eliminated arms. Comparing
    them is still the best available answer, but a reader must be told."""
    cache = DrawCache(_evaluate_factory({"a": 10.0, "b": 4.0}))
    cache("a", 0), cache("a", 1), cache("b", 0)
    rec = make_tier1(cache, {}, StageTimings(), indifference_zone=2.0)(5.0)
    assert "unequal_draw_counts" in rec.stale_flags


def test_tier1_raises_when_tier0_completed_nothing():
    """Raising is correct: it demotes to tier 2 rather than inventing a leader
    from an empty cache."""
    cache = DrawCache(_evaluate_factory({"a": 1.0}))
    with pytest.raises(DataError, match="zero draws"):
        make_tier1(cache, {}, StageTimings(), indifference_zone=2.0)(5.0)


def test_tier1_indifference_set_widens_with_the_zone():
    cache = DrawCache(_evaluate_factory({"a": 10.0, "b": 9.0}))
    for c in ("a", "b"):
        for d in range(4):
            cache(c, d)
    tight = make_tier1(cache, {}, StageTimings(), indifference_zone=0.01)(5.0)
    loose = make_tier1(cache, {}, StageTimings(), indifference_zone=5.0)(5.0)
    assert len(loose.indifference_set) >= len(tight.indifference_set)
    assert set(loose.indifference_set) == {"a", "b"}


# ------------------------------------------------------ §17.4 timings
def test_timings_are_recorded_even_when_a_stage_raises():
    """The timing of a failure is what distinguishes a timeout from an instant
    crash, which is the whole diagnostic question after a demotion."""
    timings = StageTimings()
    with pytest.raises(ValueError), timings.stage("boom"):
        raise ValueError("x")
    assert "boom" in timings.stage_timings_ms


def test_timings_survive_demotion_all_the_way_down():
    timings = StageTimings()
    table = build_tier_table(_board())

    def dying_tier0(allowance_s):
        with timings.stage("tier0_allocate"):
            raise RuntimeError("kernel exploded")

    rec = recommend(
        {0: dying_tier0,
         3: make_tier3(table, ["RB"], set(), timings)},
        budget_s=5.0, demote_after_s=1.0,
        timings=timings.stage_timings_ms,
    )
    assert rec.tier == 3
    assert "tier0_allocate" in rec.stage_timings_ms, (
        "the failing stage's timing is the reason the demotion happened"
    )
    assert "tier3_static" in rec.stage_timings_ms


def test_total_and_describe_are_readable():
    timings = StageTimings()
    with timings.stage("a"):
        pass
    with timings.stage("b"):
        pass
    assert timings.total_ms >= 0
    assert "a" in timings.describe() and "b" in timings.describe()


def test_tier3_exhaustion_demotes_to_the_hard_failure():
    """Every rung failing must raise, not return an empty recommendation."""
    timings = StageTimings()
    table = build_tier_table(_board())
    with pytest.raises(RuntimeError, match="every tier failed"):
        recommend({3: make_tier3(table, ["RB"], set(table["player_id"]),
                                 timings)},
                  budget_s=2.0, demote_after_s=1.0)
