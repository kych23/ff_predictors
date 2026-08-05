"""Allocator, indifference set, rollout, and the degradation ladder (§15.5, §17, §7.3)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from src.app.cockpit.ladder import Recommendation, recommend
from src.core.config import load_league, load_strategy
from src.engine.decision.allocate import allocate
from src.engine.sim.rollout import rollout, softmax_adp_pick


@pytest.fixture(scope="module")
def league():
    return load_league()


@pytest.fixture(scope="module")
def strategy(league):
    return load_strategy(league)


def _board(n=200):
    rng = np.random.default_rng(0)
    positions = rng.choice(["QB", "RB", "WR", "TE", "K", "DST"], n,
                           p=[0.12, 0.28, 0.36, 0.12, 0.06, 0.06])
    return pd.DataFrame({
        "player_id": [f"p{i}" for i in range(n)],
        "player_name": [f"Player {i}" for i in range(n)],
        "position": positions,
        "team": [f"T{i % 32}" for i in range(n)],
        "bye_week": rng.integers(4, 14, n),
        "value": np.sort(rng.uniform(2, 22, n))[::-1],
        "value_source": "test", "coverage": "full",
        "adp": np.arange(1, n + 1) + rng.normal(0, 2, n),
        "adp_stdev": rng.uniform(1, 8, n),
    })


# ------------------------------------------------------------- allocator
def _maker(truth: dict[str, float], noise: float, seed=0):
    rng = np.random.default_rng(seed)

    def evaluate(candidate: str, draw: int) -> np.ndarray:
        return truth[candidate] + rng.normal(0, noise, 64)
    return evaluate


def test_allocator_finds_a_clearly_better_candidate():
    truth = {"a": 40.0, "b": 20.0, "c": 18.0, "d": 15.0}
    result = allocate(list(truth), _maker(truth, noise=2.0),
                      initial_draws=2, max_draws=16, indifference_zone=2.0)
    assert result.leader == "a"
    assert result.p_best > 0.9


def test_leader_is_the_survivor_not_a_noisy_shallow_arm():
    """Successive halving spends budget on survivors, so eliminated arms hold
    few draws and noisy means. Ranking by raw mean lets one of them win and
    defeats the procedure — the signature is a leader with a LOW p(best)."""
    truth = {"a": 30.0, "b": 29.5, "c": 29.0, "d": 28.5}
    result = allocate(list(truth), _maker(truth, noise=15.0, seed=3),
                      initial_draws=2, max_draws=32, indifference_zone=1.0)
    leader_draws = result.estimates[result.leader].draws_used
    assert leader_draws == max(e.draws_used for e in result.estimates.values())
    assert result.p_best >= 0.5, "a leader the bootstrap disbelieves is incoherent"


def test_indifference_set_holds_genuinely_tied_candidates():
    truth = {"a": 20.0, "b": 19.9, "c": 19.8}
    result = allocate(list(truth), _maker(truth, noise=8.0),
                      initial_draws=2, max_draws=8, indifference_zone=2.0)
    assert len(result.indifference_set) > 1, (
        "candidates within the zone must be reported as tied, not ranked"
    )


def test_indifference_set_narrows_when_the_gap_is_real():
    truth = {"a": 60.0, "b": 20.0, "c": 19.0}
    result = allocate(list(truth), _maker(truth, noise=1.0),
                      initial_draws=2, max_draws=16, indifference_zone=2.0)
    assert result.indifference_set == ["a"]


def test_draw_indices_are_consecutive_so_doubling_reuses():
    seen: list[tuple[str, int]] = []

    def evaluate(candidate: str, draw: int) -> np.ndarray:
        seen.append((candidate, draw))
        return np.full(8, {"a": 10.0, "b": 5.0}[candidate])

    allocate(["a", "b"], evaluate, initial_draws=2, max_draws=8,
             indifference_zone=1.0)
    for cand in ("a", "b"):
        draws = [d for c, d in seen if c == cand]
        assert draws == list(range(len(draws))), (
            "draw indices must be consecutive from 0 or doubling cannot reuse"
        )


def test_aleatory_se_is_not_divided_by_k_times_r():
    """CRN pins the SAME R uniforms across all K draws, so there are only R
    independent aleatory samples, not K*R.

    The fixture therefore reuses one fixed rep vector for every draw — which is
    exactly what §15.1's `c = 0` rule produces. With sd 5 over 100 reps the
    aleatory SE is 5/sqrt(100) = 0.5 no matter how many draws are taken;
    dividing by K*R instead would give 0.5/sqrt(K), understating it by sqrt(K).
    """
    fixed = 10.0 + np.random.default_rng(0).normal(0, 5.0, 100)

    def evaluate(candidate: str, draw: int) -> np.ndarray:
        return fixed.copy()          # identical reps across every draw

    result = allocate(["a"], evaluate, initial_draws=8, max_draws=8)
    est = result.estimates["a"]
    assert est.aleatory_se == pytest.approx(fixed.std(ddof=1) / 10.0, rel=0.02)
    assert est.aleatory_se > 0.4, "must not shrink with K under CRN"
    assert est.epistemic_se == pytest.approx(0.0, abs=1e-9), (
        "identical draws carry no parameter uncertainty"
    )


def test_allocator_respects_a_deadline():
    def slow(candidate: str, draw: int) -> np.ndarray:
        time.sleep(0.02)
        return np.full(4, 10.0)

    start = time.perf_counter()
    result = allocate(["a", "b", "c", "d"], slow, initial_draws=2,
                      max_draws=50, deadline=start + 0.25)
    assert time.perf_counter() - start < 1.0
    assert result.stopped_because in ("deadline", "budget")


# --------------------------------------------------------------- rollout
def test_rollout_fills_every_seat(league, strategy):
    board = _board()
    result = rollout(board, league, strategy, my_seat=3, root="a" * 32, rep=0)
    assert len(result.rosters) == league.teams
    for roster in result.rosters:
        assert len(roster) == league.roster.rounds


def test_rollout_never_drafts_a_player_twice(league, strategy):
    board = _board()
    result = rollout(board, league, strategy, my_seat=0, root="b" * 32, rep=1)
    picked = np.concatenate(result.rosters)
    assert len(picked) == len(set(picked.tolist()))


def test_rollout_honors_the_forced_candidate(league, strategy):
    board = _board()
    forced_row = 7
    result = rollout(board, league, strategy, my_seat=3,
                     forced=(3, forced_row), root="c" * 32, rep=0)
    assert forced_row in result.rosters[3].tolist()


def test_rollout_is_crn_stable(league, strategy):
    """Same rep must reproduce the same opponent picks — the invariant the
    paired comparison depends on."""
    board = _board()
    a = rollout(board, league, strategy, my_seat=3, root="d" * 32, rep=5)
    b = rollout(board, league, strategy, my_seat=3, root="d" * 32, rep=5)
    for x, y in zip(a.rosters, b.rosters):
        assert np.array_equal(x, y)


def test_different_reps_give_different_drafts(league, strategy):
    board = _board()
    a = rollout(board, league, strategy, my_seat=3, root="e" * 32, rep=0)
    b = rollout(board, league, strategy, my_seat=3, root="e" * 32, rep=1)
    assert not all(np.array_equal(x, y) for x, y in zip(a.rosters, b.rosters))


def test_softmax_prefers_players_near_the_current_pick():
    board = _board(50)
    candidates = board.index.to_numpy()
    early = [softmax_adp_pick(board, candidates, 5, 8.0, u)
             for u in np.linspace(0.01, 0.99, 40)]
    assert np.median(early) < 25, "at pick 5 the board should skew early"


# ---------------------------------------------------------------- ladder
def _rec(tier: int) -> Recommendation:
    return Recommendation(tier=tier, leader="x", leader_name="X",
                          ranked=pd.DataFrame([{"player_name": "X"}]),
                          indifference_set=["x"])


def test_ladder_prefers_the_highest_tier():
    rec = recommend({0: lambda b: _rec(0), 2: lambda b: _rec(2)},
                    budget_s=5.0, demote_after_s=5.0)
    assert rec.tier == 0


def test_ladder_demotes_when_a_tier_raises():
    def broken(_budget):
        raise RuntimeError("kernel exploded")

    rec = recommend({0: broken, 2: lambda b: _rec(2), 3: lambda b: _rec(3)},
                    budget_s=5.0, demote_after_s=5.0)
    assert rec.tier == 2, "a failed tier must demote, not propagate"


def test_ladder_falls_all_the_way_to_the_static_list():
    def broken(_budget):
        raise RuntimeError("nope")

    rec = recommend({0: broken, 2: broken, 3: lambda b: _rec(3)},
                    budget_s=5.0, demote_after_s=5.0)
    assert rec.tier == 3


def test_ladder_raises_only_when_every_tier_fails():
    def broken(_budget):
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="no data source left"):
        recommend({0: broken, 3: broken}, budget_s=1.0, demote_after_s=1.0)


def test_ladder_records_elapsed_time():
    rec = recommend({0: lambda b: _rec(0)}, budget_s=5.0, demote_after_s=5.0)
    assert rec.elapsed_s >= 0.0
    assert "tier 0" in rec.describe()


# ------------------------------------------------- roster completeness
def test_rollout_forces_mandatory_slots(league, strategy):
    """Caps stop a manager taking six kickers; nothing stopped them taking
    ZERO. Measured on the real board, 2 of 12 seats finished with no QB and
    every seat fielded 6-7 of 9 starters — so E[$] was measuring roster
    construction failure rather than player value."""
    from src.engine.sim.rollout import unfilled_mandatory

    board = _board(300)
    result = rollout(board, league, strategy, my_seat=3, root="f" * 32, rep=0)
    for seat, roster in enumerate(result.rosters):
        counts = board.loc[roster, "position"].value_counts().to_dict()
        missing = unfilled_mandatory(counts, league)
        assert not missing, f"seat {seat + 1} cannot fill {missing}"


def test_unfilled_mandatory_ignores_flex():
    """Flex is satisfied by any eligible position, so it never forces a pick."""
    from src.engine.sim.rollout import unfilled_mandatory

    cfg = load_league()
    full = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1}
    assert unfilled_mandatory(full, cfg) == {}
    assert unfilled_mandatory({**full, "QB": 0}, cfg) == {"QB": 1}


def test_every_seat_can_field_a_full_lineup_in_a_bye_free_week(league, strategy):
    """A zero selection value means greedy_lineup never starts a player, so a
    roster carrying one silently fields short. K/DST and unvalued rookies both
    hit this; §3.4 warns explicitly against the constant zero."""
    import numpy as np
    from src.core.config.slots import build_slot_plan
    from src.engine.sim import kernel

    board = _board(300)
    board = board.copy()
    board["bye_week"] = 0                     # isolate from bye effects
    result = rollout(board, league, strategy, my_seat=3, root="ab" * 16, rep=0)

    class _Proj:
        positions = board["position"].to_numpy()
        rate_p50 = np.maximum(board["value"].to_numpy(), 0.5)
        games_hazard = np.full((len(board), 3), 1.0)
        bye_weeks = np.zeros(len(board), dtype=int)
        n_players = len(board)

    plan = build_slot_plan(league.roster.slots, league.roster.flex_eligibility)
    masks = kernel.build_masks(_Proj(), result.rosters, plan, 3)
    starters = masks[:, 0, :].sum(axis=1)
    assert (starters == plan.n_starters).all(), (
        f"short lineups: {starters.tolist()} vs {plan.n_starters} slots"
    )
