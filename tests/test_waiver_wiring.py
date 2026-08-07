"""The waiver floor is reachable from the evaluation path (§15.6, §8.6).

`apply_waiver` was written, tested, and then never called: for weeks its only
call site in the repository was its own unit test. The simulator carried every
drafted bust at his realized production for all fourteen weeks, which prices a
late-round flier as though it could never be cut.

These tests are about the WIRING, not the algorithm — `test_posterior_waiver.py`
covers the swap logic. What is asserted here is that the correction is actually
applied, that the config switch works in both directions, and that the pool is
the undrafted players rather than anybody's roster.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from src.engine.sim import kernel


@dataclass
class _Bundle:
    n_players: int
    rate_p50: np.ndarray
    positions: np.ndarray


def _bundle(n=10):
    return _Bundle(n_players=n,
                   rate_p50=np.linspace(20.0, 2.0, n),
                   positions=np.array(["RB"] * n))


def _strategy(enabled=True, pool_size=4, in_decision_path=True):
    return SimpleNamespace(simulation=SimpleNamespace(waiver={
        "enabled": enabled, "in_decision_path": in_decision_path,
        "detection_lag_weeks": 1, "pool_size": pool_size,
        "shrinkage_k_games": 6,
    }))


def _cfg(weeks=4):
    return SimpleNamespace(schedule=SimpleNamespace(regular_season_weeks=weeks))


# ------------------------------------------------------------- the pool
def test_the_pool_excludes_every_rostered_player():
    b = _bundle(10)
    rosters = [np.array([0, 1]), np.array([2, 3])]
    pool = kernel.free_agent_pool(b, rosters, size=10)
    assert set(pool).isdisjoint({0, 1, 2, 3})
    assert set(pool) == {4, 5, 6, 7, 8, 9}


def test_the_pool_takes_the_best_available_not_an_arbitrary_slice():
    """A waiver wire is picked over; what matters is the top of what is left."""
    b = _bundle(10)
    pool = kernel.free_agent_pool(b, [np.array([0])], size=3)
    assert set(pool) == {1, 2, 3}          # rate_p50 descends with index


def test_an_exhausted_pool_is_empty_not_negative():
    b = _bundle(4)
    pool = kernel.free_agent_pool(b, [np.arange(4)], size=10)
    assert pool.size == 0


def test_pool_size_caps_the_search():
    b = _bundle(10)
    assert kernel.free_agent_pool(b, [], size=4).size == 4


# ---------------------------------------------------------- the switch
def _bust_bundle(n=10):
    """Two drafted starters priced high; the pool sits between them and the
    bench. Only a *realized* collapse should trigger a claim."""
    rate = np.array([18., 5., 17., 5., 12., 11., 10., 9., 8., 7.])[:n]
    return _Bundle(n_players=n, rate_p50=rate,
                   positions=np.array(["RB"] * n))


def _inputs(reps=3, weeks=8, n=10, teams=2, bust=True):
    rng = np.random.default_rng(0)
    points = rng.normal(10.0, 2.0, (n, weeks, reps)).astype("float32")
    if bust:
        points[0, :, :] = 0.2      # team 0's starter craters, every week
    masks = np.zeros((teams, weeks, n), dtype="float32")
    masks[0, :, 0] = 1.0
    masks[1, :, 2] = 1.0
    team_week = kernel.evaluate_rosters(points, masks)
    rosters = [np.array([0, 1]), np.array([2, 3])]
    return points, masks, team_week, rosters


def test_disabled_returns_the_input_untouched():
    """The pre-waiver behaviour must stay exactly reproducible."""
    points, masks, team_week, rosters = _inputs()
    out = kernel.apply_waiver_floor(points, masks, team_week, rosters,
                                    _bust_bundle(), _cfg(8),
                                    _strategy(enabled=False))
    assert out is team_week


def test_enabled_actually_changes_the_evaluation():
    """The regression this file exists for: with the switch on, the number the
    objective consumes must differ from the raw one. If these are equal the
    correction is not wired in, whatever the config says."""
    points, masks, team_week, rosters = _inputs()
    out = kernel.apply_waiver_floor(points, masks, team_week, rosters,
                                    _bust_bundle(), _cfg(8),
                                    _strategy(enabled=True))
    assert out.shape == team_week.shape
    assert not np.array_equal(out, team_week)


def test_the_correction_replaces_the_bust_rather_than_erasing_him():
    """A dropped starter is replaced by a pool player, so the team-week rises
    to roughly replacement -- it does not go to zero, and it does not stay at
    the bust's production."""
    points, masks, team_week, rosters = _inputs()
    out = kernel.apply_waiver_floor(points, masks, team_week, rosters,
                                    _bust_bundle(), _cfg(8),
                                    _strategy(enabled=True))
    before, after = team_week[0, :, 0], out[0, :, 0]
    assert before.max() < 1.0            # the bust really did crater
    assert after[-1] > 5.0               # and was replaced by late season


def test_the_first_weeks_are_unchanged_because_a_drop_takes_time_to_decide():
    """`detection_lag_weeks` is the whole point: dropping a bust the instant he
    busts is the clairvoyance bug, and it makes every flier identical."""
    points, masks, team_week, rosters = _inputs()
    out = kernel.apply_waiver_floor(points, masks, team_week, rosters,
                                    _bust_bundle(), _cfg(8),
                                    _strategy(enabled=True))
    assert out[0, 0, 0] == pytest.approx(team_week[0, 0, 0])


def test_a_team_with_no_bust_is_left_alone():
    """Team 1's starter performs; nothing about him should change."""
    points, masks, team_week, rosters = _inputs()
    out = kernel.apply_waiver_floor(points, masks, team_week, rosters,
                                    _bust_bundle(), _cfg(8),
                                    _strategy(enabled=True))
    assert np.allclose(out[1], team_week[1])


def test_an_empty_pool_is_a_no_op_rather_than_a_crash():
    """Late in a deep league there is genuinely nobody left to claim."""
    points, masks, team_week, rosters = _inputs()
    full = [np.arange(10)]
    out = kernel.apply_waiver_floor(points, masks, team_week, full,
                                    _bust_bundle(), _cfg(8),
                                    _strategy(enabled=True))
    assert out is team_week


def test_a_missing_waiver_block_defaults_to_off():
    """A strategy file without the key must not crash the pick path."""
    points, masks, team_week, rosters = _inputs()
    bare = SimpleNamespace(simulation=SimpleNamespace(waiver={}))
    out = kernel.apply_waiver_floor(points, masks, team_week, rosters,
                                    _bust_bundle(), _cfg(8), bare)
    assert out is team_week


def test_the_decision_path_switch_is_independent_of_enabled():
    """`enabled` alone must not put a 450 ms correction on the pick clock."""
    points, masks, team_week, rosters = _inputs()
    strategy = _strategy(enabled=True, in_decision_path=False)
    on_clock = kernel.apply_waiver_floor(points, masks, team_week, rosters,
                                         _bust_bundle(), _cfg(8), strategy,
                                         in_decision_path=True)
    offline = kernel.apply_waiver_floor(points, masks, team_week, rosters,
                                        _bust_bundle(), _cfg(8), strategy)
    assert on_clock is team_week                 # refused on the clock
    assert not np.array_equal(offline, team_week)  # still applied offline


def test_the_shipped_strategy_enables_the_waiver_offline():
    """Config drift check: this defaulted to off for the whole period the call
    site was missing, and nothing noticed."""
    from src.core.config import load_league, load_strategy

    waiver = load_strategy(load_league()).simulation.waiver
    assert waiver["enabled"] is True
    # Deliberately off on the clock — see the cost note in strategy.yaml.
    assert waiver["in_decision_path"] is False
