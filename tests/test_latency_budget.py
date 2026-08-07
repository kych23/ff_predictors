"""§17.4 latency budget — the recommendation fits inside the pick clock.

Draft night is the one deadline this project cannot renegotiate. A
recommendation that arrives after the pick is worth nothing, which is why §7.3
demotes on a wall clock rather than on a judgment call, and why the budget is
asserted rather than hoped for.

**Reference machine is the build machine (Apple M4 Pro, 24 GB).** These
numbers are not portable and are not meant to be; the point is to catch a
regression on the machine that will actually be sitting at the draft.

Marked `slow` and excluded from the default run by `addopts`. §22.1's day-20
chaos rehearsal runs it explicitly.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from src.app.cockpit.ladder import recommend
from src.core.config import load_league, load_strategy
from src.core.config.slots import build_slot_plan
from src.engine.decision.ladder_tiers import (
    DrawCache,
    StageTimings,
    make_tier0,
    make_tier1,
    make_tier3,
)
from src.engine.decision.tiers import build_tier_table
from src.engine.sim import kernel
from src.engine.sim.draws import ProjectionBundle, draw_points

ROOT = "3dc0676bb0114bd3bb66e8d97c210fbb"


@pytest.fixture(scope="module")
def cfg():
    return load_league()


def _bundle(n_players: int, weeks: int) -> ProjectionBundle:
    rng = np.random.default_rng(0)
    p50 = rng.uniform(4.0, 22.0, n_players)
    positions = np.array(["QB", "RB", "WR", "TE"])[
        rng.integers(0, 4, n_players)]
    return ProjectionBundle(
        player_ids=np.array([f"p{i}" for i in range(n_players)]),
        positions=positions,
        nfl_teams=np.array([f"T{i % 32}" for i in range(n_players)]),
        slot_idx=rng.integers(0, 7, n_players),
        bye_weeks=rng.integers(5, 15, n_players),
        is_kdst=np.zeros(n_players, dtype=bool),
        rate_p10=p50 * 0.7, rate_p50=p50, rate_p90=p50 * 1.4,
        weekly_sigma=p50 * 0.4,
        games_hazard=np.full((n_players, weeks), 0.85),
        kdst_quantiles=np.full((n_players, 101), np.nan),
    )


@pytest.mark.slow
def test_one_full_draw_and_evaluate_fits_the_pick_clock(cfg):
    """The hot path at production size: 250 players, 17 weeks, 2048 reps."""
    strategy = load_strategy(cfg)
    budget = float(strategy.simulation.decision["latency_budget_seconds"])
    weeks = cfg.schedule.regular_season_weeks + len(cfg.schedule.playoff_weeks)
    bundle = _bundle(250, weeks)
    chol = np.linalg.cholesky(np.eye(7))
    plan = build_slot_plan(cfg.roster.slots, cfg.roster.flex_eligibility)
    rosters = [np.arange(t * 15, (t + 1) * 15) for t in range(cfg.teams)]

    start = time.perf_counter()
    points = draw_points(bundle, chol, ROOT, reps=2048)
    masks = kernel.build_masks(bundle, rosters, plan, weeks)
    kernel.evaluate_rosters(points, masks)
    elapsed = time.perf_counter() - start

    assert elapsed < budget, (
        f"one draw+evaluate took {elapsed:.1f}s against a "
        f"{budget:.0f}s budget"
    )


@pytest.mark.slow
def test_the_ladder_returns_inside_its_budget_even_when_tier_zero_overruns():
    """THE property. Tier 0 is allowed to be slow; the LADDER is not. A tier
    that ignores its allowance must be cut off, not waited for."""
    timings = StageTimings()
    table = build_tier_table(pd.DataFrame({
        "player_id": [f"p{i}" for i in range(40)],
        "player_name": [f"Player {i}" for i in range(40)],
        "position": ["RB"] * 20 + ["WR"] * 20,
        "value": np.linspace(20, 2, 40),
        "adp": np.arange(1, 41),
    }))

    def slow_tier0(allowance_s):
        with timings.stage("tier0_allocate"):
            time.sleep(allowance_s + 0.3)
            raise TimeoutError("overran its allowance")

    start = time.perf_counter()
    rec = recommend({0: slow_tier0, 3: make_tier3(table, ["RB"], set(), timings)},
                    budget_s=2.0, demote_after_s=0.5,
                    timings=timings.stage_timings_ms)
    elapsed = time.perf_counter() - start

    assert rec.tier == 3
    assert elapsed < 2.0, f"the ladder took {elapsed:.2f}s to demote"
    assert "tier0_allocate" in rec.stage_timings_ms


@pytest.mark.slow
def test_tier1_reuse_is_effectively_free():
    """Tier 1 exists to be reached when the clock has nearly run out. If it
    cost anything material it would be unreachable exactly when needed."""
    cache = DrawCache(lambda c, d: np.random.default_rng(d).normal(30, 5, 256))
    names = {"a": "A", "b": "B"}
    timings = StageTimings()
    make_tier0(cache, ["a", "b"], names, timings, indifference_zone=2.0,
               initial_draws=4, max_draws=8)(allowance_s=10.0)

    start = time.perf_counter()
    rec = make_tier1(cache, names, timings, indifference_zone=2.0)(0.1)
    elapsed = time.perf_counter() - start

    assert rec.tier == 1
    assert elapsed < 0.05, f"tier 1 took {elapsed * 1000:.0f}ms of pure reuse"


@pytest.mark.slow
def test_the_static_board_answers_in_milliseconds():
    """Tier 3's whole justification: no simulator, no model, one parquet-shaped
    frame. If it were not near-instant it would share a failure mode with the
    tiers it exists to catch."""
    table = build_tier_table(pd.DataFrame({
        "player_id": [f"p{i}" for i in range(300)],
        "player_name": [f"Player {i}" for i in range(300)],
        "position": (["RB"] * 100 + ["WR"] * 100 + ["QB"] * 50 + ["TE"] * 50),
        "value": np.linspace(25, 1, 300),
        "adp": np.arange(1, 301),
    }))
    timings = StageTimings()
    fn = make_tier3(table, ["RB", "WR"], set(), timings)

    start = time.perf_counter()
    for _ in range(20):
        fn(1.0)
    elapsed = (time.perf_counter() - start) / 20

    assert elapsed < 0.05, f"tier 3 averaged {elapsed * 1000:.0f}ms"


@pytest.mark.slow
def test_stage_timings_sum_to_roughly_the_wall_clock():
    """§17.4's timings are only useful if they account for the time. A stage
    map that omits the expensive step points at the wrong thing."""
    timings = StageTimings()
    with timings.stage("a"):
        time.sleep(0.05)
    with timings.stage("b"):
        time.sleep(0.10)
    assert 140 < timings.total_ms < 220
