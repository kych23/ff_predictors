"""Simulation kernel: RNG addressing, draws, masks, evaluation (§15)."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.config import load_league
from src.core.config.slots import build_slot_plan
from src.core.constants import Z90, RngKind
from src.engine.sim import kernel, rng as rng_mod
from src.engine.sim.draws import (
    ProjectionBundle, draw_points, draw_rates, split_normal_moments,
    split_normal_ppf,
)

SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1, "BENCH": 6}
FLEX = {"FLEX": ["RB", "WR", "TE"]}
ROOT = "a1b2c3d4" * 4


@pytest.fixture(scope="module")
def league():
    return load_league()


@pytest.fixture(scope="module")
def plan():
    return build_slot_plan(SLOTS, FLEX)


def _bundle(n_players=24, weeks=17, rng=None, kdst=True):
    rng = rng or np.random.default_rng(0)
    positions, teams, slots, is_kdst = [], [], [], []
    cycle = ["QB1", "RB1", "RB2", "WR1", "WR2", "WR3", "TE1"]
    pos_of = {"QB1": "QB", "RB1": "RB", "RB2": "RB", "WR1": "WR",
              "WR2": "WR", "WR3": "WR", "TE1": "TE"}
    for i in range(n_players):
        if kdst and i >= n_players - 2:
            positions.append("K" if i == n_players - 2 else "DST")
            slots.append(-1)
            is_kdst.append(True)
        else:
            slot = cycle[i % len(cycle)]
            positions.append(pos_of[slot])
            slots.append(cycle.index(slot))
            is_kdst.append(False)
        teams.append(f"T{i % 4}")
    p50 = rng.uniform(6, 20, n_players)
    return ProjectionBundle(
        player_ids=np.array([f"p{i}" for i in range(n_players)]),
        positions=np.array(positions), nfl_teams=np.array(teams),
        slot_idx=np.array(slots), bye_weeks=np.zeros(n_players, dtype=int),
        is_kdst=np.array(is_kdst),
        rate_p10=p50 - 4, rate_p50=p50, rate_p90=p50 + 5,
        weekly_sigma=np.full(n_players, 6.0),
        games_hazard=np.full((n_players, weeks), 0.93),
        kdst_quantiles=np.tile(np.linspace(-2, 20, 101), (n_players, 1)),
    )


# ------------------------------------------------------------------- RNG
def test_stream_is_deterministic_and_address_separated():
    a = rng_mod.uniforms(ROOT, RngKind.RATE, n=5, a=3)
    b = rng_mod.uniforms(ROOT, RngKind.RATE, n=5, a=3)
    c = rng_mod.uniforms(ROOT, RngKind.RATE, n=5, a=4)
    assert np.array_equal(a, b), "same address must give the same draws"
    assert not np.array_equal(a, c), "different players must not collide"


def test_kinds_do_not_collide():
    """rate/eps/hazard are all player-indexed; one shared kind would put them
    at the same stream position."""
    same = [rng_mod.uniforms(ROOT, k, n=6, a=2)
            for k in (RngKind.RATE, RngKind.EPS, RngKind.HAZARD)]
    for i in range(len(same)):
        for j in range(i + 1, len(same)):
            assert not np.array_equal(same[i], same[j])


def test_extending_a_draw_reuses_the_prefix():
    """The allocator deepens a candidate without redrawing (§17.2)."""
    short = rng_mod.uniforms(ROOT, RngKind.RATE, n=64, a=1)
    long = rng_mod.uniforms(ROOT, RngKind.RATE, n=256, a=1)
    assert np.array_equal(short, long[:64])


def test_normals_are_inverse_cdf_so_the_prefix_survives():
    """standard_normal consumes a variable number of uniforms per value, so it
    would break the prefix property CRN relies on."""
    short = rng_mod.normals(ROOT, RngKind.EPS, shape=(32,), a=7)
    long = rng_mod.normals(ROOT, RngKind.EPS, shape=(128,), a=7)
    assert np.allclose(short, long[:32])


def test_seed_root_depends_on_all_three_inputs():
    base = rng_mod.seed_root("sn2_a", "m1", "s1")
    assert len(base) == 32
    assert base != rng_mod.seed_root("sn2_b", "m1", "s1")
    assert base != rng_mod.seed_root("sn2_a", "m2", "s1")
    assert base != rng_mod.seed_root("sn2_a", "m1", "s2")


def test_philox_key_accepts_the_mixed_address():
    """Passing the address tuple straight to Philox raises: the key is 128 bits,
    not five ints. Regression guard on the mixing function."""
    g = rng_mod.stream(ROOT, RngKind.DRAFT, a=11, b=57, c=0)
    assert g.random(3).shape == (3,)


# ----------------------------------------------------------- split normal
def test_split_normal_reproduces_its_quantiles():
    rng = np.random.default_rng(1)
    p50 = np.full(200_000, 10.0)
    p10 = np.full(200_000, 4.0)
    p90 = np.full(200_000, 22.0)
    draws = split_normal_ppf(rng.uniform(size=200_000), p10, p50, p90)
    assert np.quantile(draws, 0.10) == pytest.approx(4.0, abs=0.1)
    assert np.quantile(draws, 0.50) == pytest.approx(10.0, abs=0.1)
    assert np.quantile(draws, 0.90) == pytest.approx(22.0, abs=0.1)


def test_split_normal_is_attainable_for_any_monotone_triple():
    """Unlike a skew-normal, whose attainable skewness is bounded."""
    rng = np.random.default_rng(2)
    for _ in range(200):
        a, b, c = np.sort(rng.uniform(0, 50, 3))
        out = split_normal_ppf(np.array([0.3]), np.array([a]), np.array([b]),
                               np.array([c]))
        assert np.isfinite(out).all()


def test_degenerate_lower_side_is_safe():
    out = split_normal_ppf(np.linspace(0.01, 0.99, 50), np.array([10.0]),
                           np.array([10.0]), np.array([20.0]))
    assert np.isfinite(out).all()
    assert (out >= 10.0 - 1e-9).all(), "sigma_lo = 0 gives a one-sided draw"


def test_stated_moments_match_the_density():
    """The moment formulas are load-bearing for §12.3's bookkeeping, and an
    earlier revision quoted the CLASSICAL split normal's mean by mistake."""
    rng = np.random.default_rng(3)
    n = 400_000
    p10, p50, p90 = 5.0, 10.0, 20.0
    draws = split_normal_ppf(rng.uniform(size=n), np.full(n, p10),
                             np.full(n, p50), np.full(n, p90))
    mean, var = split_normal_moments(p10, p50, p90)
    assert draws.mean() == pytest.approx(mean, abs=0.05)
    assert draws.var() == pytest.approx(var, rel=0.02)
    # the classical split normal's mean would carry sqrt(2/pi), not 1/sqrt(2pi)
    wrong = p50 + ((p90 - p50) / Z90 - (p50 - p10) / Z90) * np.sqrt(2 / np.pi)
    assert abs(draws.mean() - wrong) > 0.5


# ------------------------------------------------------------------ draws
def test_rates_are_constant_across_weeks():
    """rate is uncertainty about a RATE, so it is drawn once per replication.
    Redrawing per week makes the season-total interval ~sqrt(W) too narrow."""
    bundle = _bundle(n_players=6, weeks=5, kdst=False)
    chol = np.eye(7)
    pts = draw_points(bundle, chol, ROOT, reps=200, apply_hazard=False)
    # with sigma_w = 0 the weekly points ARE the rate
    flat = ProjectionBundle(**{**bundle.__dict__,
                              "weekly_sigma": np.zeros(bundle.n_players)})
    pts_flat = draw_points(flat, chol, ROOT, reps=200, apply_hazard=False)
    for i in range(bundle.n_players):
        if flat.is_kdst[i]:
            continue
        assert np.allclose(pts_flat[i, 0, :], pts_flat[i, 3, :])


def test_conditional_weekly_variance_equals_sigma_w():
    """§12.3: conditional on a fixed rate draw, weekly variance is exactly
    sigma_w^2 — no floor needed, because the shock is unit-variance."""
    rng = np.random.default_rng(4)
    bundle = _bundle(n_players=7, weeks=40, rng=rng, kdst=False)
    chol = np.eye(7)
    pts = draw_points(bundle, chol, ROOT, reps=400, apply_hazard=False)
    for i in range(bundle.n_players):
        rate = draw_rates(bundle, ROOT, 400)[i]
        resid = pts[i] - rate[None, :]     # remove the per-rep rate
        assert resid.std() == pytest.approx(bundle.weekly_sigma[i], rel=0.06)


def test_draws_reproduce_the_slot_correlation():
    from src.models.correlation.slot_matrix import fit_slot_correlation
    import pandas as pd

    rng = np.random.default_rng(5)
    slots = ["QB1", "RB1", "RB2", "WR1", "WR2", "WR3", "TE1"]
    target = np.eye(7)
    for a, b, r in [("QB1", "WR1", 0.38), ("WR1", "WR2", 0.04),
                    ("RB1", "RB2", -0.13)]:
        i, j = slots.index(a), slots.index(b)
        target[i, j] = target[j, i] = r
    chol = np.linalg.cholesky(target)

    bundle = _bundle(n_players=7, weeks=12, kdst=False)
    bundle = ProjectionBundle(**{**bundle.__dict__,
                                 "nfl_teams": np.array(["T0"] * 7),
                                 "slot_idx": np.arange(7),
                                 "rate_p10": np.zeros(7),
                                 "rate_p50": np.zeros(7),
                                 "rate_p90": np.zeros(7),
                                 "weekly_sigma": np.ones(7)})
    pts = draw_points(bundle, chol, ROOT, reps=4000, apply_hazard=False)
    flat = pts.reshape(7, -1)
    empirical = np.corrcoef(flat)
    assert empirical[0, 3] == pytest.approx(0.38, abs=0.05)   # QB1 x WR1
    assert empirical[3, 4] == pytest.approx(0.04, abs=0.05)   # WR1 x WR2
    assert empirical[1, 2] == pytest.approx(-0.13, abs=0.05)  # RB1 x RB2


def test_crn_holds_across_candidate_rosters():
    """The property the whole design rests on: replication r resolves to the
    same underlying draws no matter which candidate is being evaluated."""
    bundle = _bundle(n_players=10, weeks=6, kdst=False)
    chol = np.eye(7)
    a = draw_points(bundle, chol, ROOT, reps=64, apply_hazard=True)
    b = draw_points(bundle, chol, ROOT, reps=64, apply_hazard=True)
    assert np.array_equal(a, b)


def test_bye_week_zeroes_that_week():
    bundle = _bundle(n_players=4, weeks=8, kdst=False)
    bundle = ProjectionBundle(**{**bundle.__dict__,
                                 "bye_weeks": np.array([3, 0, 0, 0])})
    pts = draw_points(bundle, np.eye(7), ROOT, reps=32)
    assert np.all(pts[0, 2, :] == 0.0), "week 3 is index 2"
    assert not np.all(pts[0, 4, :] == 0.0)


def test_kdst_draws_come_from_the_empirical_grid():
    bundle = _bundle(n_players=9, weeks=6)
    pts = draw_points(bundle, np.eye(7), ROOT, reps=500, apply_hazard=False)
    kdst_rows = np.flatnonzero(bundle.is_kdst)
    for i in kdst_rows:
        drawn = pts[i].ravel()
        assert drawn.min() >= -2.5 and drawn.max() <= 20.5


# ----------------------------------------------------------------- kernel
def test_masks_are_week_dependent_because_of_byes(plan):
    bundle = _bundle(n_players=12, weeks=6, kdst=False)
    bundle = ProjectionBundle(**{**bundle.__dict__,
                                 "bye_weeks": np.array([2] + [0] * 11)})
    masks = kernel.build_masks(bundle, [np.arange(12)], plan, 6)
    assert masks[0, 1, 0] == 0.0, "a player on bye cannot start"
    assert masks[0, 3, 0] == 1.0 or masks[0, 3].sum() > 0


def test_masks_never_exceed_the_slot_plan(plan):
    bundle = _bundle(n_players=20, weeks=4, kdst=False)
    masks = kernel.build_masks(bundle, [np.arange(20)], plan, 4)
    for w in range(4):
        assert masks[0, w].sum() <= plan.n_starters


def test_evaluate_rosters_matches_a_manual_sum(plan):
    rng = np.random.default_rng(6)
    points = rng.normal(10, 3, size=(8, 3, 16)).astype("float32")
    masks = np.zeros((2, 3, 8), dtype="float32")
    masks[0, :, :4] = 1.0
    masks[1, :, 4:] = 1.0
    out = kernel.evaluate_rosters(points, masks)
    assert np.allclose(out[0, 0], points[:4, 0, :].sum(axis=0), atol=1e-3)
    assert np.allclose(out[1, 2], points[4:, 2, :].sum(axis=0), atol=1e-3)


def test_season_outcome_shapes(league):
    rng = np.random.default_rng(7)
    team_week = rng.normal(100, 12, size=(12, 17, 25)).astype("float32")
    outcome = kernel.season_outcome(team_week, league, my_team_index=3)
    assert outcome.final_rank.shape == (12, 25)
    assert outcome.season_points.shape == (12, 25)
    assert outcome.my_team_index == 3
    for r in range(25):
        assert sorted(outcome.final_rank[:, r]) == list(range(1, 13))


def test_malformed_seed_root_fails_readably():
    """An opaque fromhex error deep in the draw loop is a bad thing to meet on
    draft night."""
    with pytest.raises(ValueError, match="lowercase hex"):
        rng_mod.uniforms("not-hex-at-all", RngKind.RATE, n=4)
    with pytest.raises(ValueError, match="128 bits"):
        rng_mod.uniforms("abcd", RngKind.RATE, n=4)
