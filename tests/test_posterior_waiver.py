"""Correlation posterior and waiver floor (§13.2, §15.6)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.engine.sim.waiver import apply_waiver
from src.models.correlation.slot_matrix import (
    DEFAULT_SLOTS, bootstrap_posterior,
)


def _panel(rng, n_seasons=6, n_teams=16, weeks=14):
    slots = list(DEFAULT_SLOTS)
    target = np.eye(len(slots))
    for a, b, r in [("QB1", "WR1", 0.38), ("WR1", "WR2", 0.04),
                    ("RB1", "RB2", -0.13)]:
        i, j = slots.index(a), slots.index(b)
        target[i, j] = target[j, i] = r
    chol = np.linalg.cholesky(target)
    rows = []
    for s in range(n_seasons):
        for t in range(n_teams):
            draws = rng.standard_normal((weeks, len(slots))) @ chol.T
            for w in range(weeks):
                for k, slot in enumerate(slots):
                    rows.append({"season": 2018 + s, "week": w + 1,
                                 "team": f"T{t}", "slot": slot,
                                 "points": 10 + 5 * draws[w, k]})
    return pd.DataFrame(rows)


# ------------------------------------------------- correlation posterior
def test_posterior_produces_distinct_draws():
    """Without real theta draws, epistemic_se measures rollout variation
    rather than parameter uncertainty (§15.5)."""
    post = bootstrap_posterior(_panel(np.random.default_rng(0)), draws=12)
    assert len(post) >= 8
    firsts = [d.correlation("QB1", "WR1") for d in post.draws]
    assert np.std(firsts) > 0, "bootstrap draws must actually differ"


def test_posterior_brackets_the_point_estimate():
    post = bootstrap_posterior(_panel(np.random.default_rng(1)), draws=20)
    point = post.point.correlation("QB1", "WR1")
    draws = [d.correlation("QB1", "WR1") for d in post.draws]
    assert min(draws) <= point <= max(draws)


def test_every_draw_is_psd_and_factorable():
    post = bootstrap_posterior(_panel(np.random.default_rng(2)), draws=15)
    for d in post.draws:
        assert d.min_eigenvalue > 0
        assert np.allclose(d.cholesky @ d.cholesky.T, d.matrix, atol=1e-8)


def test_draw_index_wraps():
    post = bootstrap_posterior(_panel(np.random.default_rng(3)), draws=5)
    assert post.draw(0) is post.draw(len(post))


def test_spread_reports_parameter_uncertainty():
    post = bootstrap_posterior(_panel(np.random.default_rng(4)), draws=20)
    assert post.spread("QB1", "WR1") > 0.0
    assert post.spread("QB1", "WR1") < 0.2, "resamples should not be wild"


# ------------------------------------------------------------- waiver
class _Bundle:
    def __init__(self, n, rate):
        self.positions = np.array(["RB"] * n)
        self.rate_p50 = rate
        self.n_players = n


def _setup(n=6, weeks=14, bust_idx=3, pool_rate=9.0, starter_rate=12.0):
    rate = np.concatenate([np.full(4, starter_rate), np.full(n - 4, pool_rate)])
    bundle = _Bundle(n, rate)
    points = np.zeros((n, weeks, 1), dtype="float32")
    points[:4] = starter_rate
    if bust_idx is not None:
        points[bust_idx] = 0.5
    points[4:] = pool_rate
    masks = np.zeros((1, weeks, n), dtype="float32")
    masks[0, :, :4] = 1.0
    team_week = np.einsum("twp,pwr->twr", masks, points)
    return points, masks, team_week, bundle


def test_waiver_swaps_a_persistent_underperformer():
    """A bust must cost real points before it is replaced — the detection lag
    is what makes fliers distinguishable from each other (§8.6)."""
    points, masks, team_week, bundle = _setup()
    out = apply_waiver(points, masks, team_week, [np.arange(4)],
                       np.array([4, 5]), bundle, detection_lag_weeks=2,
                       regular_season_weeks=14)
    assert out.n_swaps >= 1
    assert out.team_week[0, -1, 0] > team_week[0, -1, 0], (
        "replacing a 0.5-point starter with a 9-point pool player must help"
    )


def test_waiver_is_not_a_max_over_realized():
    """max(player, replacement) would assume perfect timing and no cost. With
    a lag, the bust's early weeks are still paid for."""
    points, masks, team_week, bundle = _setup()
    out = apply_waiver(points, masks, team_week, [np.arange(4)],
                       np.array([4, 5]), bundle, detection_lag_weeks=2,
                       regular_season_weeks=14)
    clairvoyant = 3 * 12.0 + 9.0
    assert out.team_week[0, 0, 0] < clairvoyant, (
        "week 1 cannot already reflect a swap that needs 2 weeks of evidence"
    )


def test_waiver_leaves_a_healthy_roster_alone():
    points, masks, team_week, bundle = _setup(bust_idx=None, pool_rate=6.0,
                                              starter_rate=14.0)
    out = apply_waiver(points, masks, team_week, [np.arange(4)],
                       np.array([4, 5]), bundle, regular_season_weeks=14)
    assert out.n_swaps == 0
    assert np.allclose(out.team_week, team_week)


def test_pool_player_cannot_be_claimed_twice():
    n, weeks = 8, 14
    bundle = _Bundle(n, np.concatenate([np.full(6, 12.0), np.full(2, 9.0)]))
    points = np.zeros((n, weeks, 1), dtype="float32")
    points[:6] = 12.0
    points[2] = 0.5
    points[5] = 0.5
    points[6:] = 9.0
    masks = np.zeros((2, weeks, n), dtype="float32")
    masks[0, :, :3] = 1.0
    masks[1, :, 3:6] = 1.0
    team_week = np.einsum("twp,pwr->twr", masks, points)
    out = apply_waiver(points, masks, team_week,
                       [np.arange(3), np.arange(3, 6)],
                       np.array([6, 7]), bundle, regular_season_weeks=weeks)
    assert out.n_swaps <= 2, "only two pool players exist"


def test_ros_averages_over_games_played_not_weeks_elapsed():
    """Byes and inactive weeks are drawn as exact zeros. Dividing by weeks
    ELAPSED biases the rest-of-season estimate down by roughly the inactive
    rate and fires drops on players who were merely on bye — measured on the
    real board that flipped the waiver's net effect from +1761 to -3071 points.
    Per-game rates are availability-neutral; games missed belong to the hazard.
    """
    n, weeks = 6, 14
    # a genuinely good starter who simply misses weeks 1-3 (bye/injury zeros)
    bundle = _Bundle(n, np.array([12.0, 12.0, 12.0, 12.0, 4.0, 4.0]))
    points = np.zeros((n, weeks, 1), dtype="float32")
    points[:4] = 12.0
    points[3, :3] = 0.0            # inactive, not bad
    points[4:] = 4.0
    masks = np.zeros((1, weeks, n), dtype="float32")
    masks[0, :, :4] = 1.0
    team_week = np.einsum("twp,pwr->twr", masks, points)

    out = apply_waiver(points, masks, team_week, [np.arange(4)],
                       np.array([4, 5]), bundle, detection_lag_weeks=2,
                       regular_season_weeks=weeks)
    assert out.n_swaps == 0, (
        "a 12-point player who missed three weeks must not be dropped for a "
        "4-point replacement"
    )
