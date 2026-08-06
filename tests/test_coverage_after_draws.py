"""§21.2 — the calibrated quantiles survive being drawn from.

The projection model is conformally calibrated so that P10/P90 bracket 80% of
outcomes. That guarantee is about the *quantiles*; the simulator does not
consume quantiles, it consumes draws. §15.2 decided not to recentre the
split-normal on its own mean precisely so the draws keep the calibration, and
this is the test that makes that decision checkable rather than merely stated.

The spec: over 200 players and 20,000 draws of `rate`, the empirical fraction
below each player's `p10` is within 0.01 of 0.10, and below `p90` within 0.01
of 0.90.

If someone "fixes" the asymmetry of the split normal by recentring it, this
fails — which is the point. The density is deliberately not mean-centred,
because matching the three calibrated quantiles is the contract and matching
the mean is not.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.constants import Z90
from src.engine.sim.draws import (
    ProjectionBundle, draw_rates, split_normal_moments, split_normal_ppf,
)

N_PLAYERS = 200
N_DRAWS = 20_000
ROOT = "3dc0676bb0114bd3bb66e8d97c210fbb"
TOL = 0.01


@pytest.fixture(scope="module")
def bundle():
    """200 players spanning the real range of asymmetry, including the
    degenerate one-sided case."""
    rng = np.random.default_rng(7)
    p50 = rng.uniform(4.0, 22.0, N_PLAYERS)
    lo_width = rng.uniform(0.15, 0.60, N_PLAYERS) * p50
    hi_width = rng.uniform(0.15, 0.90, N_PLAYERS) * p50
    lo_width[0] = 0.0                       # p10 == p50: one-sided
    p10 = np.maximum(p50 - lo_width, 0.0)
    p90 = p50 + hi_width
    return ProjectionBundle(
        player_ids=np.array([f"p{i}" for i in range(N_PLAYERS)]),
        positions=np.array(["RB"] * N_PLAYERS),
        nfl_teams=np.array([f"T{i % 32}" for i in range(N_PLAYERS)]),
        slot_idx=np.zeros(N_PLAYERS, dtype=int),
        bye_weeks=np.zeros(N_PLAYERS, dtype=int),
        is_kdst=np.zeros(N_PLAYERS, dtype=bool),
        rate_p10=p10, rate_p50=p50, rate_p90=p90,
        weekly_sigma=p50 * 0.4,
        games_hazard=np.ones((N_PLAYERS, 17)),
        kdst_quantiles=np.full((N_PLAYERS, 101), np.nan),
    )


@pytest.fixture(scope="module")
def rates(bundle):
    return draw_rates(bundle, ROOT, N_DRAWS)      # (P, R)


@pytest.fixture(scope="module")
def proper(bundle):
    """Rows with a non-degenerate lower half.

    Player 0 has `p10 == p50`, so half its mass sits exactly ON p50 and none
    below it: `P(X < p10)` is 0 and `P(X < p50)` is 0 rather than 0.10 and
    0.50. That is correct behaviour for a one-sided draw, not a coverage
    failure, and it is asserted directly in
    `test_a_degenerate_lower_half_is_a_safe_one_sided_draw`. Including it in
    a strict-inequality coverage statistic would make this file fail for a
    reason that has nothing to do with calibration.
    """
    return bundle.rate_p10 < bundle.rate_p50


@pytest.mark.slow
def test_p10_coverage_holds_after_drawing(bundle, rates, proper):
    below = (rates < bundle.rate_p10[:, None]).mean(axis=1)[proper]
    worst = float(np.abs(below - 0.10).max())
    assert worst <= TOL, (
        f"worst |empirical - 0.10| = {worst:.4f} over {proper.sum()} players; "
        f"the drawn distribution no longer matches the calibrated P10"
    )


@pytest.mark.slow
def test_p90_coverage_holds_after_drawing(bundle, rates):
    below = (rates < bundle.rate_p90[:, None]).mean(axis=1)
    worst = float(np.abs(below - 0.90).max())
    assert worst <= TOL, f"worst |empirical - 0.90| = {worst:.4f}"


@pytest.mark.slow
def test_the_median_is_the_median(bundle, rates, proper):
    """u < 0.5 takes the lower branch and u >= 0.5 the upper, so exactly half
    the mass sits either side of p50 by construction. Recentring on the mean
    would break this first."""
    below = (rates < bundle.rate_p50[:, None]).mean(axis=1)[proper]
    assert float(np.abs(below - 0.50).max()) <= TOL


@pytest.mark.slow
def test_the_central_interval_carries_eighty_percent(bundle, rates, proper):
    inside = ((rates >= bundle.rate_p10[:, None])
              & (rates <= bundle.rate_p90[:, None])).mean(axis=1)
    assert float(np.abs(inside[proper] - 0.80).max()) <= 2 * TOL
    # The degenerate row carries 90%, not 80%, and that is right: its lower
    # half is a point mass sitting ON p10, so the closed interval [p10, p90]
    # contains the entire lower half rather than 40% of it.
    assert float(inside[~proper].min()) > 0.85


# ------------------------------------------------ why no recentring (§15.2)
def test_the_draws_are_deliberately_not_mean_centred(bundle, rates):
    """The mean of an asymmetric split normal is NOT p50, and that is correct.
    Someone tempted to 'fix' the offset would restore the mean and destroy the
    P10/P90 coverage the tests above assert — this pins the tradeoff so the
    decision has to be made deliberately."""
    expected_mean, variance = split_normal_moments(
        bundle.rate_p10, bundle.rate_p50, bundle.rate_p90)
    realized = rates.mean(axis=1)
    # Tolerance in units of the sample mean's OWN standard error, not a flat
    # constant: these players span p50 from 4 to 22, so a fixed atol is
    # simultaneously too tight for the volatile ones and too loose for the rest.
    se = np.sqrt(variance / N_DRAWS)
    assert np.all(np.abs(realized - expected_mean) < 5 * se)

    skewed = bundle.rate_p90 - bundle.rate_p50 > 1.5 * (
        bundle.rate_p50 - bundle.rate_p10)
    assert skewed.any(), "fixture has no strongly skewed players"
    assert np.all(realized[skewed] > bundle.rate_p50[skewed]), (
        "a right-skewed player's mean must sit above his median"
    )


def test_the_moment_formula_uses_sqrt_two_pi_not_sqrt_two_over_pi():
    """The classical split normal carries sqrt(2/pi); this density is the
    equal-weight two-piece form and carries 1/sqrt(2*pi). An earlier revision
    shipped the classical constant, which is wrong by a factor of ~2.5."""
    p10, p50, p90 = np.array([5.0]), np.array([10.0]), np.array([20.0])
    mean, _ = split_normal_moments(p10, p50, p90)
    sigma_lo = (10.0 - 5.0) / Z90
    sigma_hi = (20.0 - 10.0) / Z90
    assert mean[0] == pytest.approx(
        10.0 + (sigma_hi - sigma_lo) / np.sqrt(2 * np.pi))
    assert mean[0] != pytest.approx(
        10.0 + (sigma_hi - sigma_lo) * np.sqrt(2 / np.pi))


# ------------------------------------------------------------- edge cases
def test_a_degenerate_lower_half_is_a_safe_one_sided_draw():
    """p10 == p50 gives sigma_lo = 0. Half the draws land exactly on p50 and
    none below it — safe, not a crash, and not negative."""
    u = np.linspace(1e-6, 1 - 1e-6, 10_000)
    x = split_normal_ppf(u, np.full(10_000, 10.0), np.full(10_000, 10.0),
                         np.full(10_000, 18.0))
    assert x.min() == pytest.approx(10.0)
    assert (x < 10.0).sum() == 0
    assert abs((x == 10.0).mean() - 0.5) < 0.01


def test_crossed_quantiles_clamp_rather_than_invert():
    """Monotonicity is enforced upstream by rearrangement, but the ppf must not
    produce an inverted draw if a crossed triple ever reaches it."""
    u = np.linspace(0.01, 0.99, 999)
    x = split_normal_ppf(u, np.full(999, 12.0), np.full(999, 10.0),
                         np.full(999, 8.0))
    assert np.all(x == pytest.approx(10.0))


@pytest.mark.slow
def test_extending_a_draw_reuses_its_prefix(bundle):
    """Inverse-CDF, never Ziggurat (§15.1). Ziggurat consumes a variable number
    of uniforms per normal, so lengthening a draw would reshuffle every value
    before it and break CRN across candidates."""
    short = draw_rates(bundle, ROOT, 500)
    long = draw_rates(bundle, ROOT, 2000)
    assert np.allclose(short, long[:, :500])
