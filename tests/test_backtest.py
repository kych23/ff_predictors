"""Objective backtest (§21.6).

The statistics here have to be right in a specific way: when the simulator IS
calibrated they must say so, and when it is miscalibrated they must say WHICH
WAY. A PIT test that only reports a p-value is not enough — "not uniform" does
not tell you whether to widen the intervals, narrow them, or move them.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.engine.audit.backtest import (
    BacktestReport, pit_test, pit_values, weekly_high_coverage,
)

RNG = np.random.default_rng(0)


def _draws(n=800, reps=2000, loc=0.0, scale=1.0):
    return RNG.normal(loc, scale, (n, reps))


# ------------------------------------------------------------------ PIT
def test_a_calibrated_forecast_is_uniform():
    sim = _draws()
    realized = RNG.normal(0.0, 1.0, sim.shape[0])
    result = pit_test(sim, realized)
    assert result.uniform, f"p={result.p_value}"
    assert result.shape == "uniform"


def test_overconfident_intervals_read_as_u_shaped():
    """Too narrow: realized values land in the tails far too often."""
    sim = _draws(scale=0.4)
    realized = RNG.normal(0.0, 1.0, sim.shape[0])
    result = pit_test(sim, realized)
    assert not result.uniform
    assert "TOO NARROW" in result.shape


def test_underconfident_intervals_read_as_inverted_u():
    sim = _draws(scale=3.0)
    realized = RNG.normal(0.0, 1.0, sim.shape[0])
    result = pit_test(sim, realized)
    assert not result.uniform
    assert "TOO WIDE" in result.shape


def test_a_forecast_biased_high_slopes_down():
    """The real backtest's failure mode. Predictions centred above the truth
    put realized values low in the forecast, so the PIT histogram declines."""
    sim = _draws(loc=1.0)
    realized = RNG.normal(0.0, 1.0, sim.shape[0])
    result = pit_test(sim, realized)
    assert not result.uniform
    assert "biased HIGH" in result.shape
    assert result.slope < 0
    assert result.mean < 0.5


def test_a_forecast_biased_low_slopes_up():
    sim = _draws(loc=-1.0)
    realized = RNG.normal(0.0, 1.0, sim.shape[0])
    result = pit_test(sim, realized)
    assert "biased LOW" in result.shape
    assert result.slope > 0


def test_a_gentle_tilt_is_caught_by_slope_where_the_mean_alone_is_not():
    """Measured on real data: mean PIT 0.452 with a histogram falling from
    ratio 1.28 to 0.75. Every threshold written in terms of edge and centre
    mass passed it, and the first version of this classifier reported
    'no simple shape'."""
    sim = _draws(loc=0.15)
    realized = RNG.normal(0.0, 1.0, sim.shape[0])
    result = pit_test(sim, realized)
    assert 0.40 < result.mean < 0.50, "the mean alone is barely moved"
    assert result.slope < -0.15
    assert "biased HIGH" in result.shape


def test_ties_do_not_manufacture_a_failure():
    """K/DST draws come from an empirical CDF, so exact ties are common. Without
    the half-mass correction they pile PIT onto grid points and fail KS for a
    reason unrelated to calibration."""
    sim = np.tile(np.arange(11, dtype=float), (500, 1))
    realized = RNG.integers(0, 11, 500).astype(float)
    values = pit_values(sim, realized)
    assert np.all((values > 0.0) & (values < 1.0))


def test_shape_mismatches_are_named_not_broadcast():
    with pytest.raises(ValueError, match="realized values against"):
        pit_values(np.zeros((5, 10)), np.zeros(4))
    with pytest.raises(ValueError, match=r"\(N, R\)"):
        pit_values(np.zeros(5), np.zeros(5))


# ---------------------------------------------------------- weekly high
def _team_week(n_teams=12, weeks=14, reps=500, edge=0.0):
    base = RNG.normal(100.0, 15.0, (n_teams, weeks, reps))
    base[0] += edge
    return base


def test_a_dominant_roster_is_predicted_to_win_more():
    strong = weekly_high_coverage(_team_week(edge=40.0),
                                  _team_week(edge=40.0)[:, :, 0], 0, 2024)
    even = weekly_high_coverage(_team_week(), _team_week()[:, :, 0], 0, 2024)
    assert strong.predicted_mean > even.predicted_mean


def test_the_interval_is_over_the_season_rate_not_the_week():
    """A per-week interval is a Bernoulli whose 90% interval is {0, 1}. It
    contains everything and tests nothing."""
    sim = _team_week()
    result = weekly_high_coverage(sim, sim[:, :, 0], 0, 2024)
    assert result.predicted_hi < 1.0
    assert result.predicted_hi > result.predicted_lo


def test_a_realized_rate_inside_the_interval_is_reported_inside():
    sim = _team_week()
    result = weekly_high_coverage(sim, sim[:, :, 0], 0, 2024)
    assert result.inside


def test_a_roster_that_never_wins_when_predicted_to_often_is_outside():
    sim = _team_week(edge=60.0)
    realized = sim[:, :, 0].copy()
    realized[0] = -1e6                     # team 0 never tops a week
    result = weekly_high_coverage(sim, realized, 0, 2024)
    assert result.realized_rate == 0.0
    assert not result.inside


def test_mismatched_realized_shape_is_rejected():
    with pytest.raises(ValueError, match="does not match"):
        weekly_high_coverage(_team_week(), np.zeros((3, 3)), 0, 2024)


# --------------------------------------------------------------- report
def _high(inside: bool, season=2020):
    from src.engine.audit.backtest import WeeklyHighResult
    return WeeklyHighResult(season=season, realized_rate=0.1 if inside else 0.9,
                            predicted_lo=0.0, predicted_hi=0.2,
                            predicted_mean=0.1, n_weeks=14)


def _report(pit_uniform: bool, insides: list[bool]):
    sim = _draws(loc=0.0 if pit_uniform else 3.0)
    realized = RNG.normal(0.0, 1.0, sim.shape[0])
    return BacktestReport(
        pit=pit_test(sim, realized),
        weekly_high=[_high(i, 2018 + n) for n, i in enumerate(insides)],
        seasons=list(range(2018, 2018 + len(insides))),
        instrument="test", n_roster_weeks=len(realized),
    )


def test_criterion_needs_both_arms():
    """§21.6: uniform PIT AND at least n-1 seasons inside."""
    assert _report(True, [True] * 8).passes
    assert _report(True, [True] * 7 + [False]).passes
    assert not _report(True, [True] * 6 + [False] * 2).passes
    assert not _report(False, [True] * 8).passes


def test_the_verdict_travels_with_the_report():
    """§21.5 rates this descriptive_only. A reader of the artifact must not be
    able to separate the number from its licence."""
    report = _report(True, [True] * 8)
    assert report.verdict == "descriptive_only"
    assert "descriptive_only" in report.describe()


def test_describe_names_the_instrument():
    assert "test" in _report(True, [True]).describe()
