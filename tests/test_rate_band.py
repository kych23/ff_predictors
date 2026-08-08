"""The simulator draws from the CALIBRATED quantiles, not a synthesized band.

`draws.py` states the contract: ``rate ~ SplitNormal(p10, p50, p90)``. The
quantile model fits those three numbers and `calibrate.py` conformally widens
them, which is the entire reason both modules exist. But `build_bundle`
renamed ``p50 -> value`` and dropped ``p10``/``p90``, and `bundle_build` then
rebuilt a band as ``value +/- 1.28 * sigma_weekly / sqrt(17)``.

That is a different quantity. ``sigma_weekly / sqrt(17)`` is the standard error
of a season MEAN under weekly sampling noise — how precisely a known rate could
be measured — not the model's uncertainty about the rate. Measured over the 206
matched players on the live bundle:

* median band width 8.817 pts/g calibrated against 4.069 synthesized, so rate
  uncertainty was **2.3x too narrow**;
* calibrated skew spans -5.24 .. +5.86 and exceeds 0.5 for **88.3%** of
  players, while the synthesized band was symmetric to 1.8e-15 for every one —
  `split_normal_ppf` degenerated to a plain normal on every single draw.

The conformal calibration work upstream had no effect downstream at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.engine.sim.bundle_build import _rate_band


def _frame(**columns) -> pd.DataFrame:
    base = {"player_id": ["a", "b", "c"], "position": ["RB", "WR", "QB"]}
    base.update(columns)
    return pd.DataFrame(base)


# ------------------------------------------------------ the calibrated band
def test_calibrated_quantiles_are_used_verbatim():
    value = np.array([10.0, 12.0, 18.0])
    df = _frame(value_p10=[6.0, 9.0, 14.0], value_p90=[17.0, 13.0, 25.0])
    p10, p90, source = _rate_band(df, value, sigma=np.array([7.0, 8.0, 9.0]))

    assert p10.tolist() == [6.0, 9.0, 14.0]
    assert p90.tolist() == [17.0, 13.0, 25.0]
    assert "calibrated for 3/3" in source


def test_asymmetry_survives():
    """The property the synthesized band destroyed. A split normal given equal
    half-widths IS a normal, so preserving skew is the whole point."""
    value = np.array([10.0])
    df = _frame(player_id=["a"], position=["RB"],
                value_p10=[8.0], value_p90=[20.0])
    p10, p90, _ = _rate_band(df, value, sigma=np.array([7.0]))
    lower, upper = value[0] - p10[0], p90[0] - value[0]
    assert upper == pytest.approx(5.0 * lower)


def test_the_band_is_not_derived_from_weekly_sigma():
    """Same calibrated quantiles, wildly different weekly sigma. If sigma still
    leaks into the band, this moves."""
    value = np.array([10.0])
    df = _frame(player_id=["a"], position=["RB"],
                value_p10=[7.0], value_p90=[14.0])
    a = _rate_band(df, value, sigma=np.array([2.0]))
    b = _rate_band(df, value, sigma=np.array([40.0]))
    assert a[0].tolist() == b[0].tolist()
    assert a[1].tolist() == b[1].tolist()


# ------------------------------------------------------------- the fallback
def test_rows_without_a_band_fall_back_and_are_counted():
    value = np.array([10.0, 8.0])
    df = _frame(player_id=["a", "b"], position=["K", "DST"],
                value_p10=[np.nan, np.nan], value_p90=[np.nan, np.nan])
    p10, p90, source = _rate_band(df, value, sigma=np.array([3.0, 3.0]))
    assert (p10 < value).all() and (p90 > value).all()
    assert "fallback for 2" in source


def test_a_bundle_with_no_quantile_columns_still_builds():
    """Older bundles predate the columns; the engine must not require them."""
    value = np.array([10.0, 8.0])
    df = _frame(player_id=["a", "b"], position=["RB", "WR"])
    p10, p90, source = _rate_band(df, value, sigma=np.array([7.0, 7.0]))
    assert np.isfinite(p10).all() and np.isfinite(p90).all()
    assert "fallback for 2" in source


def test_a_partially_calibrated_bundle_mixes_per_row():
    value = np.array([10.0, 8.0])
    df = _frame(player_id=["a", "b"], position=["RB", "K"],
                value_p10=[6.0, np.nan], value_p90=[17.0, np.nan])
    p10, p90, source = _rate_band(df, value, sigma=np.array([7.0, 3.0]))
    assert (p10[0], p90[0]) == (6.0, 17.0)
    assert np.isfinite(p10[1]) and np.isfinite(p90[1])
    assert "calibrated for 1/2" in source


# ------------------------------------------------ invariants the draw needs
def test_the_band_always_brackets_the_median():
    """K/DST and floored rookies have their `value` replaced downstream of the
    quantile model, so a stale band can sit entirely on one side of it.
    `split_normal_ppf` cannot represent p50 outside [p10, p90]."""
    value = np.array([30.0, 1.0])
    df = _frame(player_id=["a", "b"], position=["DST", "RB"],
                value_p10=[5.0, 6.0], value_p90=[9.0, 17.0])
    p10, p90, _ = _rate_band(df, value, sigma=np.array([3.0, 7.0]))
    assert (p10 <= value).all() and (value <= p90).all()


def test_the_lower_bound_is_never_negative():
    value = np.array([2.0])
    df = _frame(player_id=["a"], position=["RB"],
                value_p10=[-5.0], value_p90=[9.0])
    p10, _, _ = _rate_band(df, value, sigma=np.array([7.0]))
    assert p10[0] >= 0.0


def test_an_inverted_band_is_rejected_rather_than_used():
    value = np.array([10.0])
    df = _frame(player_id=["a"], position=["RB"],
                value_p10=[15.0], value_p90=[5.0])
    p10, p90, source = _rate_band(df, value, sigma=np.array([7.0]))
    assert p10[0] <= value[0] <= p90[0]
    assert "fallback for 1" in source


# ------------------------------------------------------ end to end, live
def test_the_shipped_bundle_carries_the_calibrated_band():
    from src.core.config import load_league
    from src.engine.decision.board import Board

    board = Board.from_bundle()
    if "value_p10" not in board.players.columns:
        pytest.skip("bundle predates value_p10/value_p90; rebuild it")

    cfg = load_league()
    weeks = cfg.schedule.regular_season_weeks + len(cfg.schedule.playoff_weeks)
    frame = board.players.reset_index(drop=True).copy()
    frame["player_id"] = frame["player_id"].astype(str)

    from src.engine.sim.bundle_build import build_projection_bundle
    bundle = build_projection_bundle(frame, cfg, weeks)

    modeled = np.isin(np.array(bundle.positions), ["QB", "RB", "WR", "TE"])
    width = (bundle.rate_p90 - bundle.rate_p10)[modeled]
    synthesized = (2 * 1.28 * bundle.weekly_sigma / np.sqrt(17))[modeled]
    assert np.median(width) > 1.8 * np.median(synthesized), (
        "the band looks like the synthesized one; the calibrated quantiles "
        "are not reaching the simulator")

    skew = ((bundle.rate_p90 - bundle.rate_p50)
            - (bundle.rate_p50 - bundle.rate_p10))[modeled]
    assert (np.abs(skew) > 0.5).mean() > 0.5, "the band lost its asymmetry"
    assert (bundle.rate_p10 <= bundle.rate_p50).all()
    assert (bundle.rate_p50 <= bundle.rate_p90).all()
    assert (bundle.rate_p10 >= 0).all()


# --------------------------------------------- the invariants hold alone
@pytest.mark.parametrize("value", [-2.0, -0.5, 0.0, 0.01, 5.0])
def test_ordering_survives_a_non_positive_value(value):
    """The zero clamp must never overtake the median.

    Clamping to zero LAST pushed p10 above a negative `value`: a [-5, -1] band
    around -2 produced ``p10 = 0 > p50 = -2``, which `split_normal_ppf` cannot
    represent. The one production caller floors `value` positive beforehand, so
    this was unreachable — but the ordering contract belongs to this function,
    not to the luck of its caller.
    """
    df = _frame(player_id=["a"], position=["RB"],
                value_p10=[value - 3.0], value_p90=[value + 3.0])
    p10, p90, _ = _rate_band(df, np.array([value]), sigma=np.array([7.0]))
    assert p10[0] <= value <= p90[0]
    assert p10[0] >= 0.0 or p10[0] == pytest.approx(value)


def test_every_emitted_band_is_drawable():
    """Whatever this returns must be a distribution `split_normal_ppf` can
    invert — including the degenerate bands the clamps can produce."""
    from src.engine.sim.draws import split_normal_ppf

    values = np.array([-2.0, 0.0, 0.5, 10.0, 30.0])
    df = _frame(player_id=list("abcde"), position=["RB"] * 5,
                value_p10=[-5.0, 0.0, 0.5, 6.0, 5.0],
                value_p90=[-1.0, 0.0, 0.5, 17.0, 9.0])
    p10, p90, _ = _rate_band(df, values, sigma=np.full(5, 7.0))

    u = np.linspace(0.001, 0.999, 25)
    for i in range(len(values)):
        drawn = split_normal_ppf(u, p10[i], values[i], p90[i])
        assert np.isfinite(drawn).all()
        assert np.all(np.diff(drawn) >= -1e-9), "ppf must be monotone in u"
