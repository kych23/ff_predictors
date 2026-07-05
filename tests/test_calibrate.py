"""Tests for IntervalCalibrator and assign_bucket."""
import numpy as np
import pandas as pd
import pytest

from src.projection.calibrate import (
    GLOBAL,
    IntervalCalibrator,
    assign_bucket,
    calibrate_oof_lofo,
)


def test_assign_bucket_rookie():
    assert assign_bucket(pd.Series({"is_rookie": True})) == "rookie"


def test_assign_bucket_second_year():
    row = pd.Series({"is_rookie": False, "seasons_of_history": 1})
    assert assign_bucket(row) == "second_year"


def test_assign_bucket_team_changed_vet():
    row = pd.Series({"is_rookie": False, "seasons_of_history": 5, "team_changed": 1})
    assert assign_bucket(row) == "team_changed_vet"


def test_assign_bucket_established_vet():
    row = pd.Series({"is_rookie": False, "seasons_of_history": 5, "team_changed": 0})
    assert assign_bucket(row) == "established_vet"


def test_assign_bucket_priority_rookie_over_team_changed():
    row = pd.Series({"is_rookie": True, "seasons_of_history": 0, "team_changed": 1})
    assert assign_bucket(row) == "rookie"


def _make_oof(n: int = 100, noise: float = 3.0) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    p50 = rng.uniform(5, 20, n)
    hw = rng.uniform(2, 6, n)
    y = p50 + rng.normal(0, noise, n)
    buckets = rng.choice(["established_vet", "rookie"], size=n, p=[0.7, 0.3])
    return pd.DataFrame({
        "p10": p50 - hw, "p50": p50, "p90": p50 + hw,
        "y": y, "bucket": buckets,
    })


def test_calibrator_fit_produces_global():
    oof = _make_oof()
    cal = IntervalCalibrator(min_bucket_n=30).fit(oof)
    assert GLOBAL in cal.scales


def test_calibrator_small_bucket_falls_back():
    oof = _make_oof(50)
    oof["bucket"] = ["rare_type"] * 5 + ["established_vet"] * 45
    cal = IntervalCalibrator(min_bucket_n=30).fit(oof)
    assert "rare_type" not in cal.scales
    assert GLOBAL in cal.scales


def test_calibrator_scales_never_shrink():
    oof = _make_oof(200)
    cal = IntervalCalibrator(min_bucket_n=30).fit(oof)
    for ls, us in cal.scales.values():
        assert ls >= 1.0
        assert us >= 1.0


def test_calibrator_transform_preserves_monotonicity():
    oof = _make_oof(100)
    cal = IntervalCalibrator(min_bucket_n=30).fit(oof)
    preds = oof[["p10", "p50", "p90"]].copy()
    out = cal.transform(preds, oof["bucket"])
    assert (out["p10"] <= out["p50"]).all()
    assert (out["p50"] <= out["p90"]).all()


def test_calibrator_widens_underconfident_intervals():
    oof = _make_oof(100, noise=10.0)
    cal = IntervalCalibrator(min_bucket_n=30).fit(oof)
    ls, us = cal.scales[GLOBAL]
    assert ls > 1.0 or us > 1.0


def _two_season_oof(noise_a: float, noise_b: float, n: int = 120) -> pd.DataFrame:
    rng = np.random.RandomState(7)
    frames = []
    for season, noise in ((2023, noise_a), (2024, noise_b)):
        p50 = rng.uniform(5, 20, n)
        frames.append(pd.DataFrame({
            "season": season,
            "p10": p50 - 4.0, "p50": p50, "p90": p50 + 4.0,
            "y": p50 + rng.normal(0, noise, n),
            "bucket": "established_vet",
        }))
    return pd.concat(frames, ignore_index=True)


def test_lofo_uses_other_seasons_residuals():
    """Season with huge residuals gets widened per the OTHER season's (tight)
    residuals — not its own — so its intervals stay near raw."""
    oof = _two_season_oof(noise_a=15.0, noise_b=0.5)
    out = calibrate_oof_lofo(oof, min_bucket_n=30)

    noisy = oof["season"] == 2023
    raw_width = (oof.loc[noisy, "p90"] - oof.loc[noisy, "p10"])
    lofo_width = (out.loc[noisy, "p90"] - out.loc[noisy, "p10"])
    # calibrator for 2023 was fit on tight 2024 residuals -> barely widens
    assert (lofo_width <= raw_width * 1.5).all()

    # fit-on-self would have widened the noisy season substantially more
    self_cal = IntervalCalibrator(min_bucket_n=30).fit(oof[noisy])
    self_out = self_cal.transform(oof.loc[noisy, ["p10", "p50", "p90"]],
                                  oof.loc[noisy, "bucket"])
    self_width = self_out["p90"] - self_out["p10"]
    assert self_width.mean() > lofo_width.mean()


def test_lofo_preserves_monotonicity():
    oof = _two_season_oof(noise_a=8.0, noise_b=2.0)
    out = calibrate_oof_lofo(oof, min_bucket_n=30)
    assert (out["p10"] <= out["p50"]).all()
    assert (out["p50"] <= out["p90"]).all()


def test_lofo_single_season_falls_back_to_self():
    oof = _two_season_oof(noise_a=5.0, noise_b=5.0)
    oof = oof[oof["season"] == 2023].reset_index(drop=True)
    out = calibrate_oof_lofo(oof, min_bucket_n=30)
    cal = IntervalCalibrator(min_bucket_n=30).fit(oof)
    expected = cal.transform(oof[["p10", "p50", "p90"]], oof["bucket"])
    pd.testing.assert_frame_equal(out[["p10", "p50", "p90"]], expected)


def test_lofo_does_not_mutate_input():
    oof = _two_season_oof(noise_a=8.0, noise_b=2.0)
    before = oof.copy()
    calibrate_oof_lofo(oof, min_bucket_n=30)
    pd.testing.assert_frame_equal(oof, before)
