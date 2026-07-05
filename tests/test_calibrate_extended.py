"""Black-box extensions for IntervalCalibrator.

Contract (from test_calibrate.py): scales never shrink below 1.0, per-bucket
scaling with GLOBAL fallback for thin buckets, monotone output.
"""
import numpy as np
import pandas as pd

from src.projection.calibrate import GLOBAL, IntervalCalibrator, assign_bucket


def _make_oof(n: int = 200, noise: float = 3.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    p50 = rng.uniform(5, 20, n)
    hw = rng.uniform(2, 6, n)
    y = p50 + rng.normal(0, noise, n)
    buckets = rng.choice(["established_vet", "rookie"], size=n, p=[0.7, 0.3])
    return pd.DataFrame({
        "p10": p50 - hw, "p50": p50, "p90": p50 + hw,
        "y": y, "bucket": buckets,
    })


def test_transform_never_narrows_intervals():
    """Scales >= 1.0 implies transformed intervals are at least as wide as raw."""
    oof = _make_oof(300, noise=8.0)
    cal = IntervalCalibrator(min_bucket_n=30).fit(oof)
    out = cal.transform(oof[["p10", "p50", "p90"]], oof["bucket"])
    raw_width = (oof["p90"] - oof["p10"]).to_numpy()
    new_width = (out["p90"] - out["p10"]).to_numpy()
    assert (new_width >= raw_width - 1e-9).all()


def test_transform_preserves_p50():
    """Calibration scales interval half-widths around the median; the point
    estimate itself must not move."""
    oof = _make_oof(200, noise=6.0)
    cal = IntervalCalibrator(min_bucket_n=30).fit(oof)
    out = cal.transform(oof[["p10", "p50", "p90"]], oof["bucket"])
    assert np.allclose(out["p50"].to_numpy(), oof["p50"].to_numpy())


def test_transform_output_row_count():
    oof = _make_oof(150)
    cal = IntervalCalibrator(min_bucket_n=30).fit(oof)
    out = cal.transform(oof[["p10", "p50", "p90"]], oof["bucket"])
    assert len(out) == len(oof)


def test_transform_handles_thin_bucket_via_fallback():
    """A bucket too small to earn its own scale at fit time must still be
    transformable (falls back to GLOBAL), not KeyError."""
    oof = _make_oof(100)
    oof.loc[oof.index[:5], "bucket"] = "rare_type"
    cal = IntervalCalibrator(min_bucket_n=30).fit(oof)
    assert "rare_type" not in cal.scales
    out = cal.transform(oof[["p10", "p50", "p90"]], oof["bucket"])
    assert len(out) == len(oof)
    assert (out["p10"] <= out["p50"]).all()
    assert (out["p50"] <= out["p90"]).all()


def test_well_calibrated_intervals_barely_widen():
    """When p10/p90 already sit at the true 10th/90th percentiles, the fitted
    scales should stay near 1 (they can never go below 1)."""
    rng = np.random.RandomState(5)
    n = 2000
    sigma = 3.0
    p50 = rng.uniform(5, 20, n)
    z90 = 1.2816
    oof = pd.DataFrame({
        "p10": p50 - z90 * sigma, "p50": p50, "p90": p50 + z90 * sigma,
        "y": p50 + rng.normal(0, sigma, n),
        "bucket": "established_vet",
    })
    cal = IntervalCalibrator(min_bucket_n=30).fit(oof)
    ls, us = cal.scales[GLOBAL]
    assert 1.0 <= ls < 1.5
    assert 1.0 <= us < 1.5


def test_noisier_data_earns_larger_scales():
    def fit_scales(noise):
        rng = np.random.RandomState(9)
        n = 800
        p50 = rng.uniform(5, 20, n)
        oof = pd.DataFrame({
            "p10": p50 - 4.0, "p50": p50, "p90": p50 + 4.0,
            "y": p50 + rng.normal(0, noise, n),
            "bucket": "established_vet",
        })
        return IntervalCalibrator(min_bucket_n=30).fit(oof).scales[GLOBAL]

    tight = fit_scales(1.0)
    loose = fit_scales(12.0)
    assert max(loose) > max(tight)


def test_assign_bucket_full_partition():
    """Every combination lands in exactly one of the four documented buckets."""
    buckets = set()
    for is_rookie in (True, False):
        for hist in (0, 1, 3, 8):
            for changed in (0, 1):
                row = pd.Series({
                    "is_rookie": is_rookie,
                    "seasons_of_history": hist,
                    "team_changed": changed,
                })
                buckets.add(assign_bucket(row))
    assert buckets <= {"rookie", "second_year", "team_changed_vet", "established_vet"}
    assert "rookie" in buckets
    assert "established_vet" in buckets
