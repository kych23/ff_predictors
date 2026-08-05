"""Ported projection layer: folds, quantiles, conformal calibration (§8.1, §12)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.config import load_league
from src.models.projection.calibrate import (
    IntervalCalibrator, assign_bucket, calibrate_oof_lofo, coverage_by_bucket,
)
from src.models.projection.folds import make_expanding_folds
from src.models.projection.quantile_model import QuantileGBM

ARTIFACTS = Path("data/artifacts")


@pytest.fixture(scope="module")
def league():
    return load_league()


# ------------------------------------------------------------------- folds
def test_expanding_folds_never_train_on_the_future(league):
    folds = make_expanding_folds(range(2012, 2026), league.training.min_train_seasons)
    for fold in folds:
        assert max(fold.train_seasons) < fold.test_season, (
            "a fold trained on a season at or after its test season would leak"
        )


def test_warmup_drops_degenerate_early_folds():
    folds = make_expanding_folds(range(2012, 2026), 5)
    assert folds[0].test_season == 2017, "first evaluable season is start + warmup"
    assert len(folds[0].train_seasons) == 5


def test_no_folds_when_history_is_too_short():
    assert make_expanding_folds([2020, 2021, 2022], 5) == []


# ---------------------------------------------------------------- quantiles
def _toy(rng, n=800):
    x = rng.uniform(0, 20, n)
    noise = rng.normal(0, 2 + 0.15 * x, n)
    return pd.DataFrame({"x1": x, "x2": rng.normal(0, 1, n)}), pd.Series(x + noise)


def test_quantile_model_is_monotone_after_rearrangement():
    """P10 <= P50 <= P90 by rearrangement, never by clipping (§8.1)."""
    rng = np.random.default_rng(0)
    X, y = _toy(rng)
    pred = QuantileGBM().fit(X, y).predict(X)
    assert (pred["p10"] <= pred["p50"]).all()
    assert (pred["p50"] <= pred["p90"]).all()


def test_quantile_model_tracks_the_signal():
    rng = np.random.default_rng(1)
    X, y = _toy(rng)
    pred = QuantileGBM().fit(X, y).predict(X)
    assert np.corrcoef(pred["p50"], y)[0, 1] > 0.8


def test_intervals_widen_where_noise_grows():
    """Heteroskedastic by construction: spread must grow with x."""
    rng = np.random.default_rng(2)
    X, y = _toy(rng, n=2000)
    pred = QuantileGBM().fit(X, y).predict(X)
    width = pred["p90"] - pred["p10"]
    lo = width[X["x1"] < 5].mean()
    hi = width[X["x1"] > 15].mean()
    assert hi > lo


# -------------------------------------------------------------- calibration
#: Phi^-1(0.9). A half-width of exactly this times the residual SD gives
#: nominal 80% coverage, so `width_factor` below 1 is genuinely overconfident.
Z90 = 1.2815515655446004


def _oof(rng, n=1500, width_factor=0.65, resid_sd=1.0):
    """Intervals of a KNOWN calibration, so the fixture cannot lie.

    residual = y - p50 ~ N(0, resid_sd). Half-width = width_factor * Z90 *
    resid_sd, so width_factor < 1 under-covers and > 1 over-covers.
    """
    y = rng.normal(10, 4, n)
    p50 = y + rng.normal(0, resid_sd, n)
    half = width_factor * Z90 * resid_sd
    buckets = rng.choice(["rookie", "second_year", "team_changed_vet",
                          "established_vet"], n)
    return pd.DataFrame({"y": y, "p50": p50, "p10": p50 - half,
                         "p90": p50 + half, "bucket": buckets,
                         "season": rng.integers(2017, 2026, n)})


def test_calibration_widens_overconfident_intervals(league):
    rng = np.random.default_rng(3)
    oof = _oof(rng)
    raw = coverage_by_bucket(oof)
    raw_all = float(raw.loc[raw.bucket == "_all", "coverage"].iloc[0])
    assert raw_all < 0.80, "fixture must start overconfident"

    calibrated = calibrate_oof_lofo(oof, league.training.min_bucket_n)
    cal = coverage_by_bucket(calibrated)
    cal_all = float(cal.loc[cal.bucket == "_all", "coverage"].iloc[0])
    assert cal_all > raw_all
    assert abs(cal_all - 0.80) < 0.06


def test_calibrator_only_widens(league):
    """Floored at 1.0 so calibration can never make a model look sharper than
    it is (§8.1)."""
    rng = np.random.default_rng(4)
    oof = _oof(rng, width_factor=3.0)   # already far too WIDE
    calibrated = calibrate_oof_lofo(oof, league.training.min_bucket_n)
    assert (calibrated["p90"] - calibrated["p10"] >=
            oof["p90"] - oof["p10"] - 1e-9).all()


def test_calibration_preserves_monotonicity(league):
    rng = np.random.default_rng(5)
    calibrated = calibrate_oof_lofo(_oof(rng), league.training.min_bucket_n)
    assert (calibrated["p10"] <= calibrated["p50"]).all()
    assert (calibrated["p50"] <= calibrated["p90"]).all()


def test_bucket_assignment_matches_the_canonical_names(league):
    from src.core.constants import CALIBRATION_BUCKETS
    probes = [
        {"is_rookie": True},
        {"is_rookie": False, "seasons_of_history": 1},
        {"is_rookie": False, "seasons_of_history": 5, "team_changed": 1},
        {"is_rookie": False, "seasons_of_history": 5, "team_changed": 0},
    ]
    assert {assign_bucket(pd.Series(p)) for p in probes} == set(CALIBRATION_BUCKETS)


# ------------------------------------------------ the real fitted artifact
def _latest_projection():
    metas = sorted(ARTIFACTS.glob("projections_*.meta.json"))
    if not metas:
        return None, None
    meta = json.loads(metas[-1].read_text())
    frame = pd.read_parquet(ARTIFACTS / f"projections_{meta['snapshot_id']}.parquet")
    return frame, meta


def test_fitted_projections_are_monotone():
    frame, meta = _latest_projection()
    if frame is None:
        pytest.skip("no projection artifact; run scripts/train_projection_v2.py")
    assert (frame["p10"] <= frame["p50"]).all()
    assert (frame["p50"] <= frame["p90"]).all()
    assert meta["target_season"] == 2026


def test_fitted_projections_cover_every_modeled_position(league):
    frame, _ = _latest_projection()
    if frame is None:
        pytest.skip("no projection artifact")
    assert set(frame["position"]) == set(league.roster.modeled_positions)


def test_fitted_projections_are_positive_and_bounded():
    """A per-game points projection outside [0, 40] means something is wrong
    with the label or the scoring, not with the model."""
    frame, _ = _latest_projection()
    if frame is None:
        pytest.skip("no projection artifact")
    assert frame["p50"].min() >= 0
    assert frame["p50"].max() < 40


def test_bundle_uses_the_model_when_available():
    from src.platform import bundle as bundle_mod
    path = Path("data/bundles/draft_night_bundle.parquet")
    if not path.exists():
        pytest.skip("no bundle; run scripts/build_bundle.py")
    b = bundle_mod.read(path)
    assert b.value_source in ("quantile_model", "prior_season_ppg")
    full = b.players[b.players["coverage"] == "full"]
    assert (full["value"] > 0).all(), "a covered player must carry a value"
