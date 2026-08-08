"""Model-layer tests: weekly variance, slot correlation, K/DST (§12.3, §13)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.config import load_league
from src.core.errors import ArtifactSnapshotMismatch, DataError
from src.models import artifacts
from src.models.artifacts import newest
from src.models.correlation.slot_matrix import (
    SlotCorrelation,
    fit_slot_correlation,
    from_prior,
)
from src.models.weekly.kdst import KDSTModel, fit_empirical
from src.models.weekly.variance import WeeklySigmaModel, fit_weekly_sigma


@pytest.fixture(scope="module")
def league():
    return load_league()


# ------------------------------------------------------------ weekly sigma
def _synthetic_summary(rng, n=600, coef=0.6, intercept=0.4):
    """Player-seasons with a KNOWN sigma-mu relationship to recover."""
    positions = rng.choice(["QB", "RB", "WR", "TE"], size=n)
    effects = {"QB": 0.0, "RB": 0.1, "WR": 0.05, "TE": -0.2}
    mu = rng.uniform(2.0, 25.0, size=n)
    log_sigma = (intercept + coef * np.log(mu)
                 + np.array([effects[p] for p in positions])
                 + rng.normal(0, 0.05, size=n))
    return pd.DataFrame({"season": 2020, "player_id": np.arange(n),
                         "position": positions, "mu": mu,
                         "sigma": np.exp(log_sigma), "games": 16})


def test_sigma_fit_recovers_a_known_relationship(league):
    rng = np.random.default_rng(0)
    summary = _synthetic_summary(rng, coef=0.6, intercept=0.4)
    model = fit_weekly_sigma(summary, league)
    assert model.log_mu_coef == pytest.approx(0.6, abs=0.03)
    assert model.intercept == pytest.approx(0.4, abs=0.06)
    assert model.n_observations == len(summary)


def test_sigma_prediction_is_monotone_in_mu(league):
    rng = np.random.default_rng(1)
    model = fit_weekly_sigma(_synthetic_summary(rng), league)
    mus = np.array([3.0, 8.0, 15.0, 22.0])
    preds = model.predict(np.array(["WR"] * 4), mus)
    assert np.all(np.diff(preds) > 0), "a higher mean must imply a wider spread"


def test_duan_smearing_is_applied(league):
    """exp(fitted log sigma) is the conditional MEDIAN, not the mean."""
    rng = np.random.default_rng(2)
    model = fit_weekly_sigma(_synthetic_summary(rng), league)
    assert model.smearing > 1.0, "smearing corrects median -> mean, so it is >1"


def test_sigma_floor_binds_for_low_variance_players(league):
    rng = np.random.default_rng(3)
    model = fit_weekly_sigma(_synthetic_summary(rng), league)
    tiny = model.predict(np.array(["QB"]), np.array([0.6]))
    assert tiny[0] >= league.weekly.sigma_floor_ratio * 0.6


def test_sigma_refuses_a_thin_sample(league):
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="refusing to fit"):
        fit_weekly_sigma(_synthetic_summary(rng, n=20), league)


def test_sigma_model_roundtrips():
    model = WeeklySigmaModel(
        intercept=0.4, log_mu_coef=0.6, position_effects={"QB": 0.0, "WR": 0.1},
        smearing=1.03, floor_ratio=0.35, n_observations=500,
        snapshot_id="sn2_x", model_version="draft_v2.y",
    )
    again = WeeklySigmaModel.from_dict(model.to_dict())
    assert again == model


# ------------------------------------------------------- slot correlation
def _slot_panel(rng, n_games=800, qb_wr=0.4, wr_wr=0.05):
    """Team-games with a prescribed correlation structure."""
    slots = ["QB1", "RB1", "RB2", "WR1", "WR2", "WR3", "TE1"]
    target = np.eye(len(slots))
    for a, b, r in [("QB1", "WR1", qb_wr), ("QB1", "WR2", qb_wr * 0.9),
                    ("WR1", "WR2", wr_wr), ("RB1", "RB2", -0.13)]:
        i, j = slots.index(a), slots.index(b)
        target[i, j] = target[j, i] = r
    chol = np.linalg.cholesky(target)
    draws = rng.standard_normal((n_games, len(slots))) @ chol.T
    rows = []
    for g in range(n_games):
        for k, slot in enumerate(slots):
            rows.append({"season": 2020 + g % 5, "week": g % 17 + 1,
                         "team": f"T{g % 32}", "slot": slot,
                         "points": 10 + 5 * draws[g, k]})
    return pd.DataFrame(rows)


def test_correlation_recovers_a_known_structure():
    rng = np.random.default_rng(10)
    corr = fit_slot_correlation(_slot_panel(rng, n_games=3000))
    assert corr.correlation("QB1", "WR1") == pytest.approx(0.40, abs=0.05)
    assert corr.correlation("WR1", "WR2") == pytest.approx(0.05, abs=0.05)
    assert corr.correlation("RB1", "RB2") == pytest.approx(-0.13, abs=0.05)


def test_matrix_is_psd_and_cholesky_reproduces_it():
    rng = np.random.default_rng(11)
    corr = fit_slot_correlation(_slot_panel(rng, n_games=2000))
    assert corr.min_eigenvalue > 0
    rebuilt = corr.cholesky @ corr.cholesky.T
    assert np.allclose(rebuilt, corr.matrix, atol=1e-8)


def test_draws_reproduce_the_target_correlation():
    rng = np.random.default_rng(12)
    corr = fit_slot_correlation(_slot_panel(rng, n_games=2000))
    draws = corr.draw(np.random.default_rng(99), 60_000)
    empirical = np.corrcoef(draws, rowvar=False)
    assert np.allclose(empirical, corr.matrix, atol=0.02)
    assert np.allclose(draws.std(axis=0), 1.0, atol=0.02), "unit variance"


def test_one_factor_model_is_infeasible_on_real_structure():
    """§13.0, encoded as a test.

    A one-factor model forces corr(WR1,WR2) = lambda_WR^2. With the MEASURED
    values, satisfying that fixes lambda_WR, and then corr(QB1,WR1) demands a
    lambda_QB above 1 — impossible. This is why the design carries an empirical
    matrix rather than a latent factor.
    """
    qb_wr1, wr1_wr2 = 0.383, 0.040          # measured 2012-2025
    lambda_wr = np.sqrt(wr1_wr2)
    lambda_qb = qb_wr1 / lambda_wr
    assert lambda_wr == pytest.approx(0.20, abs=0.01)
    assert lambda_qb > 1.0, (
        "if this ever drops below 1 the factor model became feasible and "
        "§13.0's central claim needs revisiting"
    )


def test_thin_pairs_are_shrunk_not_guessed():
    """A correlation from 30 observations is noise; shrink it to zero."""
    rng = np.random.default_rng(13)
    panel = _slot_panel(rng, n_games=2000)
    panel = panel[~((panel.slot == "WR3") & (panel.index % 40 != 0))]
    corr = fit_slot_correlation(panel)
    n = corr.pair_counts[corr.index_of("QB1"), corr.index_of("WR3")]
    if n < 200:
        assert corr.correlation("QB1", "WR3") == 0.0


def test_prior_matrix_loads_and_factors():
    """The fallback is a checked-in copy of measured data (§13.3)."""
    from pathlib import Path

    import yaml
    prior = yaml.safe_load(Path("config/correlation_prior.yaml").read_text())
    corr = from_prior(prior)
    assert corr.source == "assumed"
    assert corr.min_eigenvalue > 0.3
    assert corr.correlation("QB1", "WR1") == pytest.approx(0.370, abs=1e-6)
    assert corr.correlation("WR1", "WR2") == pytest.approx(0.037, abs=1e-6)


def test_prior_matches_the_fitted_matrix_on_real_data():
    """The checked-in prior must not drift from what the data says."""
    from pathlib import Path

    import yaml
    prior = yaml.safe_load(Path("config/correlation_prior.yaml").read_text())
    fitted_path = Path("data/artifacts")
    npz = newest("correlation_*.npz", root=fitted_path)
    if npz is None:
        pytest.skip("no fitted correlation artifact; run scripts/fit_models.py")
    corr = from_prior(prior)
    import json
    with np.load(npz, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        fitted = SlotCorrelation(
            slots=tuple(meta["slots"]), matrix=data["matrix"],
            cholesky=data["cholesky"], pair_counts=data["pair_counts"],
            min_eigenvalue=float(meta["min_eigenvalue"]),
            was_projected=False, source="fitted",
        )
    for a, b in [("QB1", "WR1"), ("WR1", "WR2"), ("RB1", "RB2"), ("QB1", "TE1")]:
        assert abs(corr.correlation(a, b) - fitted.correlation(a, b)) < 0.05, (
            f"{a}x{b}: prior {corr.correlation(a,b):+.3f} has drifted from "
            f"fitted {fitted.correlation(a,b):+.3f}"
        )


# ------------------------------------------------------------------- K/DST
def test_empirical_ppf_reproduces_the_marginal():
    rng = np.random.default_rng(20)
    sample = pd.Series(rng.gamma(3.0, 2.5, size=5000))
    dist = fit_empirical(sample, "DST")
    drawn = dist.ppf(rng.uniform(size=50_000))
    assert drawn.mean() == pytest.approx(sample.mean(), rel=0.05)
    assert drawn.std() == pytest.approx(sample.std(), rel=0.10)


def test_ppf_is_deterministic_in_u():
    """Inverse-CDF sampling is what keeps CRN intact (§15.1): the same uniform
    must map to the same value regardless of when it is drawn."""
    rng = np.random.default_rng(21)
    dist = fit_empirical(pd.Series(rng.normal(7, 5, 3000)), "K")
    u = np.array([0.1, 0.5, 0.9])
    assert np.array_equal(dist.ppf(u), dist.ppf(u))


def test_ppf_is_monotone():
    rng = np.random.default_rng(22)
    dist = fit_empirical(pd.Series(rng.normal(7, 5, 3000)), "K")
    out = dist.ppf(np.linspace(0, 1, 50))
    assert np.all(np.diff(out) >= 0)


def test_empirical_refuses_a_thin_sample():
    with pytest.raises(ValueError, match="mostly noise"):
        fit_empirical(pd.Series(np.arange(50.0)), "DST")


def test_kdst_unknown_position_raises():
    rng = np.random.default_rng(23)
    model = KDSTModel(distributions={
        "K": fit_empirical(pd.Series(rng.normal(8, 4, 2000)), "K")})
    with pytest.raises(KeyError, match="no empirical distribution"):
        model.ppf("DST", np.array([0.5]))


# --------------------------------------------------------------- artifacts
def test_artifact_roundtrip(tmp_path, league):
    rng = np.random.default_rng(30)
    sigma = fit_weekly_sigma(_synthetic_summary(rng), league,
                             snapshot_id="sn2_test", model_version="v")
    corr = fit_slot_correlation(_slot_panel(rng, n_games=1200),
                                snapshot_id="sn2_test", model_version="v")
    kdst = KDSTModel(
        distributions={"K": fit_empirical(pd.Series(rng.normal(8, 4, 2000)), "K"),
                       "DST": fit_empirical(pd.Series(rng.normal(6, 5, 2000)), "DST")},
        snapshot_id="sn2_test", model_version="v")
    artifacts.save_weekly_sigma(sigma, root=tmp_path)
    artifacts.save_correlation(corr, root=tmp_path)
    artifacts.save_kdst(kdst, root=tmp_path)

    loaded = artifacts.load_all("sn2_test", root=tmp_path)
    assert loaded.weekly_sigma.log_mu_coef == pytest.approx(sigma.log_mu_coef)
    assert np.allclose(loaded.correlation.matrix, corr.matrix)
    assert loaded.kdst.distributions["K"].mean == pytest.approx(
        kdst.distributions["K"].mean)


def test_provenance_mismatch_is_caught(tmp_path, league):
    """Stamping snapshot ids without comparing them lets a model fitted on one
    snapshot pair silently with another (§20.3)."""
    rng = np.random.default_rng(31)
    sigma = fit_weekly_sigma(_synthetic_summary(rng), league,
                             snapshot_id="sn2_OTHER")
    corr = fit_slot_correlation(_slot_panel(rng, n_games=1200),
                                snapshot_id="sn2_test")
    kdst = KDSTModel(distributions={
        "K": fit_empirical(pd.Series(rng.normal(8, 4, 2000)), "K")},
        snapshot_id="sn2_test")
    with pytest.raises(ArtifactSnapshotMismatch, match="weekly_sigma"):
        artifacts.ArtifactSet(sigma, corr, kdst, "sn2_test").assert_consistent()


def test_missing_artifact_is_reported(tmp_path):
    with pytest.raises(DataError, match="no weekly-sigma artifact"):
        artifacts.load_weekly_sigma("sn2_nope", root=tmp_path)
