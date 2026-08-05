"""Opponent policy fit and the pre-registered gate (§14.1, §14.2)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import check_grad

from src.models.opponents.diagnostics import run_gate
from src.models.opponents.fit import (
    Choice, _objective_and_grad, _pack, fit_managers, league_mean_loglik,
)
from src.platform.sources.league_history import load_history

POSITIONS = ("QB", "RB", "WR", "TE")


def _synthetic(rng, *, n=400, n_managers=6, signal=0.0, seasons=(2021, 2022, 2023)):
    """Choices generated from a KNOWN policy.

    ``signal`` scales a per-manager positional preference. At 0 the managers
    are identical and the gate must NOT fire; large values make identity
    genuinely predictive and it must.
    """
    choices = []
    prefs = {m: rng.normal(0, signal, len(POSITIONS)) for m in range(n_managers)}
    for i in range(n):
        m = i % n_managers
        overall = 30 + (i % 100)
        adp = np.sort(rng.uniform(overall - 10, overall + 70, 40))
        pos = rng.integers(0, len(POSITIONS), 40)
        utility = -(adp - overall) / 8.0 + prefs[m][pos]
        p = np.exp(utility - utility.max())
        p /= p.sum()
        choices.append(Choice(
            manager=f"m{m}", season=seasons[i % len(seasons)], overall=overall,
            slot_z=rng.uniform(-1, 1), adp=adp, position_idx=pos,
            chosen=int(rng.choice(len(adp), p=p)),
        ))
    return choices


# --------------------------------------------------------------------- fit
def test_analytic_gradient_matches_numeric():
    """The gradient is load-bearing for runtime, so it is verified, not assumed."""
    rng = np.random.default_rng(0)
    choices = _synthetic(rng, n=120, n_managers=4)
    index = {f"m{i}": i for i in range(4)}
    packed = _pack(choices, index)
    args = (packed, 4, len(POSITIONS), 3.0, 0.5)
    theta = rng.normal(0, 0.3, 4 * 7)
    theta[::7] = np.log(8.0)
    err = check_grad(lambda t: _objective_and_grad(t, *args)[0],
                     lambda t: _objective_and_grad(t, *args)[1], theta)
    grad_norm = np.linalg.norm(_objective_and_grad(theta, *args)[1])
    assert err / grad_norm < 1e-5


def test_fit_recovers_a_strong_positional_preference():
    rng = np.random.default_rng(1)
    choices = _synthetic(rng, n=900, n_managers=3, signal=1.5)
    fit = fit_managers(choices, POSITIONS, prior_beta_sd=5.0)
    spreads = [max(fm.beta.values()) - min(fm.beta.values())
               for fm in fit.managers.values()]
    assert max(spreads) > 0.5, "a real preference must survive the fit"


def test_tau_stays_positive():
    """tau is optimized as log tau, so positivity is structural."""
    rng = np.random.default_rng(2)
    fit = fit_managers(_synthetic(rng, n=200), POSITIONS)
    assert all(fm.tau > 0 for fm in fit.managers.values())


def test_beta_is_sum_to_zero():
    """Imposed at fit time via contrasts; re-centering afterwards would leave a
    flat direction in the likelihood."""
    rng = np.random.default_rng(3)
    fit = fit_managers(_synthetic(rng, n=200), POSITIONS)
    for fm in fit.managers.values():
        assert sum(fm.beta.values()) == pytest.approx(0.0, abs=1e-9)
        assert sum(fm.slot_coef.values()) == pytest.approx(0.0, abs=1e-9)


def test_slot_variation_is_reported():
    """A manager whose slot never varied has beta/slot_coef collinear."""
    rng = np.random.default_rng(4)
    choices = _synthetic(rng, n=120, n_managers=2)
    fixed = [Choice(c.manager, c.season, c.overall, 0.5, c.adp,
                    c.position_idx, c.chosen) for c in choices]
    fit = fit_managers(fixed, POSITIONS)
    assert all(not fm.slot_varies for fm in fit.managers.values())


def test_per_manager_beats_pooled_on_its_own_data():
    """Sanity: more parameters must fit the training data better. That this is
    true and the gate still fails is the whole point of the gate."""
    rng = np.random.default_rng(5)
    choices = _synthetic(rng, n=600, n_managers=4, signal=1.0)
    per_manager = fit_managers(choices, POSITIONS).loglik_of(choices)
    pooled = league_mean_loglik(choices, POSITIONS)
    assert per_manager > pooled


# -------------------------------------------------------------------- gate
def test_gate_does_not_fire_on_identical_managers():
    """The null case: managers drawn from one policy. Identity carries no
    information, so a shuffle must predict held-out picks as well."""
    rng = np.random.default_rng(6)
    choices = _synthetic(rng, n=300, n_managers=5, signal=0.0)
    gate = run_gate(choices, POSITIONS, n_permutations=25, seed=1)
    assert not gate.passed
    assert gate.permutation_p > 0.10


def test_gate_fires_on_a_strong_genuine_signal():
    """The power case: if identity really is predictive, the gate must say so,
    or it would be a test that always says no."""
    rng = np.random.default_rng(7)
    choices = _synthetic(rng, n=1200, n_managers=4, signal=2.0)
    gate = run_gate(choices, POSITIONS, n_permutations=25, seed=2,
                    prior_beta_sd=5.0)
    assert gate.permutation_p <= 0.10, (
        f"gate missed a real signal (p={gate.permutation_p:.3f}); it would "
        f"reject the layer regardless of the data"
    )


def test_gate_reports_observations_per_parameter():
    rng = np.random.default_rng(8)
    choices = _synthetic(rng, n=210, n_managers=3)
    gate = run_gate(choices, POSITIONS, n_permutations=5)
    assert gate.obs_per_parameter == pytest.approx(210 / (3 * 7))
    assert "obs/parameter" in gate.describe()


# ----------------------------------------------------------------- history
def test_history_parses_and_validates_the_snake():
    """Every transcribed pick is re-checked against the snake sequence implied
    by its slot; a transcription error would corrupt choice sets silently."""
    load = load_history()
    assert len(load.picks) > 600
    assert not load.snake_errors, f"snake mismatches: {load.snake_errors[:3]}"


def test_history_covers_the_2026_cohort():
    load = load_history()
    managers = set(load.managers)
    expected = {"Christian", "Sheil", "Ashay", "Gavin", "Alex", "Akhil",
                "Kevin", "Raam", "Ryan", "Zach", "Dhruv"}
    assert expected <= managers
    assert "Kyle" not in managers, "my own seat is not captured as an opponent"
