"""Sobol decomposition (§15.5).

Every test uses an analytic function whose true indices are known, because the
failure mode of a Sobol implementation is producing numbers that look
reasonable and mean something else — most often by building AB_i backwards,
which silently swaps the first-order and total estimators.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.engine.audit.sobol import saltelli_samples, sobol_indices


def test_ab_takes_column_i_from_b():
    """AB_i is A with ONE column swapped in from B. Built the other way round,
    the estimators exchange meanings — plausible numbers, opposite conclusion.
    This implementation had it backwards until this test ran: an ignored knob
    scored a total index of 1.00."""
    a, b, ab = saltelli_samples(16, [(0, 1)] * 3, seed=0)
    for i in range(3):
        assert np.allclose(ab[i][:, i], b[:, i])
        for j in range(3):
            if j != i:
                assert np.allclose(ab[i][:, j], a[:, j])


def test_an_additive_model_has_no_interaction():
    """f = 3x0 + x1: first-order indices must sum to ~1 and S_T ≈ S."""
    def evaluate(x):
        return 3.0 * x[0] + 1.0 * x[1]

    result = sobol_indices(["big", "small"], [(0, 1), (0, 1)], evaluate,
                           n=32768, seed=1)
    table = result.table.set_index("knob")
    assert table.loc["big", "first_order"] == pytest.approx(0.9, abs=0.05)
    assert table.loc["small", "first_order"] == pytest.approx(0.1, abs=0.05)
    assert result.table["first_order"].sum() == pytest.approx(1.0, abs=0.05)
    assert abs(table.loc["big", "interaction"]) < 0.05


def test_a_pure_interaction_is_invisible_to_first_order():
    """f = x0*x1 on centred inputs has S_i ≈ 0 and S_Ti ≈ 1 for both. This is
    exactly the case a one-at-a-time elasticity sweep cannot see, which is why
    §13.4 and §15.5 are two tables and not one."""
    def evaluate(x):
        return x[0] * x[1]

    result = sobol_indices(["a", "b"], [(-1, 1), (-1, 1)], evaluate,
                           n=4096, seed=2)
    table = result.table.set_index("knob")
    for knob in ("a", "b"):
        assert abs(table.loc[knob, "first_order"]) < 0.15
        assert table.loc[knob, "total"] > 0.8
        assert table.loc[knob, "interaction"] > 0.7


def test_an_ignored_knob_scores_zero():
    def evaluate(x):
        return x[0]

    result = sobol_indices(["used", "ignored"], [(0, 1), (0, 1)], evaluate,
                           n=2048, seed=3)
    table = result.table.set_index("knob")
    assert abs(table.loc["ignored", "total"]) < 0.05
    assert table.loc["used", "first_order"] == pytest.approx(1.0, abs=0.05)


def test_total_index_is_at_least_first_order_for_real_effects():
    def evaluate(x):
        return 2 * x[0] + x[1] + 3 * x[0] * x[1]

    result = sobol_indices(["a", "b"], [(0, 1), (0, 1)], evaluate,
                           n=4096, seed=4)
    for _, r in result.table.iterrows():
        assert r["total"] >= r["first_order"] - 0.05


def test_evaluation_count_matches_the_saltelli_design():
    calls = {"n": 0}

    def evaluate(x):
        calls["n"] += 1
        return float(x.sum())

    result = sobol_indices(["a", "b", "c"], [(0, 1)] * 3, evaluate, n=32)
    assert calls["n"] == 32 * (3 + 2)
    assert result.n_evaluations == calls["n"]


def test_a_constant_model_is_rejected_not_reported_as_zero():
    """Zero variance means either degenerate bounds or an `evaluate` ignoring
    its input. Dividing by it would emit nan indices that read as data."""
    with pytest.raises(ValueError, match="did not vary"):
        sobol_indices(["a"], [(0, 1)], lambda x: 42.0, n=32)


def test_mismatched_names_and_bounds_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        sobol_indices(["a", "b"], [(0, 1)], lambda x: float(x[0]), n=32)


def test_tiny_samples_are_rejected():
    with pytest.raises(ValueError, match="too small"):
        sobol_indices(["a"], [(0, 1)], lambda x: float(x[0]), n=4)


def test_convergence_flag_catches_an_undersampled_run():
    """The failure is silent: the table prints identically whether n was 48 or
    4096, and only the values give it away. Measured on the real board at
    n=48, hazard_scale came back with S=+1.897 and interaction=-0.998."""
    rng = np.random.default_rng(0)

    def noisy(x):
        return float(x[0] + 5.0 * rng.standard_normal())

    result = sobol_indices(["a", "b"], [(0, 1), (0, 1)], noisy, n=16, seed=0)
    assert not result.converged
    assert "NOT CONVERGED" in result.describe()


def test_a_well_sampled_run_reports_converged():
    result = sobol_indices(["a", "b"], [(0, 1), (0, 1)],
                           lambda x: 3 * x[0] + x[1], n=8192, seed=1)
    assert result.converged
    assert "NOT CONVERGED" not in result.describe()
