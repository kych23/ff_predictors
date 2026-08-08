"""Elasticity table (§13.4).

The table's whole value is that a reader trusts the ordering. So the tests are
about the perturbations being *clean*: each knob must move what it claims and
nothing else, and none may break an invariant the simulator relies on. The
first version of `rate_p50_scale` failed exactly that way — it scaled the
median past P90, `split_normal_ppf` clamped the negative half-width to zero,
and the knob reported a large effect with the wrong sign.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.engine.audit.elasticity import (
    IRRELEVANT_BELOW,
    KNOBS,
    Knob,
    elasticity_table,
)
from src.engine.sim.draws import ProjectionBundle


def _bundle(n=12, weeks=17):
    rng = np.random.default_rng(0)
    p50 = rng.uniform(6.0, 18.0, n)
    return ProjectionBundle(
        player_ids=np.array([f"P{i}" for i in range(n)]),
        positions=np.array(["RB"] * n),
        nfl_teams=np.array([f"T{i % 4}" for i in range(n)]),
        slot_idx=np.zeros(n, dtype=int),
        bye_weeks=np.zeros(n, dtype=int),
        is_kdst=np.zeros(n, dtype=bool),
        rate_p10=p50 * 0.7, rate_p50=p50, rate_p90=p50 * 1.4,
        weekly_sigma=p50 * 0.4,
        games_hazard=np.full((n, weeks), 0.9),
        kdst_quantiles=np.full((n, 101), np.nan),
    )


def _chol(k=2):
    corr = np.array([[1.0, 0.4], [0.4, 1.0]])
    return np.linalg.cholesky(corr)


def _knob(name: str) -> Knob:
    return next(k for k in KNOBS if k.name == name)


# ------------------------------------------------- perturbations are clean
@pytest.mark.parametrize("factor", [0.9, 1.1])
def test_no_knob_breaks_quantile_monotonicity(factor):
    """P10 <= P50 <= P90 is a load-bearing invariant, not a nicety:
    split_normal_ppf clamps a negative half-width to zero and silently deletes
    a tail. A knob that violates it measures the clamp, not the knob."""
    bundle, chol = _bundle(), _chol()
    for knob in KNOBS:
        out, _ = knob.apply(bundle, chol, factor)
        assert np.all(out.rate_p10 <= out.rate_p50 + 1e-9), knob.name
        assert np.all(out.rate_p50 <= out.rate_p90 + 1e-9), knob.name


def test_level_scale_moves_all_three_quantiles_together():
    out, _ = _knob("rate_level_scale").apply(_bundle(), _chol(), 1.1)
    base = _bundle()
    for field in ("rate_p10", "rate_p50", "rate_p90"):
        assert np.allclose(getattr(out, field), getattr(base, field) * 1.1)


def test_spread_scale_holds_the_median_fixed():
    """Otherwise it is a level knob wearing a spread knob's name, and the two
    rows in the table stop being distinguishable."""
    base = _bundle()
    out, _ = _knob("rate_spread_scale").apply(base, _chol(), 1.25)
    assert np.allclose(out.rate_p50, base.rate_p50)
    assert np.all(out.rate_p90 - out.rate_p50
                  > base.rate_p90 - base.rate_p50 - 1e-9)


def test_hazard_stays_a_probability():
    out, _ = _knob("hazard_scale").apply(_bundle(), _chol(), 1.5)
    assert np.all(out.games_hazard <= 1.0)
    assert np.all(out.games_hazard >= 0.0)


def test_correlation_scale_leaves_marginal_variance_alone():
    """Only the off-diagonals move. Scaling the diagonal too would confound
    'coupling matters' with 'total variance matters', which is a different row
    of this same table."""
    _, chol = _knob("correlation_scale").apply(_bundle(), _chol(), 0.5)
    corr = chol @ chol.T
    assert np.allclose(np.diag(corr), 1.0)
    assert corr[0, 1] == pytest.approx(0.2, abs=1e-9)


def test_correlation_scale_stays_positive_semidefinite():
    """Scaling off-diagonals up is not PSD-preserving in general; the shrink
    loop exists for that. Without it np.linalg.cholesky raises mid-audit."""
    corr = np.full((5, 5), 0.6)
    np.fill_diagonal(corr, 1.0)
    base = np.linalg.cholesky(corr)
    _, chol = _knob("correlation_scale").apply(_bundle(), base, 1.6)
    assert np.linalg.eigvalsh(chol @ chol.T).min() > 0


# ------------------------------------------------------------- the table
def _linear_sim(sensitivity: float):
    """A stand-in objective that responds only to the projection level."""
    def sim(bundle, chol) -> float:
        return 20.0 + sensitivity * float(bundle.rate_p50.mean())
    return sim


def test_verdicts_partition_by_magnitude():
    # mean rate_p50 is ~12, so a 10% perturbation at sensitivity 3.0 moves
    # about $3.6 — comfortably past the $2.00 zone.
    table = elasticity_table(_bundle(), _chol(), _linear_sim(3.0),
                             indifference_zone=2.0)
    row = table.set_index("knob").loc["rate_level_scale"]
    assert row["verdict"] == "nail_it"
    # nothing else touches rate_p50 in this stand-in
    others = table[table["knob"] != "rate_level_scale"]
    assert set(others["verdict"]) == {"irrelevant"}


def test_zone_boundary_is_respected():
    table = elasticity_table(_bundle(), _chol(), _linear_sim(1.0),
                             indifference_zone=1000.0)
    assert "nail_it" not in set(table["verdict"])


def test_irrelevant_floor_is_applied():
    table = elasticity_table(_bundle(), _chol(), _linear_sim(0.0),
                             indifference_zone=2.0)
    assert set(table["verdict"]) == {"irrelevant"}
    assert table["max_abs_move"].max() < IRRELEVANT_BELOW


def test_table_is_sorted_by_effect():
    table = elasticity_table(_bundle(), _chol(), _linear_sim(1.0),
                             indifference_zone=2.0)
    assert list(table["max_abs_move"]) == sorted(table["max_abs_move"],
                                                 reverse=True)


def test_one_sided_response_is_not_hidden_by_the_central_difference():
    """A knob clipped on one side — the hazard at 1.0 — has a central
    difference that understates it. The verdict reads max_abs_move for exactly
    that reason."""
    def sim(bundle, chol) -> float:
        return 20.0 + 100.0 * max(0.0, 0.9 - float(bundle.games_hazard.mean()))

    table = elasticity_table(_bundle(), _chol(), sim, indifference_zone=2.0)
    row = table.set_index("knob").loc["hazard_scale"]
    assert row["verdict"] == "nail_it"
    assert abs(row["d_dollars"]) < row["max_abs_move"]


def test_simulate_receives_the_same_cholesky_object_for_non_correlation_knobs():
    """CRN across arms requires the untouched inputs to be identical, not
    merely equal."""
    seen = []

    def sim(bundle, chol) -> float:
        seen.append(chol)
        return 20.0

    chol = _chol()
    elasticity_table(_bundle(), chol, sim, indifference_zone=2.0,
                     knobs=(_knob("weekly_sigma_scale"),))
    assert all(c is chol for c in seen)
