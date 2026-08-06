"""§21.3 — correlation is an empirical A/B, never a closed form.

Run the kernel twice on the same roster and the same seed: once with the
fitted slot matrix, once with the identity. Everything else is held fixed, so
the difference is the correlation and nothing else.

Two conditions make the comparison clean, and both were learned the hard way:

* **Fixed `rate` draw.** `Var(rate)` is per-player season uncertainty and is
  identical in both arms, so leaving it in only dilutes the ratio toward 1 and
  makes the test weaker the more uncertain the projection is. Degenerate
  quantiles (p10 = p50 = p90) pin it exactly.
* **Active weeks only.** The availability Bernoulli multiplies both arms
  equally in expectation but adds variance to each, again diluting the ratio.
  `apply_hazard=False` removes it.

Under the slot-matrix model (AD-2) the expected ratio is not an approximation:

    Var(team week) = sum over NFL teams of  sum_{p,q in team} s_p s_q C[i_p, i_q]

so arm (b) checks an exact identity to Monte-Carlo tolerance rather than the
±15% band the one-factor draft needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.engine.sim.draws import ProjectionBundle, draw_points
from src.models.correlation.slot_matrix import DEFAULT_SLOTS

ROOT = "3dc0676bb0114bd3bb66e8d97c210fbb"
REPS = 6000
WEEKS = 8


def _fitted_matrix() -> np.ndarray:
    """The measured structure, at the magnitudes scripts/fit_models.py reports
    on 2015-2025: QB1xWR1 +0.382, WR1xWR2 +0.045, RB1xRB2 -0.125."""
    slots = list(DEFAULT_SLOTS)
    c = np.eye(len(slots))
    measured = {("QB1", "WR1"): 0.382, ("QB1", "WR2"): 0.325,
                ("QB1", "TE1"): 0.300, ("QB1", "WR3"): 0.240,
                ("WR1", "WR2"): 0.045, ("WR1", "WR3"): 0.040,
                ("WR2", "WR3"): 0.040, ("RB1", "RB2"): -0.125,
                ("QB1", "RB1"): 0.092}
    for (a, b), r in measured.items():
        i, j = slots.index(a), slots.index(b)
        c[i, j] = c[j, i] = r
    return c


def _roster_bundle(n_teams=3, per_team=4):
    """A roster whose players share NFL teams — without shared teams there is
    nothing for the correlation to act on and the ratio is 1 by construction."""
    slots = list(DEFAULT_SLOTS)
    order = ["QB1", "RB1", "WR1", "WR2"][:per_team]
    ids, positions, nfl, slot_idx, sigma = [], [], [], [], []
    for t in range(n_teams):
        for k, slot in enumerate(order):
            ids.append(f"T{t}_{slot}")
            positions.append(slot[:-1])
            nfl.append(f"NFL{t}")
            slot_idx.append(slots.index(slot))
            sigma.append(6.0 + k)
    n = len(ids)
    rate = np.full(n, 12.0)
    return ProjectionBundle(
        player_ids=np.array(ids), positions=np.array(positions),
        nfl_teams=np.array(nfl), slot_idx=np.array(slot_idx),
        bye_weeks=np.zeros(n, dtype=int), is_kdst=np.zeros(n, dtype=bool),
        # degenerate on purpose: this is what "condition on a fixed rate" means
        rate_p10=rate, rate_p50=rate, rate_p90=rate,
        weekly_sigma=np.array(sigma, dtype=float),
        games_hazard=np.ones((n, WEEKS)),
        kdst_quantiles=np.full((n, 101), np.nan),
    )


def _team_week_variance(bundle, chol) -> float:
    points = draw_points(bundle, chol, ROOT, reps=REPS, apply_hazard=False)
    totals = points.sum(axis=0)                    # (W, R) — one roster
    return float(np.var(totals, axis=1).mean())


def _predicted_variance(bundle, corr: np.ndarray) -> float:
    total = 0.0
    for team in np.unique(bundle.nfl_teams):
        idx = np.flatnonzero(bundle.nfl_teams == team)
        s = bundle.weekly_sigma[idx]
        c = corr[np.ix_(bundle.slot_idx[idx], bundle.slot_idx[idx])]
        total += float(s @ c @ s)
    return total


@pytest.fixture(scope="module")
def arms():
    bundle = _roster_bundle()
    fitted = _fitted_matrix()
    on = _team_week_variance(bundle, np.linalg.cholesky(fitted))
    off = _team_week_variance(bundle, np.eye(len(DEFAULT_SLOTS)))
    return bundle, fitted, on, off


# ------------------------------------------------------------ arm (a)
def test_correlation_strictly_increases_team_week_variance(arms):
    """THE gate. The measured structure is net positive — the QB-receiver
    couplings dominate the negative RB1xRB2 — so a roster stacking same-team
    players must be more volatile week to week, not less."""
    _, _, on, off = arms
    assert on > off, f"variance with correlation {on:.2f} <= without {off:.2f}"


def test_the_increase_is_large_enough_to_be_a_test(arms):
    """A 1% difference would pass arm (a) while being indistinguishable from
    Monte-Carlo noise at any feasible rep count."""
    _, _, on, off = arms
    assert on / off > 1.05


# ------------------------------------------------------------ arm (b)
def test_variance_matches_the_slot_matrix_identity(arms):
    """Under AD-2 this is exact, not a ±15% band: the one-factor model the
    band was written for does not exist any more."""
    bundle, fitted, on, _ = arms
    predicted = _predicted_variance(bundle, fitted)
    assert on == pytest.approx(predicted, rel=0.05)


def test_uncorrelated_arm_matches_the_sum_of_squares(arms):
    bundle, _, _, off = arms
    assert off == pytest.approx(float((bundle.weekly_sigma ** 2).sum()),
                                rel=0.05)


# ------------------------------------------- the sign is genuinely tested
def test_flipping_the_sign_of_the_couplings_is_caught():
    """§5.2's hazard: a test that passes with the correlation sign flipped is
    not testing the correlation. Negating the off-diagonals must break arm
    (a)."""
    bundle = _roster_bundle()
    fitted = _fitted_matrix()
    flipped = np.eye(fitted.shape[0]) - (fitted - np.eye(fitted.shape[0]))
    # keep it a legal correlation matrix
    assert np.linalg.eigvalsh(flipped).min() > 0

    on = _team_week_variance(bundle, np.linalg.cholesky(fitted))
    wrong = _team_week_variance(bundle, np.linalg.cholesky(flipped))
    off = _team_week_variance(bundle, np.eye(fitted.shape[0]))
    assert wrong < off < on


def test_rb_pairing_alone_lowers_variance():
    """RB1xRB2 is the one measured NEGATIVE coupling (-0.125): two backs in the
    same committee split a fixed workload. A roster made only of RB pairs must
    be LESS volatile with correlation on — the opposite direction from the
    stack, from the same matrix."""
    slots = list(DEFAULT_SLOTS)
    n_teams = 4
    ids, nfl, slot_idx = [], [], []
    for t in range(n_teams):
        for slot in ("RB1", "RB2"):
            ids.append(f"T{t}_{slot}")
            nfl.append(f"NFL{t}")
            slot_idx.append(slots.index(slot))
    n = len(ids)
    rate = np.full(n, 12.0)
    bundle = ProjectionBundle(
        player_ids=np.array(ids), positions=np.array(["RB"] * n),
        nfl_teams=np.array(nfl), slot_idx=np.array(slot_idx),
        bye_weeks=np.zeros(n, dtype=int), is_kdst=np.zeros(n, dtype=bool),
        rate_p10=rate, rate_p50=rate, rate_p90=rate,
        weekly_sigma=np.full(n, 6.0),
        games_hazard=np.ones((n, WEEKS)),
        kdst_quantiles=np.full((n, 101), np.nan),
    )
    fitted = _fitted_matrix()
    on = _team_week_variance(bundle, np.linalg.cholesky(fitted))
    off = _team_week_variance(bundle, np.eye(fitted.shape[0]))
    assert on < off


# ------------------------------------------------------------- arm (c)
def test_dst_is_currently_drawn_independently_of_the_opposing_offense():
    """Arm (c) of §21.3 asserts corr(my DST, the opposing offense's team-week
    points) < 0. **That coupling is not implemented.** `draws.draw_points`
    routes every K/DST row through the empirical inverse-CDF with its own
    uniform stream and no opponent term, so the realized correlation is zero,
    not negative.

    This test pins the gap rather than skipping it. If someone wires
    `dst_opponent_coupling` into the kernel, this fails and arm (c) becomes
    implementable — which is the point.
    """
    n = 2
    grid = np.linspace(-5.0, 20.0, 101)
    bundle = ProjectionBundle(
        player_ids=np.array(["DST", "QB"]),
        positions=np.array(["DST", "QB"]),
        nfl_teams=np.array(["NFL0", "NFL1"]),
        slot_idx=np.array([-1, 0]),
        bye_weeks=np.zeros(n, dtype=int),
        is_kdst=np.array([True, False]),
        rate_p10=np.array([0.0, 18.0]), rate_p50=np.array([0.0, 18.0]),
        rate_p90=np.array([0.0, 18.0]),
        weekly_sigma=np.array([np.nan, 7.0]),
        games_hazard=np.ones((n, WEEKS)),
        kdst_quantiles=np.vstack([grid, np.full(101, np.nan)]),
    )
    points = draw_points(bundle, np.eye(len(DEFAULT_SLOTS)), ROOT, reps=REPS,
                         apply_hazard=False)
    rho = np.corrcoef(points[0].ravel(), points[1].ravel())[0, 1]
    assert abs(rho) < 0.03, (
        "DST now covaries with an offense — arm (c) of §21.3 is implementable "
        f"and this placeholder should become the real assertion (rho={rho:.3f})"
    )
