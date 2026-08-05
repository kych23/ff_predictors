"""Weekly point draws (§12.3, §15.2).

The decomposition, stated once:

    rate_{p,r}     ~ SplitNormal(p10, p50, p90)        # ONCE per (p, r)
    points_{p,w,r} = [rate_{p,r} + sigma_w(p) * e_{slot(p),team(p),w,r}] * active
    e              = L @ z                             # L = Cholesky of §13
    active         = Bernoulli(hazard[p,w]) * (bye[p] != w)

``rate`` is drawn once per replication and held across weeks because it is
uncertainty about a *rate*, not a weekly outcome. Drawing it per week — as an
earlier revision did — makes the implied season-total interval about sqrt(17)
too narrow and supplies no week-to-week variance at all.

``e`` is a unit-variance correlated shock from the empirical slot matrix, so
conditional weekly variance is exactly ``sigma_w^2`` with no floor needed, and
every pairwise same-team correlation is reproduced — including the ones no
one-factor model can hold simultaneously (QB1-WR1 at 0.370 alongside WR1-WR2
at 0.037).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.constants import Z90, RngKind
from src.engine.sim import rng as rng_mod


@dataclass(frozen=True)
class ProjectionBundle:
    """The only object the kernel accepts from the modeling layers (§12.6)."""

    player_ids: np.ndarray       # (P,)
    positions: np.ndarray        # (P,)
    nfl_teams: np.ndarray        # (P,)
    slot_idx: np.ndarray         # (P,) index into the correlation matrix, -1 if none
    bye_weeks: np.ndarray        # (P,) int, 0 = none
    is_kdst: np.ndarray          # (P,) bool
    rate_p10: np.ndarray         # (P,) float32
    rate_p50: np.ndarray
    rate_p90: np.ndarray
    weekly_sigma: np.ndarray     # (P,)
    games_hazard: np.ndarray     # (P, W)
    kdst_quantiles: np.ndarray   # (P, Q) — NaN rows where not is_kdst
    snapshot_id: str = ""
    model_version: str = ""

    @property
    def n_players(self) -> int:
        return len(self.player_ids)

    @property
    def n_weeks(self) -> int:
        return self.games_hazard.shape[1]


def split_normal_ppf(u: np.ndarray, p10: np.ndarray, p50: np.ndarray,
                     p90: np.ndarray) -> np.ndarray:
    """Equal-weight two-piece normal matched exactly to three quantiles.

        sigma_lo = (p50 - p10) / Phi^-1(0.9)
        sigma_hi = (p90 - p50) / Phi^-1(0.9)
        x = p50 + sigma_lo * Phi^-1(u)   for u <  0.5
        x = p50 + sigma_hi * Phi^-1(u)   for u >= 0.5

    Attainable for ANY monotone triple, with no numerical solve — unlike a
    skew-normal, whose attainable skewness is bounded (|gamma1| < 0.995) and
    whose median has no closed form. Degenerate p10 == p50 gives sigma_lo = 0,
    a safe one-sided draw.

    MOMENTS (half the mass either side by construction):
        mean = p50 + (sigma_hi - sigma_lo) / sqrt(2*pi)
        var  = (sigma_lo^2 + sigma_hi^2)/2 - (sigma_hi - sigma_lo)^2 / (2*pi)

    This is NOT the classical split normal, whose mass below the mode is
    sigma_lo/(sigma_lo+sigma_hi) and whose mean carries a sqrt(2/pi) factor.
    That density does not match p50 at u = 0.5, which is why this one is used.

    No recentering to p50: the calibrated quantiles are the contract, and
    recentering would break the P10/P90 coverage §21.2 asserts.
    """
    from scipy.special import ndtri

    sigma_lo = np.maximum(p50 - p10, 0.0) / Z90
    sigma_hi = np.maximum(p90 - p50, 0.0) / Z90
    z = ndtri(np.clip(u, 1e-12, 1 - 1e-12))
    return p50 + np.where(u < 0.5, sigma_lo * z, sigma_hi * z)


def split_normal_moments(p10, p50, p90):
    """(mean, variance) of the density above — used by the variance tests."""
    sigma_lo = np.maximum(p50 - p10, 0.0) / Z90
    sigma_hi = np.maximum(p90 - p50, 0.0) / Z90
    mean = p50 + (sigma_hi - sigma_lo) / np.sqrt(2 * np.pi)
    var = ((sigma_lo ** 2 + sigma_hi ** 2) / 2
           - (sigma_hi - sigma_lo) ** 2 / (2 * np.pi))
    return mean, var


def draw_rates(bundle: ProjectionBundle, root: str, reps: int) -> np.ndarray:
    """(P, R) season-rate draws — once per (player, replication)."""
    p = bundle.n_players
    out = np.empty((p, reps), dtype="float64")
    for i in range(p):
        u = rng_mod.uniforms(root, RngKind.RATE, n=reps, a=i)
        out[i] = split_normal_ppf(u, bundle.rate_p10[i], bundle.rate_p50[i],
                                  bundle.rate_p90[i])
    return out


def draw_team_shocks(team_index: np.ndarray, cholesky: np.ndarray, root: str,
                     *, weeks: int, reps: int) -> np.ndarray:
    """(T_nfl, W, R, S) correlated unit-variance shocks per NFL team-week."""
    n_teams = int(team_index.max()) + 1 if len(team_index) else 0
    n_slots = cholesky.shape[0]
    out = np.empty((n_teams, weeks, reps, n_slots), dtype="float64")
    for t in range(n_teams):
        for w in range(weeks):
            z = rng_mod.normals(root, RngKind.WEEK, shape=(reps, n_slots),
                                a=t, b=w)
            out[t, w] = z @ cholesky.T
    return out


def draw_points(bundle: ProjectionBundle, cholesky: np.ndarray, root: str, *,
                reps: int, apply_hazard: bool = True) -> np.ndarray:
    """(P, W, R) float32 weekly points.

    C-order with replications fastest-varying so the per-week reductions in
    ``kernel.evaluate_rosters`` are contiguous.
    """
    p, w = bundle.n_players, bundle.n_weeks
    teams = np.unique(bundle.nfl_teams)
    team_of = {t: i for i, t in enumerate(teams)}
    team_index = np.array([team_of[t] for t in bundle.nfl_teams])

    rates = draw_rates(bundle, root, reps)                       # (P, R)
    shocks = draw_team_shocks(team_index, cholesky, root, weeks=w, reps=reps)

    out = np.empty((p, w, reps), dtype="float64")
    for i in range(p):
        slot = int(bundle.slot_idx[i])
        if bundle.is_kdst[i]:
            # empirical inverse-CDF; K/DST never use the split normal
            grid = np.linspace(0.0, 1.0, bundle.kdst_quantiles.shape[1])
            for week in range(w):
                u = rng_mod.uniforms(root, RngKind.KDST, n=reps, a=i, b=week)
                out[i, week] = np.interp(u, grid, bundle.kdst_quantiles[i])
            continue
        t = team_index[i]
        if slot >= 0:
            shock = shocks[t, :, :, slot]                        # (W, R)
        else:
            # no slot in the matrix (deep bench): independent, unit variance
            shock = rng_mod.normals(root, RngKind.EPS, shape=(w, reps), a=i)
        out[i] = rates[i][None, :] + bundle.weekly_sigma[i] * shock

    if apply_hazard:
        for i in range(p):
            for week in range(w):
                u = rng_mod.uniforms(root, RngKind.HAZARD, n=reps, a=i, b=week)
                active = u < bundle.games_hazard[i, week]
                if bundle.bye_weeks[i] == week + 1:
                    active[:] = False
                out[i, week] *= active
    return out.astype("float32")
