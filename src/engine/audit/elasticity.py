"""Elasticity table (§13.4) — which knobs actually move E[$].

Perturb one modeling knob by +/-`perturb`, re-simulate under **common random
numbers**, and read the change in expected dollars. CRN is what makes this
affordable: the paired difference removes the shared Monte-Carlo noise, so a
$0.30 effect is visible at a few thousand replications instead of a few
hundred thousand.

The verdict bands come straight from the indifference zone, which is the only
scale on which "does this matter" has an answer:

    |dE[$]| >= indifference_zone   -> nail_it       (worth real modeling effort)
    0.25 .. indifference_zone      -> hardcode_it   (pick a defensible constant)
    <  0.25                        -> irrelevant    (stop discussing it)

**Every knob here is global — it perturbs all twelve rosters at once.** E[$] is
zero-sum against a fixed pot, so a global perturbation moves MY dollars only
through composition: which players my roster happens to hold, and how the
payout's convexity treats the resulting spread. That is the right question for
a modeling-effort budget ("would a better sigma model change my draft?") and
the wrong one for a marginal-player question ("how much is this receiver
worth?"). Do not read these rows as player valuations.

**The lambda knobs from the design's original list are gone.** §13.0 and AD-2
replaced the one-factor loading model with a direct empirical slot matrix,
after measurement showed no set of loadings can hold QB1xWR1 = 0.383 and
WR1xWR2 = 0.040 at once. There is no lambda_QB to perturb. Its role is taken
by `correlation_scale`, which shrinks the fitted matrix toward the identity —
the same question ("how much does same-team coupling matter?") asked of the
model that actually shipped.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

#: Below this, in dollars, a knob is not worth discussing. Deliberately not
#: configurable: it is a statement about the resolution of the whole study.
IRRELEVANT_BELOW: float = 0.25


@dataclass(frozen=True)
class Knob:
    """One perturbable quantity, and how to apply it."""

    name: str
    #: (bundle, cholesky, factor) -> (bundle, cholesky). Multiplicative unless
    #: `additive`, in which case `factor` arrives as a signed absolute step.
    apply: Callable
    description: str = ""
    additive: bool = False


def _scale_field(field: str):
    def apply(bundle, chol, factor):
        return replace(bundle, **{field: getattr(bundle, field) * factor}), chol
    return apply


def _scale_level(bundle, chol, factor):
    """Scale the whole projected distribution: P10, P50 and P90 together.

    Scaling P50 alone pushes the median past P90, and `split_normal_ppf` clamps
    the resulting negative half-width to zero — so the upper tail vanishes and
    the knob reads as a large NEGATIVE elasticity that is really a violated
    monotonicity invariant. Measured: +10% on P50 alone moved E[$] from $24.22
    to $21.38, the wrong direction and for the wrong reason.
    """
    return replace(bundle, rate_p10=bundle.rate_p10 * factor,
                   rate_p50=bundle.rate_p50 * factor,
                   rate_p90=bundle.rate_p90 * factor), chol


def _scale_spread(bundle, chol, factor):
    """Widen or narrow the P10-P90 interval about P50, leaving P50 fixed."""
    return replace(
        bundle,
        rate_p10=bundle.rate_p50 - (bundle.rate_p50 - bundle.rate_p10) * factor,
        rate_p90=bundle.rate_p50 + (bundle.rate_p90 - bundle.rate_p50) * factor,
    ), chol


def _scale_hazard(bundle, chol, factor):
    # A hazard is a probability. Scaling past 1.0 would silently make players
    # available every week including byes, which reads as a large elasticity
    # that is really a clipped input.
    return replace(bundle,
                   games_hazard=np.clip(bundle.games_hazard * factor, 0.0, 1.0)), chol


def _scale_correlation(bundle, chol, factor):
    """Shrink the fitted slot matrix toward the identity by `factor`.

    Off-diagonals scale; the diagonal stays at 1 so each player's marginal
    weekly variance is untouched and the elasticity measures COUPLING rather
    than a confounded change in total variance.
    """
    corr = chol @ chol.T
    off = corr - np.diag(np.diag(corr))
    scaled = np.eye(corr.shape[0]) + off * factor
    # shrink toward the identity until PSD — a scaled correlation matrix is not
    # guaranteed to stay PSD for factor > 1
    for _ in range(40):
        eigenvalues = np.linalg.eigvalsh(scaled)
        if eigenvalues.min() > 1e-8:
            break
        scaled = 0.9 * scaled + 0.1 * np.eye(corr.shape[0])
    return bundle, np.linalg.cholesky(scaled)


KNOBS: tuple[Knob, ...] = (
    Knob("rate_level_scale", _scale_level,
         "the projection itself, P10/P50/P90 together — the reference "
         "elasticity everything else is judged against"),
    Knob("weekly_sigma_scale", _scale_field("weekly_sigma"),
         "week-to-week variance; drives the weekly-high prize"),
    Knob("rate_spread_scale", _scale_spread,
         "width of the calibrated P10-P90 interval, P50 held fixed"),
    Knob("hazard_scale", _scale_hazard,
         "availability (§12.4); clipped at 1.0"),
    Knob("correlation_scale", _scale_correlation,
         "same-NFL-team coupling, shrunk toward the identity (replaces the "
         "retired lambda knobs, AD-2)"),
)


def elasticity_table(bundle, cholesky: np.ndarray, simulate: Callable, *,
                     indifference_zone: float, perturb: float = 0.10,
                     knobs: tuple[Knob, ...] = KNOBS) -> pd.DataFrame:
    """dE[$]/dknob at +/-`perturb`, paired under CRN.

    `simulate(bundle, cholesky) -> float` must return MY expected dollars and
    must use the same seed root every call — that is the entire CRN contract,
    and violating it turns every row below into Monte-Carlo noise.
    """
    base = simulate(bundle, cholesky)
    rows = []
    for knob in knobs:
        up_b, up_c = knob.apply(bundle, cholesky, 1.0 + perturb)
        dn_b, dn_c = knob.apply(bundle, cholesky, 1.0 - perturb)
        up = simulate(up_b, up_c)
        down = simulate(dn_b, dn_c)

        # Central difference for the derivative, max absolute one-sided move
        # for the verdict. They differ when the response is one-sided — a
        # hazard clipped at 1.0 is exactly that — and using only the central
        # difference would report such a knob as irrelevant.
        central = (up - down) / 2.0
        largest = max(abs(up - base), abs(down - base))
        rows.append({
            "knob": knob.name,
            "description": knob.description,
            "base_dollars": round(base, 4),
            "plus_dollars": round(up, 4),
            "minus_dollars": round(down, 4),
            "d_dollars": round(central, 4),
            "max_abs_move": round(largest, 4),
            "elasticity": round(central / base * (1 / perturb), 4) if base else None,
            "verdict": _verdict(largest, indifference_zone),
        })
    return pd.DataFrame(rows).sort_values("max_abs_move", ascending=False,
                                          ignore_index=True)


def _verdict(magnitude: float, indifference_zone: float) -> str:
    if magnitude >= indifference_zone:
        return "nail_it"
    if magnitude >= IRRELEVANT_BELOW:
        return "hardcode_it"
    return "irrelevant"
