"""ADP-survival probability (DrafterSpec.md §4.8.1).

The only market input in the live loop (recommender side of the wall, §4.0).
Probability that a player lasts to my next pick.

A symmetric normal mis-models ADP: real pick distributions are right-skewed and
bounded at pick 1, so a normal OVER-states elite (top-5) survival and UNDER-states
late fliers. v1 models the draft pick as a right-skewed, lower-bounded
**log-normal in pick-number** (parameterized by ADP mean + adp_stdev), flooring
elite-pick survival near 0. (v1.5: empirical survival curves.)

Survival to my next pick N = P(player's draft pick > N) = 1 - F_lognormal(N).


PORTED from src/recommender/survival.py (DraftEngineDesign.md §9.2).
Tier-2 fallback path (§7.3). Behavior unchanged from v1; the only edits
are the v2 config API (load_league, flex_eligibility) and the DEF -> DST
position rename (§10.7).
"""
from __future__ import annotations

import math

from scipy.stats import lognorm


def _lognormal_params(adp_mean: float, adp_stdev: float) -> tuple:
    """Solve log-normal (sigma, scale) so its MEAN == adp_mean, SD == adp_stdev."""
    adp_mean = max(float(adp_mean), 1.0)
    sd = float(adp_stdev) if adp_stdev and adp_stdev > 0 else max(adp_mean * 0.15, 0.5)
    sigma2 = math.log(1.0 + (sd / adp_mean) ** 2)
    sigma = math.sqrt(sigma2)
    mu = math.log(adp_mean) - sigma2 / 2.0
    scale = math.exp(mu)  # scipy lognorm: scale = exp(mu), shape = sigma
    return sigma, scale


def survival_probability(adp_mean: float, adp_stdev: float, next_pick: int) -> float:
    """P(player is still available at overall pick ``next_pick``).

    = P(draft pick > next_pick). If I pick at ``next_pick``, the player survives iff
    nobody took him in picks 1..next_pick-1, i.e. his pick number >= next_pick.
    """
    if next_pick is None:
        return 1.0
    # No public ADP -> the player isn't in the draft market, so opponents drafting
    # by ADP won't take him -> he survives to my next pick. (Without this guard a
    # single NaN-ADP player poisons _expected_best_surviving and silently zeroes the
    # whole VONA wait term.)
    try:
        if adp_mean is None or math.isnan(float(adp_mean)):
            return 1.0
    except (TypeError, ValueError):
        return 1.0
    sigma, scale = _lognormal_params(adp_mean, adp_stdev)
    # survival = P(pick >= next_pick) = sf evaluated just below next_pick
    return float(lognorm.sf(next_pick - 0.5, s=sigma, scale=scale))
