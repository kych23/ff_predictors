"""Round -> risk appetite schedule (DrafterSpec.md §4.8.1 / Fork 1).

Swaps the point estimate for a round-shifted quantile: floor-leaning early (~P25,
protect starters) shifting to ceiling-leaning late (~P85, upside swings). One
tunable knob — this is where the risk edge changes picks. Using only the middle
estimate would discard the uncertainty the engine works to produce (§3).

We have P10/P50/P90 per player; the schedule chooses a target quantile alpha by
round and we piecewise-linearly interpolate the player's value at that alpha.
"""
from __future__ import annotations

import numpy as np

EARLY_ALPHA = 0.25   # floor-leaning
LATE_ALPHA = 0.85    # ceiling-leaning


def round_to_alpha(draft_round: int, total_rounds: int,
                   early: float = EARLY_ALPHA, late: float = LATE_ALPHA) -> float:
    """Linearly shift the target quantile from ``early`` (round 1) to ``late`` (last)."""
    if total_rounds <= 1:
        return early
    frac = (draft_round - 1) / (total_rounds - 1)
    frac = min(max(frac, 0.0), 1.0)
    return early + frac * (late - early)


def value_vector_at_alpha(p10, p50, p90, alpha: float) -> np.ndarray:
    p10 = np.asarray(p10, dtype=float)
    p50 = np.asarray(p50, dtype=float)
    p90 = np.asarray(p90, dtype=float)
    a = min(max(alpha, 0.10), 0.90)
    if a <= 0.50:
        w = (a - 0.10) / 0.40
        return p10 + w * (p50 - p10)
    w = (a - 0.50) / 0.40
    return p50 + w * (p90 - p50)
