"""Greedy starting-lineup selection (§15.3, AD-5).

Sort by value descending, take each player the oracle permits. Optimal on a
transversal matroid, which the slot family always is — so this replaces the v1
ILP in the hot path entirely. The ILP survives only as a test oracle
(``tests/oracles/ilp_lineup.py``).

**Non-clairvoyance** is the load-bearing rule around this function, not inside
it: the caller passes *pre-week projections*, scores the chosen lineup with
*drawn* values, and never selects on the draws. Selecting on draws inflates
every roster, inflates high-variance rosters most, and reverses the
ceiling-vs-floor conclusions the simulator exists to produce.
"""
from __future__ import annotations

import numpy as np

from src.core.config.slots import SlotPlan
from src.domain.roster.matroid import build_oracle


def greedy_lineup(
    values: np.ndarray,
    positions: np.ndarray,
    plan: SlotPlan,
    *,
    require_positive: bool = True,
) -> np.ndarray:
    """Boolean starter mask over the given players.

    ``require_positive`` reflects that maximizing weight over a matroid takes a
    maximum-weight *independent set*, not a basis: a negative-value player is
    never started even with an open slot. In the hot path the selection input is
    ``rate_p50 * hazard * (bye != w)`` and is non-negative, so the rule is
    inert there — it matters for the ILP-equivalence tests, whose random
    rosters carry arbitrary weights, and for DST projections, which can go
    negative (the points-allowed tier bottoms at -4).

    Ties are broken by index so the result is deterministic, which the CRN
    invariant depends on.
    """
    values = np.asarray(values, dtype="float64")
    positions = np.asarray(positions)
    if values.shape != positions.shape:
        raise ValueError(
            f"values {values.shape} and positions {positions.shape} differ"
        )

    mask = np.zeros(values.shape, dtype=bool)
    if values.size == 0:
        return mask

    oracle = build_oracle(plan)
    # stable sort on the negated value: descending by value, then by index
    order = np.argsort(-values, kind="stable")
    for i in order:
        if require_positive and values[i] <= 0:
            break  # sorted, so nothing later can qualify either
        position = str(positions[i])
        if oracle.can_add(position):
            oracle.add(position)
            mask[i] = True
    return mask


def lineup_points(
    selection_values: np.ndarray,
    realized_values: np.ndarray,
    positions: np.ndarray,
    plan: SlotPlan,
) -> float:
    """Select on ``selection_values``, score on ``realized_values``.

    The two-argument shape is the point: it makes the non-clairvoyance rule
    impossible to violate by accident at the call site.
    """
    mask = greedy_lineup(selection_values, positions, plan)
    return float(np.asarray(realized_values, dtype="float64")[mask].sum())
