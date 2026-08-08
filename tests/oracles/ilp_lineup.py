"""ILP lineup oracle — TEST USE ONLY (§15.3).

The v1 ILP is retired from the hot path; greedy over the transversal matroid
replaces it. It survives here as an independent implementation to check greedy
against, using scipy.optimize.milp directly rather than the v1 wrapper, so the
two share no code.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from src.core.config.slots import SlotPlan


def ilp_best_lineup(values: np.ndarray, positions: np.ndarray,
                    plan: SlotPlan) -> float:
    """Maximum total value of any feasible starter set.

    Assignment formulation: x[i, c] = 1 iff player i occupies slot copy c.
    Each player takes at most one copy; each copy takes at most one player.
    Inequality (not equality) constraints, so this returns a maximum-weight
    INDEPENDENT SET — matching greedy's positivity rule, which is what makes
    the comparison meaningful when weights can be negative.
    """
    n, k = len(values), len(plan.copies)
    if n == 0 or k == 0:
        return 0.0
    n_vars = n * k
    c = np.zeros(n_vars)
    ub = np.zeros(n_vars)
    for i in range(n):
        for j, elig in enumerate(plan.copies):
            idx = i * k + j
            if str(positions[i]) in elig:
                ub[idx] = 1.0
                c[idx] = -float(values[i])

    a_player = np.zeros((n, n_vars))
    for i in range(n):
        a_player[i, i * k:(i + 1) * k] = 1.0
    a_copy = np.zeros((k, n_vars))
    for j in range(k):
        a_copy[j, j::k] = 1.0

    res = milp(
        c=c,
        constraints=[LinearConstraint(a_player, 0, 1),
                     LinearConstraint(a_copy, 0, 1)],
        integrality=np.ones(n_vars),
        bounds=Bounds(lb=np.zeros(n_vars), ub=ub),
    )
    if not res.success:
        raise RuntimeError(f"ILP oracle failed: {res.message}")
    return float(-res.fun)
