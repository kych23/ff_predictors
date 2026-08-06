"""Sobol first-order and total indices (§15.5). OFFLINE ONLY.

The elasticity table (§13.4) perturbs one knob at a time and reports a local
slope. That is the right tool for a research budget and the wrong one for
"where does the variance in E[$] come from", because knobs interact: a wider
projection interval matters more when weekly variance is already high, and a
one-at-a-time sweep cannot see that at all.

Sobol indices decompose the variance instead:

    S_i   first-order — the share removed by learning knob i alone
    S_Ti  total       — the share remaining if EVERYTHING ELSE were known

`S_Ti - S_i` is the interaction. When that gap is large, one-at-a-time
elasticities are misleading and the two tables should be read together.

The Saltelli A/B/AB design needs `n * (k + 2)` model evaluations, where the
"model" here is a full season simulation. That is why this is offline: the
pick clock cannot afford it, and §15.5 says so explicitly.

**Sobol needs far more replications per evaluation than the elasticity table,
and the reason is structural.** §13.4 compares perturbed runs against a base
run under common random numbers, so the Monte-Carlo error shared between them
cancels in the difference. Sobol takes no differences against a common base —
each of the `n*(k+2)` points is its own absolute value — so the simulator's
residual MC error enters the variance directly and is attributed to
*interaction*, since it is the part no single knob explains.

Measured on the live 2026 board: at 256 replications per evaluation and n=512,
`Var(E[$])` across the Saltelli sample was 11.10 while a ±10% move on the
strongest knob shifts E[$] by roughly $2. Most of that variance was simulator
noise, and the estimator returned first-order indices near −0.3 — impossible
for a real decomposition, and exactly what `SobolResult.converged` exists to
catch. Raising `n` alone will not fix it; the replications per evaluation have
to rise too.

Estimators are Saltelli 2010 / Jansen, which are the low-bias forms:

    S_i  = (1/n) Σ f(B) · (f(AB_i) − f(A))            / Var
    S_Ti = (1/2n) Σ (f(A) − f(AB_i))²                 / Var
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd


#: An index outside this band, or a materially negative interaction, is a
#: finite-sample artifact rather than a finding. Both estimators are unbiased
#: but high-variance, and at small n they routinely produce S > 1 and
#: S_T < S — neither of which is possible for a real decomposition.
SANE_INDEX_RANGE = (-0.2, 1.2)
MAX_NEGATIVE_INTERACTION = -0.15


@dataclass(frozen=True)
class SobolResult:
    table: pd.DataFrame
    n_base: int
    n_evaluations: int
    variance: float

    @property
    def converged(self) -> bool:
        """Whether the numbers are internally consistent enough to read.

        Checked rather than assumed because the failure is silent: the table
        prints identically whether n was 48 or 4096, and only the values give
        it away.
        """
        lo, hi = SANE_INDEX_RANGE
        return bool(
            self.table["first_order"].between(lo, hi).all()
            and self.table["total"].between(lo, hi).all()
            and (self.table["interaction"] >= MAX_NEGATIVE_INTERACTION).all()
        )

    def describe(self) -> str:
        lines = [f"{self.n_evaluations} evaluations, base sample {self.n_base}, "
                 f"Var(E[$]) = {self.variance:.4f}"]
        for _, r in self.table.iterrows():
            lines.append(f"  {r['knob']:22s} S={r['first_order']:+.3f}  "
                         f"S_T={r['total']:+.3f}  "
                         f"interaction={r['interaction']:+.3f}")
        if not self.converged:
            lines.append("")
            lines.append("  NOT CONVERGED — DO NOT READ THESE NUMBERS.")
            lines.append("  A first-order index above 1, or a total index below")
            lines.append("  its own first-order index, is impossible for a real")
            lines.append("  decomposition. Both estimators are unbiased but")
            lines.append(f"  high-variance, and n={self.n_base} is not enough.")
            lines.append("  Raise the base sample until this warning clears.")
        return "\n".join(lines)


def saltelli_samples(n: int, bounds: Sequence[tuple[float, float]],
                     seed: int = 0) -> tuple[np.ndarray, np.ndarray, list]:
    """A, B and the k AB_i matrices.

    **AB_i is A with column i taken from B** — base A, one column swapped in.
    Building it the other way round (base B, column i from A) silently
    exchanges the first-order and total estimators: every remaining column
    differs from A instead of just one, so `f(A) - f(AB_i)` measures the
    complement of what the formula assumes. It produces entirely plausible
    numbers that mean the opposite thing, which is why
    `test_ab_takes_column_i_from_b` pins it at the matrix level. This
    implementation had the orientation backwards until that test ran: an
    ignored knob came back with a total index of 1.00.
    """
    rng = np.random.default_rng(seed)
    k = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    a = lo + (hi - lo) * rng.random((n, k))
    b = lo + (hi - lo) * rng.random((n, k))
    ab = []
    for i in range(k):
        mat = a.copy()
        mat[:, i] = b[:, i]
        ab.append(mat)
    return a, b, ab


def sobol_indices(knob_names: Sequence[str],
                  bounds: Sequence[tuple[float, float]],
                  evaluate: Callable[[np.ndarray], float],
                  *, n: int = 64, seed: int = 0) -> SobolResult:
    """`evaluate(knob_vector) -> E[$]`, called n*(k+2) times.

    `evaluate` must be deterministic given its input — the same seed root on
    every call. Without that, Monte-Carlo noise enters the differences and the
    indices absorb it as interaction, which reads as "everything interacts
    with everything" and is the classic way to make a Sobol table meaningless.
    """
    if len(knob_names) != len(bounds):
        raise ValueError("knob_names and bounds must be the same length")
    if n < 8:
        raise ValueError(f"base sample {n} is too small for a variance estimate")

    a, b, ab = saltelli_samples(n, bounds, seed)
    fa = np.array([evaluate(row) for row in a])
    fb = np.array([evaluate(row) for row in b])
    fab = [np.array([evaluate(row) for row in mat]) for mat in ab]

    variance = float(np.var(np.concatenate([fa, fb]), ddof=1))
    if variance <= 0:
        raise ValueError(
            "E[$] did not vary across the sample; either the bounds are "
            "degenerate or `evaluate` is ignoring its input"
        )

    rows = []
    for i, name in enumerate(knob_names):
        first = float(np.mean(fb * (fab[i] - fa)) / variance)
        total = float(0.5 * np.mean((fa - fab[i]) ** 2) / variance)
        rows.append({
            "knob": name,
            "first_order": first,
            "total": total,
            # Negative first-order indices are a finite-sample artifact, not a
            # bug: they mean the true index is near zero and n is small. Kept
            # rather than clipped, because clipping hides the diagnostic.
            "interaction": total - first,
        })

    table = pd.DataFrame(rows).sort_values("total", ascending=False,
                                           ignore_index=True)
    return SobolResult(table=table, n_base=n,
                       n_evaluations=n * (len(knob_names) + 2),
                       variance=variance)
