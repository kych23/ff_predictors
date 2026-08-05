"""Empirical slot correlation matrix (§13.0, §13.2, AD-2).

**There is no latent factor here, and that is a measured decision.** A
one-factor model forces ``corr(WR1, WR2) = lambda_WR^2``. Measured on
2012-2025 with full-PPR scoring:

======================  =======
QB1 x WR1               0.370
WR1 x WR2               0.037
RB1 x RB2              -0.139
======================  =======

Satisfying ``lambda_WR^2 = 0.037`` gives ``lambda_WR = 0.19``, and then
``corr(QB1, WR1) = 0.370`` requires ``lambda_QB = 1.92`` — impossible, since a
standardized loading cannot exceed 1. The factor model is *infeasible* on this
data, not merely inaccurate.

The mechanism is physical: a QB's yardage is the SUM of his receivers', so he
correlates with each of them while they cannibalize one another. One factor can
express the hub or the cannibalization, never both.

So the matrix is estimated directly and drawn from by Cholesky. That is
strictly simpler than the factor model it replaces: no EM, no identification
normalization, no sign ambiguity, no convergence triggers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

#: Offensive slots the matrix covers. DST couples to the OPPOSING offense and
#: is handled as a cross-block (§13.2.1), not a row here.
DEFAULT_SLOTS = ("QB1", "RB1", "RB2", "WR1", "WR2", "WR3", "TE1")

#: Below this, a pairwise correlation is too thin to trust and is shrunk to 0.
MIN_PAIR_OBSERVATIONS = 200


@dataclass(frozen=True)
class SlotCorrelation:
    """A PSD correlation matrix over slots, with its Cholesky factor cached."""

    slots: tuple[str, ...]
    matrix: np.ndarray          # (S, S)
    cholesky: np.ndarray        # (S, S) lower triangular
    pair_counts: np.ndarray     # (S, S) int — pairwise n, for provenance
    min_eigenvalue: float
    was_projected: bool
    source: str = "fitted"
    snapshot_id: str = ""
    model_version: str = ""

    def index_of(self, slot: str) -> int:
        return self.slots.index(slot)

    def draw(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """(size, S) correlated unit-variance normals."""
        z = rng.standard_normal((size, len(self.slots)))
        return z @ self.cholesky.T

    def correlation(self, a: str, b: str) -> float:
        return float(self.matrix[self.index_of(a), self.index_of(b)])


def _nearest_psd(matrix: np.ndarray, *, epsilon: float = 1e-8) -> np.ndarray:
    """Higham-style projection: clip eigenvalues, then renormalize diagonals.

    Pairwise-complete estimation does not guarantee positive definiteness —
    different cells use different subsets of the data. MEASURED: the real
    matrix has min eigenvalue 0.341 and needs no projection, so this branch is
    expected never to fire in production; it exists because "expected" is not
    "guaranteed".
    """
    sym = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(sym)
    values = np.clip(values, epsilon, None)
    rebuilt = vectors @ np.diag(values) @ vectors.T
    scale = np.sqrt(np.diag(rebuilt))
    rebuilt = rebuilt / np.outer(scale, scale)
    np.fill_diagonal(rebuilt, 1.0)
    return rebuilt


def fit_slot_correlation(
    panel: pd.DataFrame, *, slots: Sequence[str] = DEFAULT_SLOTS,
    snapshot_id: str = "", model_version: str = "",
) -> SlotCorrelation:
    """Estimate the matrix pairwise-complete from a slotted weekly panel.

    ``panel`` carries (season, week, team, slot, points) — the output of
    ``panel.assign_slots``.

    Pairwise-complete, NOT listwise: only ~2,600 team-games have all seven
    offensive slots populated, while pairwise n runs 5,500-6,200. Listwise
    deletion would discard more than half the data and bias toward healthy,
    fully-rostered teams — exactly the teams whose correlation structure is
    least representative.

    Points are standardized within (season, slot) before correlating, so the
    result is scale-free and a high-scoring era does not dominate.
    """
    slots = tuple(slots)
    wide = panel.pivot_table(
        index=["season", "week", "team"], columns="slot", values="points"
    )
    for slot in slots:
        if slot not in wide.columns:
            wide[slot] = np.nan
    wide = wide[list(slots)]

    # standardize within (season, slot): scale-free, era-neutral
    seasons = wide.index.get_level_values("season")
    standardized = wide.groupby(seasons).transform(
        lambda col: (col - col.mean()) / col.std(ddof=1) if col.std(ddof=1) else col * 0
    )

    size = len(slots)
    matrix = np.eye(size)
    counts = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(i + 1, size):
            pair = standardized.iloc[:, [i, j]].dropna()
            n = len(pair)
            counts[i, j] = counts[j, i] = n
            if n < MIN_PAIR_OBSERVATIONS:
                rho = 0.0   # too thin to trust; shrink rather than guess
            else:
                rho = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
                rho = 0.0 if np.isnan(rho) else rho
            matrix[i, j] = matrix[j, i] = rho

    eigenvalues = np.linalg.eigvalsh(matrix)
    min_eig = float(eigenvalues.min())
    projected = min_eig <= 1e-8
    if projected:
        matrix = _nearest_psd(matrix)
        min_eig = float(np.linalg.eigvalsh(matrix).min())

    return SlotCorrelation(
        slots=slots, matrix=matrix, cholesky=np.linalg.cholesky(matrix),
        pair_counts=counts, min_eigenvalue=min_eig, was_projected=projected,
        source="fitted", snapshot_id=snapshot_id, model_version=model_version,
    )


def from_prior(prior: dict, *, slots: Sequence[str] | None = None
               ) -> SlotCorrelation:
    """Load the checked-in measured matrix (``config/correlation_prior.yaml``).

    The fallback is a copy of measured data rather than a derivation from
    published constants. Earlier revisions derived fallback loadings through a
    units conversion and got the units wrong twice; a measured matrix has no
    conversion to get wrong.
    """
    slots = tuple(slots or prior["slots"])
    size = len(slots)
    matrix = np.eye(size)
    for i, a in enumerate(slots):
        for j, b in enumerate(slots):
            matrix[i, j] = float(prior["matrix"][a][b])
    matrix = (matrix + matrix.T) / 2.0
    eig = float(np.linalg.eigvalsh(matrix).min())
    return SlotCorrelation(
        slots=slots, matrix=matrix, cholesky=np.linalg.cholesky(matrix),
        pair_counts=np.zeros((size, size), dtype=int),
        min_eigenvalue=eig, was_projected=False, source="assumed",
    )
