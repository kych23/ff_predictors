"""Pre-registered gate on the per-manager opponent model (§14.2, AD-9).

The layer costs a week and has the least data support in the design: after the
2025 ADP gap the cohort carries roughly 4.7 observations per parameter, and
three of eleven managers have one usable season or none. So the question
"is manager identity predictive at all?" is settled BEFORE the layer is built,
not after.

Two statistics, both permutation-based:

``permutation_p``
    Shuffle manager labels **within season** and refit. Under the null, who
    drafted which pick carries no information, so a shuffled fit should predict
    held-out picks as well as the real one. ``p`` is the fraction of shuffles
    whose held-out log-likelihood is at least the real fit's.

``max_statistic_p``
    ``max_m max_pos |beta_m,pos|`` against its permutation null. Averaging over
    eleven managers can bury one genuine outlier inside ten nulls; the max
    statistic lets a single real signal survive without eleven Bonferroni-
    crushed tests.

A null result is a deliverable. On failure the system ships league-mean plus
the slot covariate and reports the null with its own power statement — which is
a better artifact than an unvalidated feature.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from src.models.opponents.fit import (
    Choice, FitResult, fit_managers, league_mean_loglik,
)


@dataclass(frozen=True)
class GateResult:
    permutation_p: float
    max_statistic_p: float
    observed_heldout: float
    null_heldout_mean: float
    league_mean_heldout: float
    observed_max_beta: float
    n_choices: int
    n_managers: int
    n_permutations: int
    obs_per_parameter: float
    passed: bool
    thresholds: dict[str, float]

    def describe(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"gate {verdict}\n"
            f"  permutation p        {self.permutation_p:.3f}  "
            f"(threshold <= {self.thresholds['permutation']:.2f})\n"
            f"  max-statistic p      {self.max_statistic_p:.3f}  "
            f"(threshold <= {self.thresholds['max_statistic']:.2f})\n"
            f"  held-out loglik      per-manager {self.observed_heldout:.2f} vs "
            f"league-mean {self.league_mean_heldout:.2f}\n"
            f"  permutation null     mean {self.null_heldout_mean:.2f}\n"
            f"  data                 {self.n_choices} choices, "
            f"{self.n_managers} managers, "
            f"{self.obs_per_parameter:.1f} obs/parameter"
        )


def _heldout_loglik(choices: list[Choice], positions, *, seasons, **fit_kw) -> float:
    """Leave-one-season-out held-out log-likelihood.

    Within-draft holdout would not test what matters: whether a manager's
    fitted vector generalizes to a draft the model has never seen.
    """
    total = 0.0
    for season in seasons:
        train = [c for c in choices if c.season != season]
        test = [c for c in choices if c.season == season]
        if not train or not test:
            continue
        fit = fit_managers(train, positions, **fit_kw)
        total += fit.loglik_of(test)
    return total


def _max_abs_beta(fit: FitResult) -> float:
    return max(
        max(abs(v) for v in fm.beta.values())
        for fm in fit.managers.values()
    )


def run_gate(
    choices: list[Choice], positions: tuple[str, ...], *,
    n_permutations: int = 200, seed: int = 20260805,
    permutation_threshold: float = 0.10,
    max_statistic_threshold: float = 0.10,
    **fit_kw,
) -> GateResult:
    seasons = sorted({c.season for c in choices})
    managers = sorted({c.manager for c in choices})
    n_params = len(managers) * (1 + 2 * (len(positions) - 1))

    observed_heldout = _heldout_loglik(choices, positions, seasons=seasons, **fit_kw)
    observed_fit = fit_managers(choices, positions, **fit_kw)
    observed_max = _max_abs_beta(observed_fit)
    pooled = league_mean_loglik(choices, positions, **fit_kw)

    rng = random.Random(seed)
    null_heldout: list[float] = []
    null_max: list[float] = []
    by_season: dict[int, list[int]] = {}
    for i, c in enumerate(choices):
        by_season.setdefault(c.season, []).append(i)

    for _ in range(n_permutations):
        shuffled = list(choices)
        # Shuffle labels WITHIN season: a manager's slot and the board they
        # faced stay together, so only identity is scrambled.
        for _season, idxs in by_season.items():
            labels = [choices[i].manager for i in idxs]
            rng.shuffle(labels)
            for i, label in zip(idxs, labels):
                c = choices[i]
                shuffled[i] = Choice(label, c.season, c.overall, c.slot_z,
                                     c.adp, c.position_idx, c.chosen)
        null_heldout.append(
            _heldout_loglik(shuffled, positions, seasons=seasons, **fit_kw)
        )
        null_max.append(_max_abs_beta(fit_managers(shuffled, positions, **fit_kw)))

    null_heldout_arr = np.array(null_heldout)
    null_max_arr = np.array(null_max)
    # left tail: shuffles that predicted held-out picks at least as well
    perm_p = float((null_heldout_arr >= observed_heldout).mean())
    max_p = float((null_max_arr >= observed_max).mean())

    return GateResult(
        permutation_p=perm_p, max_statistic_p=max_p,
        observed_heldout=observed_heldout,
        null_heldout_mean=float(null_heldout_arr.mean()),
        league_mean_heldout=pooled, observed_max_beta=observed_max,
        n_choices=len(choices), n_managers=len(managers),
        n_permutations=n_permutations,
        obs_per_parameter=len(choices) / n_params if n_params else 0.0,
        passed=(perm_p <= permutation_threshold
                and max_p <= max_statistic_threshold),
        thresholds={"permutation": permutation_threshold,
                    "max_statistic": max_statistic_threshold},
    )
