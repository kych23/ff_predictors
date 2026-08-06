"""Waiver replacement floor (§15.6, §8.6).

**Never** ``max(realized, replacement)``. A max over realized production assumes
dropping a bust the instant it becomes one, with perfect timing and no cost —
the clairvoyance bug in different clothes, and it makes every late-round
lottery ticket identical. The detection lag is what distinguishes fliers from
each other: a flier who busts visibly costs less than one who lingers in an
ambiguous timeshare for eight weeks.

**Shape.** This returns a corrected ``team_week``, not masks. The drop decision
depends on observed points, which are per-replication, so the post-waiver roster
differs across replications — a ``(T, W, P)`` mask has no ``R`` axis and cannot
express that, and ``(T, W, P, R)`` would be 585 MB at production sizes. Outcomes
are instead a sparse swap list applied as an additive correction.

**Rest-of-season projection** is a single-k shrinkage, deliberately cruder than
the per-stat ``k`` deferred in §22.2, and self-contained here so this does not
depend on unbuilt work:

    ros_mu = (n_obs * mean(observed) + k * rate_p50) / (n_obs + k)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WaiverResult:
    team_week: np.ndarray        # (T, W, R) corrected
    n_swaps: int
    swaps_per_team: np.ndarray   # (T,) mean swaps per replication


def _baseline_by_position(bundle, pool_idx: np.ndarray) -> dict[str, float]:
    """Replacement level = mean projected value of the pool at each position.

    Set from the actual undrafted pool rather than a constant: with 72 bench
    spots across 12 teams the week-1 free-agent pool is thin and the
    replacement level is low, which is what raises the value of bench depth in
    this league specifically.
    """
    out: dict[str, float] = {}
    for pos in np.unique(bundle.positions[pool_idx]):
        at_pos = pool_idx[bundle.positions[pool_idx] == pos]
        out[str(pos)] = float(np.mean(bundle.rate_p50[at_pos])) if len(at_pos) else 0.0
    return out


def apply_waiver(
    points: np.ndarray, masks: np.ndarray, team_week: np.ndarray,
    rosters: list[np.ndarray], pool_idx: np.ndarray, bundle, *,
    detection_lag_weeks: int = 2, shrinkage_k_games: int = 6,
    regular_season_weeks: int = 14,
) -> WaiverResult:
    """Swap persistent underperformers for the best available pool player.

    One claim per team per week; the pool is shared, so a claimed player is
    gone for everyone, resolved in reverse standings order.
    """
    n_teams, weeks, reps = team_week.shape
    corrected = team_week.copy()
    baseline = _baseline_by_position(bundle, pool_idx)
    total_swaps = 0
    per_team = np.zeros(n_teams)

    for r in range(reps):
        claimed: set[int] = set()
        below = {t: {} for t in range(n_teams)}      # player -> consecutive weeks
        swapped: dict[int, dict[int, int]] = {t: {} for t in range(n_teams)}

        for w in range(1, regular_season_weeks):
            # standings through the prior week set the claim order
            standing = np.argsort(corrected[:, :w, r].sum(axis=1))
            for t in standing:                        # worst picks first
                roster = rosters[t]
                if len(roster) == 0:
                    continue
                # Average over GAMES PLAYED, not weeks elapsed. Byes and
                # inactive weeks are drawn as exact zeros, so dividing by w
                # biases ros_mu down by roughly the inactive rate and fires
                # spurious drops on players who were merely on bye. Per-game
                # rates are availability-neutral by construction; the games
                # missed belong to the hazard, not to the quality estimate.
                observed = points[roster, :w, r]
                played = (observed != 0).sum(axis=1)
                ros = ((observed.sum(axis=1) + shrinkage_k_games
                        * bundle.rate_p50[roster])
                       / (played + shrinkage_k_games))

                worst, worst_gap = None, 0.0
                for i, player in enumerate(roster):
                    if player in swapped[t]:
                        continue
                    floor = baseline.get(str(bundle.positions[player]), 0.0)
                    if ros[i] < floor:
                        below[t][player] = below[t].get(player, 0) + 1
                        gap = floor - ros[i]
                        if (below[t][player] >= detection_lag_weeks
                                and gap > worst_gap):
                            worst, worst_gap = player, gap
                    else:
                        below[t][player] = 0

                if worst is None:
                    continue
                pos = bundle.positions[worst]
                avail = [p for p in pool_idx
                         if p not in claimed and bundle.positions[p] == pos]
                if not avail:
                    continue
                best = max(avail, key=lambda p: bundle.rate_p50[p])
                claimed.add(best)
                swapped[t][worst] = best
                total_swaps += 1
                per_team[t] += 1

                # additive correction for the remaining weeks the dropped
                # player would have started
                started = masks[t, w:, worst] > 0
                if started.any():
                    delta = (points[best, w:, r] - points[worst, w:, r]) * started
                    corrected[t, w:, r] += delta.astype(corrected.dtype)

    return WaiverResult(team_week=corrected, n_swaps=total_swaps,
                        swaps_per_team=per_team / max(reps, 1))
